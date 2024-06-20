# This notebook uses data generated on an x64 workstation using the gfdl-ws site files and intel compiler,
# using
# ```
# module load ifort/11.1.073
# module load intel_compilers
# module use /home/sdu/publicmodules
# module load netcdf/4.2
# module load mpich2/1.5b1
# ```
# for the `build/intel/env` file and run-time environment.
# 

# This experiment has linear stratification for initial conditions and a surface cooling buoyancy flux with no wind stress. 
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)


bml_prog_z=scipy.io.netcdf_file('BML/prog_z.nc','r')
kpp_prog_z=scipy.io.netcdf_file('KPP/prog_z.nc','r')
epbl_prog_z=scipy.io.netcdf_file('EPBL/prog_z.nc','r')
bml_visc=scipy.io.netcdf_file('BML/visc.nc','r')
kpp_visc=scipy.io.netcdf_file('KPP/visc.nc','r')
epbl_visc=scipy.io.netcdf_file('EPBL/visc.nc','r')


t = bml_prog_z.variables['Time'][:]
zw = -bml_prog_z.variables['zw'][:]
zt = -bml_prog_z.variables['zt'][:]


plt.subplot(131);
plt.contourf(t[1:], zt[:24], bml_prog_z.variables['temp'][1:,:24,1,1].T, levels=numpy.arange(13.8,15.05,.05));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $\theta(z,t)$');
plt.subplot(132);
plt.contourf(t[1:], zt[:24], kpp_prog_z.variables['temp'][1:,:24,1,1].T, levels=numpy.arange(13.8,15.05,.05));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $\theta(z,t)$');
plt.subplot(133);
plt.contourf(t[1:], zt[:24], epbl_prog_z.variables['temp'][1:,:24,1,1].T, levels=numpy.arange(13.8,15.05,.05));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $\theta(z,t)$');


plt.subplot(121);
plt.plot(t[1:], bml_prog_z.variables['temp'][1:,0,1,1].T, label='BML');
plt.plot(t[1:], kpp_prog_z.variables['temp'][1:,0,1,1].T, label='KPP');
plt.plot(t[1:], epbl_prog_z.variables['temp'][1:,0,1,1].T, label='EPBL');
plt.legend(loc='lower left'); plt.xlabel('Time (days)'); plt.ylabel('SST ($\degree$C)');
plt.subplot(122);
plt.plot(t[:], bml_visc.variables['MLD_003'][:,1,1].T, label='BML');
plt.plot(t[:], kpp_visc.variables['MLD_003'][:,1,1].T, label='KPP');
plt.plot(t[:], epbl_visc.variables['MLD_003'][:,1,1].T, label='EPBL');
plt.legend(loc='upper left'); plt.xlabel('Time (days)'); plt.ylabel('MLD$_{0.03}$ (m)');


# <h1 align="center">MOM6 diagnostics for KPP single column cooling test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of KPP boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/CVMix SCM tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=40cm,1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'cooling_KPP'

fname_A1 = './KPP/visc_40cm.nc'
fname_B1 = './KPP/visc_1m.nc'
fname_C1 = './KPP/visc_10m.nc'
fname_D1 = './KPP/visc_CM4.nc'

fname_A2 = './KPP/prog_40cm.nc'
fname_B2 = './KPP/prog_1m.nc'
fname_C2 = './KPP/prog_10m.nc'
fname_D2 = './KPP/prog_CM4.nc'


deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -110
secday = 86400

dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_A = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_A = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_A = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_A = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_A = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_A2,'r')
   
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_B = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_B = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_B = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_B = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_B = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_B2,'r')
   
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_C = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_C = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_C = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_C = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_C = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
   
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_D = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_D = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_D = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_D = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_D = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
   
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# KPP diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_KPP_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# Total diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kd_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kd_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kd_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kd_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.14)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_Kd_interface.png'
fig.savefig(fname,dpi=dpi);



# KPP non-local transport  


figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = secday*KPP_dTdt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = secday*KPP_dTdt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = secday*KPP_dTdt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = secday*KPP_dTdt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_nonlocal_temp_tendency.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.8, vmax=0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.80, vmax=0.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.80, vmax=0.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.80, vmax=0.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-110,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'KPP boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')


ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-110,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((14,15))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);
























# <h1 align="center">MOM6 diagnostics for EPBL single column cooling test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of EPBL boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/CVMix SCM tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=40cm,1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'cooling_EPBL'

fname_A1 = './EPBL/visc_40cm.nc'
fname_B1 = './EPBL/visc_1m.nc'
fname_C1 = './EPBL/visc_10m.nc'
fname_D1 = './EPBL/visc_CM4.nc'

fname_A2 = './EPBL/prog_40cm.nc'
fname_B2 = './EPBL/prog_1m.nc'
fname_C2 = './EPBL/prog_10m.nc'
fname_D2 = './EPBL/prog_CM4.nc'


deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -110


dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')
for v in visc.variables: print(v)
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_A = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_A2,'r')
print(' ')
for v in prog.variables: print(v)

    
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_B = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_B2,'r')
    
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_C = visc.variables['ePBL_h_ML'][:,0,0]


# tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
    
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_D = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
    
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_EPBL_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.80, vmax=0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.80, vmax=0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.0, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.8, vmax=0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.8, vmax=0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=13.5, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-110,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'Boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-110,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((14,15))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);





















# <h1 align="center">MOM6/OM4 mass transports through straits and throughflows</h1> 
# 
# Uses Python/3.4.3
# 
# This notebook provides example calculations of the mass transport through various sections saved from OM4 simulations.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import netCDF4
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import glob


# data layout for a section is (t,z,y,x)


mks2Sv = 1.0/1e9   

path_static= '/archive/gold/datasets/OM4_025/mosaic.v20140610.unpacked/'
#path_static= '/archive/bls/ulm_201505_awg_v20150923_mom6sis2_2015.11.30/CM4_c96L32_am4g6_2000_sis2_lowmix_hycom1/gfdl.ncrc2-intel-prod-openmp/pp/ocean_annual/'
#path_static = '/archive/Bonnie.Samuels/ulm_201505_4dUpdates_awg_v20150923_mom6sis2_2015.11.12/CM4_c96L32_am4g6_2000_sis2/gfdl.ncrc2-intel-prod-openmp/pp/ocean_annual/'

static = netCDF4.Dataset(path_static+'ocean_static.nc')
for v in static.variables: print (v)
    
# geometric factors 
lon   = static.variables['geolon'][:]
lat   = static.variables['geolat'][:]
nomlon   = static.variables['xq'][:]
nomlat   = static.variables['yq'][:]

#wet   = static.variables['wet'][:,:].filled(0.)
wet   = static.variables['wet'][:]
area  = static.variables['areacello'][:]*wet
#depth = static.variables['deptho'][:]*wet

    


plt.figure(figsize=(16,10))
field = wet
#plt.pcolormesh(lon,lat,field)
plt.pcolormesh(nomlon,nomlat,field)
plt.colorbar()
plt.title('Lines for mass transport diagnostics',fontsize=24) 
plt.xlim(-300,60)
plt.ylim(-80,90)
plt.ylabel(r'Latitude[$\degree$N]',fontsize=24)
plt.xlabel(r'Longitude[$\degree$E]',fontsize=24)
plt.gca().set_axis_bgcolor('gray')
axis = plt.gca()

pth = '/archive/bls/ulm_201505_awg_v20150923_mom6sis2_2015.11.30/CM4_c96L32_am4g6_2000_sis2_lowmix_hycom1/gfdl.ncrc2-intel-prod-openmp/pp/'
for p in glob.glob(pth+'ocean_[A-Z]*'):
    #print(p)
    afile = glob.glob(p+'/ts/120hr/5yr/ocean_*')[0]
    varname=afile.split('.')[-2]
    rg = netCDF4.Dataset(afile)
    #for v in rg.variables: print(v,end=' ')
    eh = rg.variables[varname]
    dims = eh.dimensions
    sx = rg.variables[dims[-1]][:]
    sy = rg.variables[dims[-2]][:]
    plt.plot(sx+0*sy,0*sx+sy,'w')
plt.xlim(-300,60);plt.ylim(-80,90);


path = './'
Agulhas = netCDF4.Dataset(path+'ocean_Agulhas_section.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Agulhas.variables: print(v)

time          = Agulhas.variables['time'][:]*(1/365.0)
trans         = Agulhas.variables['umo'][:]
transA        = trans.sum(axis=1)
trans_Agulhas = mks2Sv*transA.sum(axis=1) 
trans_Agulhas_line = trans[0,0,:,0]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Agulhas_line.shape)
print(trans_Agulhas.shape)


path = './'
Barents = netCDF4.Dataset(path+'ocean_Barents_opening.1948010100-1952123123.umo.nc')

# need to save umo not vmo 

print()      
print("VARIABLES IN MAIN FILE")
for v in Barents.variables: print(v)

trans          = Barents.variables['umo'][:]
transA         = trans.sum(axis=1)
trans_Barents  = mks2Sv*transA.sum(axis=1) 
trans_Barents_line = trans[0,0,:,0]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Barents_line.shape)
print(trans_Barents.shape)


path = './'
Canada = netCDF4.Dataset(path+'ocean_Canadian_Archipelago.1948010100-1952123123.vmo.nc')

# need to save umo not vmo 

print()      
print("VARIABLES IN MAIN FILE")
for v in Canada.variables: print(v)
    
trans          = Canada.variables['vmo'][:]
transA         = trans.sum(axis=1)
trans_Canada  = mks2Sv*transA.sum(axis=2) 
trans_Canada_line = trans[0,0,0,:]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Canada_line.shape)
print(trans_Canada.shape)


path = './'
Denmark = netCDF4.Dataset(path+'ocean_Denmark_Strait.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Denmark.variables: print(v)

trans         = Denmark.variables['vmo'][:]
transA        = trans.sum(axis=1)
trans_Denmark = mks2Sv*transA.sum(axis=2) 
trans_Denmark_line = trans[0,0,0,:]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Denmark_line.shape)
print(trans_Denmark.shape)


path = './'
Drake  = netCDF4.Dataset(path+'ocean_Drake_Passage.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Drake.variables: print(v)
    
trans        = Drake.variables['umo'][:]
transA       = trans.sum(axis=1)
trans_Drake  = mks2Sv*transA.sum(axis=1) 
trans_Drake_line = trans[0,0,:,0]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Drake_line.shape)
print(trans_Drake.shape)


path = './'
English = netCDF4.Dataset(path+'ocean_English_Channel.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in English.variables: print(v) 

trans        = English.variables['umo'][:]
transA       = trans.sum(axis=1)
trans_English  = mks2Sv*transA.sum(axis=1) 
trans_English_line = trans[0,0,:,0]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_English_line.shape)
print(trans_English.shape)


path = './'
Faroe_Scot = netCDF4.Dataset(path+'ocean_Faroe_Scotland.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Faroe_Scot.variables: print(v) 

trans        = Faroe_Scot.variables['umo'][:]
transA       = trans.sum(axis=1)
trans_Faroe_Scot  = mks2Sv*transA.sum(axis=1) 
trans_Faroe_Scot_line = trans[0,0,:,0]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Faroe_Scot_line.shape)
print(trans_Faroe_Scot.shape)


path = './'
Florida = netCDF4.Dataset(path+'ocean_Florida_Bahamas.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Florida.variables: print(v) 

trans        = Florida.variables['vmo'][:]
transA       = trans.sum(axis=1)
trans_Florida  = mks2Sv*transA.sum(axis=2) 
trans_Florida_line = trans[0,0,0,:]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Florida_line.shape)
print(trans_Florida.shape)


path = './'
Fram = netCDF4.Dataset(path+'ocean_Fram_Strait.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Fram.variables: print(v) 

trans        = Fram.variables['vmo'][:]
transA       = trans.sum(axis=1)
trans_Fram   = mks2Sv*transA.sum(axis=2) 
trans_Fram_line = trans[0,0,0,:]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Fram_line.shape)
print(trans_Fram.shape)


path = './'
Gibraltar = netCDF4.Dataset(path+'ocean_Gibraltar_Strait.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Gibraltar.variables: print(v) 

trans        = Gibraltar.variables['umo'][:]
transA       = trans.sum(axis=1)
trans_Gibraltar   = mks2Sv*transA.sum(axis=1) 
trans_Gibraltar_line = trans[0,0,:,0]
print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Gibraltar_line.shape)
print(trans_Gibraltar.shape)


path = './'
Iceland_Faroe_U = netCDF4.Dataset(path+'ocean_Iceland_Faroe_U.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Iceland_Faroe_U.variables: print(v) 

transU     = Iceland_Faroe_U.variables['umo'][:]
transA     = transU.sum(axis=1)
trans_Iceland_Faroe_U = mks2Sv*transA.sum(axis=1) 
trans_Iceland_Faroe_U_line = transU[0,0,:,0]

print() 
print()
print(time.shape) 
print(transU.shape)
print(trans_Iceland_Faroe_U_line.shape)
print(trans_Iceland_Faroe_U.shape)

path = './'
Iceland_Faroe_V = netCDF4.Dataset(path+'ocean_Iceland_Faroe_V.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Iceland_Faroe_V.variables: print(v) 

transV     = Iceland_Faroe_V.variables['vmo'][:]
transA     = transV.sum(axis=1)
trans_Iceland_Faroe_V = mks2Sv*transA.sum(axis=2) 
trans_Iceland_Faroe_V_line = transV[0,0,0,:]

trans_Iceland_Faroe = trans_Iceland_Faroe_U + trans_Iceland_Faroe_V

print() 
print()
print(time.shape) 
print(transV.shape)
print(trans_Iceland_Faroe_V_line.shape)
print(trans_Iceland_Faroe_V.shape)


path = './'
Iceland_Norway = netCDF4.Dataset(path+'ocean_Iceland_Norway.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Iceland_Norway.variables: print(v) 

trans        = Iceland_Norway.variables['vmo'][:]
transA       = trans.sum(axis=1)
trans_Iceland_Norway = mks2Sv*transA.sum(axis=2) 
trans_Iceland_Norway_line = trans[0,0,0,:]

print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Iceland_Norway_line.shape)
print(trans_Iceland_Norway.shape)


path = './'
Indo = netCDF4.Dataset(path+'ocean_Indonesian_Throughflow.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Indo.variables: print(v) 

trans        = Indo.variables['vmo'][:]
transA       = trans.sum(axis=1)
trans_Indo = mks2Sv*transA.sum(axis=2) 
trans_Indo_line = trans[0,0,0,:]

print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Indo_line.shape)
print(trans_Indo.shape)


path = './'
Moz = netCDF4.Dataset(path+'ocean_Mozambique_Channel.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Moz.variables: print(v) 

trans        = Moz.variables['vmo'][:]
transA       = trans.sum(axis=1)
trans_Moz = mks2Sv*transA.sum(axis=2) 
trans_Moz_line = trans[0,0,0,:]

print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Moz_line.shape)
print(trans_Moz.shape)


path = './'
PEUC = netCDF4.Dataset(path+'ocean_Pacific_undercurrent.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in PEUC.variables: print(v) 

trans        = PEUC.variables['umo'][:]
transA       = trans.sum(axis=1)
trans_PEUC = mks2Sv*transA.sum(axis=1) 
trans_PEUC_line = trans[0,0,:,0]

print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_PEUC_line.shape)
print(trans_PEUC.shape)


path = './'
Taiwan = netCDF4.Dataset(path+'ocean_Taiwan_Luzon.1948010100-1952123123.umo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Taiwan.variables: print(v) 

trans        = Taiwan.variables['umo'][:]
transA       = trans.sum(axis=1)
trans_Taiwan = mks2Sv*transA.sum(axis=1) 
trans_Taiwan_line = trans[0,0,:,0]

print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Taiwan_line.shape)
print(trans_Taiwan.shape)


path = './'
Windward = netCDF4.Dataset(path+'ocean_Windward_Passage.1948010100-1952123123.vmo.nc')

print()      
print("VARIABLES IN MAIN FILE")
for v in Windward.variables: print(v) 

trans        = Windward.variables['vmo'][:]
transA       = trans.sum(axis=1)
trans_Windward = mks2Sv*transA.sum(axis=2) 
trans_Windward_line = trans[0,0,0,:]

print() 
print()
print(time.shape) 
print(trans.shape)
print(trans_Windward_line.shape)
print(trans_Windward.shape)


# for easy setup of subplots
def newSP(y,x):
    global __spv, __spi ; __spv = (y,x) ; __spi = 1 ; plt.subplot(__spv[0], __spv[1], __spi)
def nextSP():
    global __spv, __spi ; __spi = __spi + 1 ; plt.subplot(__spv[0], __spv[1], __spi)


plt.figure(figsize=(16,12))
newSP(3,3)

plt.subplot(331);
plt.plot(time, trans_Agulhas);
plt.title(r'Agulhas',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(332);
plt.plot(time, trans_Barents);
plt.title(r'Barents Opening',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(333);
plt.plot(time, trans_Canada);
plt.title(r'Canadian Archepelago',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(334);
plt.plot(time, trans_Denmark);
plt.title(r'Denmark Strait',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(335);
plt.plot(time, trans_Drake);
plt.title(r'Drake Passage',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(336);
plt.plot(time, trans_English);
plt.title(r'English Channel',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(337);
plt.plot(time, trans_Faroe_Scot);
plt.title(r'Faroe-Scotland',fontsize=18)
plt.xlabel('Time (years)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(338);
plt.plot(time, trans_Florida);
plt.title(r'Florida-Bahamas',fontsize=18)
plt.xlabel('Time (years)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(339);
plt.plot(time, trans_Fram);
plt.title(r'Fram Strait',fontsize=18)
plt.xlabel('Time (years)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);







plt.figure(figsize=(16,10))
newSP(3,3)

plt.subplot(331);
plt.plot(time, trans_Gibraltar);
plt.title(r'Gibraltar Strait',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(332);
plt.plot(time, trans_Iceland_Faroe);
plt.title(r'Iceland-Faroe',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(333);
plt.plot(time, trans_Iceland_Norway);
plt.title(r'Iceland-Norway',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(334);
plt.plot(time, trans_Indo);
plt.title(r'Indonesian Throughflow',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(335);
plt.plot(time, trans_Moz);
plt.title(r'Mozambique Channel',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(336);
plt.plot(time, trans_PEUC);
plt.title(r'Pacific Equatorial Undercurrent',fontsize=18)
#plt.xlabel('Time (days)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(337);
plt.plot(time, trans_Taiwan);
plt.title(r'Taiwan-Luzon Strait',fontsize=18)
plt.xlabel('Time (years)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);

plt.subplot(338);
plt.plot(time, trans_Windward);
plt.title(r'Windward Passage',fontsize=18)
plt.xlabel('Time (years)',fontsize=18)
plt.ylabel('Transport (Sv)',fontsize=18);







# This "flow downslope" example involves four sub-directories, `layer`, `rho`, `sigma` and `z`, in which the model is running in one of four coordinate configurations. To  use this notebook it is assumed you have run each of those experiments in place and have kept the output generated in default configuration.
# 
# ## CAVEAT: This is a tutorial of how to make vertical section plots which also illustrates some poor ways to plot data for comparison. Read through to the end.
# 
# We will use matplotlib. The line `%pylab inline` loads all the necessary packages, including `numpy` and causes images to appear in the page. We will use `scipy`'s netcdf package to read MOM6 output. Note that this only works if MOM6 is compiled with `NETCDF=3`.
# 
# To see this notebook with figures see https://gist.github.com/adcroft/dde8d3fafd77d0caaa5613e64f1d7eff.
# 

get_ipython().magic('pylab inline')
import scipy.io.netcdf


# ### Accessing MOM6 output
# 
# Let's open some MOM6 output files and see what variables we diagnosed. We'll use `scipy.io.netcdf_file` but you could also use the `netCDF4` package and `netCDF4.Dataset` with the same syntax. Both functions return objects with members (functions or dictionaries) such as `.variables`:
# 

layer_file = scipy.io.netcdf_file('layer/prog.nc')
rho_file = scipy.io.netcdf_file('rho/prog.nc')
sigma_file = scipy.io.netcdf_file('sigma/prog.nc')
z_file = scipy.io.netcdf_file('z/prog.nc')
for v in layer_file.variables:
    print(v,layer_file.variables[v].shape,layer_file.variables[v].long_name)


# The above output is the result of this `diag_table`:
# 

get_ipython().system('head -15 layer/diag_table')


# We actually only asked for "e", "h", "u", "v", "salt" and "temp" in the `diag_table`. The other variables (such as "xh", "yq", etc) are all 1-dimensional "axis" variables associated with netcdf dimensions. This is a consequence of the CF convention used in MOM6 output.
# 
# ### Coordinates to use for plotting
# 
# Because MOM6 uses curvilinear coordinates in the horizontal, the axis variables for the horizontal dimensions are not always useful. 
# 
# This particular experiment *is* in Cartesian coordinates so we *can* use the variables "xh" for the h-point locations and "xq" for the u-points. In this instance we can read the 1D axis variables to use as coordinates:
# 

# Use the CF dimension-variable as the horizontal coordinate
xh = layer_file.variables['xh'][:] # This is the coordinate of the cell centers (h-points in 1D)
xq = layer_file.variables['xq'][:] # This is the coordinate of the cell corners (u-points in 1D)
xq = numpy.concatenate(([2*xq[0]-xq[1]],xq)) # Inserts left most edge of domain in to u-point coordinates


# In addition to output requested in the `diag_table`, MOM6 normally writes a files with static coordinates and metrics to `ocean_geometry.nc`:
# 

geom_file = scipy.io.netcdf_file('layer/ocean_geometry.nc')
for v in geom_file.variables:
    print(v,geom_file.variables[v].shape,geom_file.variables[v].long_name)


# Here, you'll see some 2D coordinate variables such as `geolon` and `geolonb` that facilitates plotting when the coordinates are curvilinear. We can check that the data is Cartesian and matches the CF dimensions-variable we read earlier with
# 

plt.plot( xh, '.', label='xh (CF 1D)');
plt.plot( geom_file.variables['lonh'][:].T, '.', label='lonh (1D)');
plt.plot( geom_file.variables['geolon'][:].T, '.', label='geolon (2D)');
plt.legend(loc='lower right');


# The above plot just confirms that we have the same values in numerous forms of the horizontal coordinates (for this experiment). In general, to make plan view plots for a 3D configuration you will always what to use the 2D coordinate data such as that in `ocean_geometry.nc` but here we will be making vertical section plots which only require the 1D form of horizontal coordinates.
# 
# ### Reading bottom depth from `ocean_geometry.nc`
# 
# Now let's plot the topography, which is contained in the variable `D` of `ocean_geometry.nc`.
# 

plt.plot( geom_file.variables['D'][0,:]); plt.title('Depth');


# "Depth" is a positive quantity for the ocean; the bottom is at height $z=-D(x,y)$. The topography visualized for this experiment is a flat shallow shelf on the left, a flat deep ocean on the right, and a linear slope in between.
# 
# The "flow downslope" example is ostensibly a 2D configuration in x-z. One would expect the y-dimension to be equal to 1 but it is instead equal to 4. Each of the 4 j-slices of an array will contain the exact same data. This is because a dimension in MOM6 can not be reduced to below the width of a parallelization halo width, which is typically equal to 4 or more.
# 
# This quickly illustrates that the model state is identical along the j-axis:
# 

print("mean square |h(j=0)-h(j=3)|^2 =",
      (( layer_file.variables['h'][-1,:,0,:]-layer_file.variables['h'][-1,:,3,:] )**2).sum() )


# So from here on, we will use j=0 in all plots.
# 
# ### Exploring vertically distributed model output
# 
# Now let's look at some model data in multiple coordinate modes. We opened the output files above. The python variable `layer_file` is a handle to the netcdf file `layer/prog.nc` which is output using the traditional isopycnal (or stacked shallow water) mode of MOM6. The other models in this experiment are all ALE-mode emulating the z*-coordinate (python variable `z_file` for `z/prog.nc`), terrain-following sigma-coordinate (`sigma_file` for `sigma/prog.nc`), and continuous isopycnal coordinate (`rho_file` for `rho/prog.nc`).
# 
# The diagnosed variables in each of these modes was the same. However, some axis data changes meaning. For example, the vertical coordinate in layer mode is a "target density":
# 

print( layer_file.variables['zl'].long_name, layer_file.variables['zl'].units, layer_file.variables['zl'][:] )


# When the model is in ALE mode emulating a z* coordinate, then the vertical coordinate is height (although we report notional depth to aid ferret with plotting):
# 

print( z_file.variables['zl'].long_name, z_file.variables['zl'].units, z_file.variables['zl'][:] )


# Let's look at salinity in the first record written by the model in each of these four coordinates. We'll plot the raw data without coordinates, i.e. in index space.
# 

plt.figure(figsize=(12,6))
plt.subplot(221);
plt.pcolormesh( layer_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title('a) Layer mode S');
plt.subplot(222);
plt.pcolormesh( rho_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title(r'b) $\rho$-coordinate S');
plt.subplot(223);
plt.pcolormesh( sigma_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title(r'c) $\sigma$-coordinate S');
plt.subplot(224);
plt.pcolormesh( z_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title('d) z*-coordinate S');


# There is no topography apparent in the plots and the salinity structure is hard to make sense of! In layer mode there is not even any horizontal structure. This is because in layer mode density is homogeneous along a layer whereas in ALE mode density (in this case salinity) is allowed to vary along layers.
# 
# Plotting output from any model in index-space ignores the coordinates which determines where the data is physically located. The apparent absence of topography is a symptom of this. In MOM6, layers always contain data but layers have variable thickness which can even vanish. This is what thickness looks like for the above salinity panels:
# 

plt.figure(figsize=(12,6))
plt.subplot(221);
plt.pcolormesh( layer_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title('a) Layer mode h');
plt.subplot(222);
plt.pcolormesh( rho_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title(r'b) $\rho$-coordinate h');
plt.subplot(223);
plt.pcolormesh( sigma_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title(r'c) $\sigma$-coordinate h');
plt.subplot(224);
plt.pcolormesh( z_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title('d) z*-coordinate h');


# The simplest distribution to explain is the terrain-following $\sigma$-coordinate in which the layers are uniformly distributed in each column to fit the topography. Thus $h$ has no vertical structure in panel c. 
# 
# For the other panels it is important to remember that the k-index increase downward in the model; k=1 (fortran convention) or k=0 (python convention) is at the surface. Panel d has a region of uniform resolution (~100m) at low k which transitions to vanished thickness (~0) at some value of k in each column. You can sort of see the topography (blue region) upside down. Panels a and b look similar and have a lot of vanished regions with even surface layers being vanished.
# 
# Before making sense of these thickness distributions, let's check that the total thickness in each column looks like the topography:
# 

plt.plot( layer_file.variables['h'][0,:,0,:].sum(axis=0), label='Layer');
plt.plot( rho_file.variables['h'][0,:,0,:].sum(axis=0), label=r'$\rho$');
plt.plot( sigma_file.variables['h'][0,:,0,:].sum(axis=0), label=r'$\sigma$');
plt.plot( z_file.variables['h'][0,:,0,:].sum(axis=0), label='z*');
plt.legend(loc='lower right'); plt.title('Coloumn total thickness');


# We see that although the thickness distributions are quite different between each model, the total thickness of each column reflects the topography we plotted earlier based on the `ocean_geometry.nc` file.
# 
# The layer thickness almost provides enough information to calculate the actual position of quantities with which we could then make plots. The missing information is the absolute position of the top or bottom.
# 
# Interfaces delineate, or bound, layers. The thickness of a layer, $h_{i,j,k}$, is related to the absolute position of the interface above, $z_{i,j,k-\frac{1}{2}}$, and below, $z_{i,j,k+\frac{1}{2}}$ by
# $$
# h_{i,j,k} = z_{i,j,k-\frac{1}{2}} - z_{i,j,k+\frac{1}{2}} \;\;\; \forall \; k=1,2,\ldots,nk
# $$
# 
# where, by convention, integer-valued indices indicate layer-centered quantities and half-valued indices indicate interface-located quantities. Interface and and layer quantities are thus staggered in the vertical.
# 
# The diagnostic variable `e` is the absolute vertical position of model interfaces. Because half-integer indices are not meaningful in most copmuter languages, there is a offset convention as follows.
# 
# In FORTRAN:
# $$
# h(i,j,k) = e(i,j,k) - e(i,j,k+1) \;\;\; \forall \; k=1,2,\ldots,nk
# $$
# 
# where arrays indices normally start at 1.
# 
# In python
# $$
# h[k,j,i] = e[k,j,i] - e[k+1,j,i] \;\;\; \forall \; k=0,1,\ldots,nk-1
# $$
# 
# where array indices start at 0. We have also indicated the [k,j,i] order of indices that arises from reading data from an model-generated netcdf file.
# 
# Let's look at where the interfaces are by plotting a line for each interface (note the use of the transpose `.T` operator to get these lines plotted in the right direction):
# 

plt.figure(figsize=(12,6))
plt.subplot(221); plt.plot( layer_file.variables['e'][0,:,0,:].T); plt.title('a) Layer mode e');
plt.subplot(222); plt.plot( rho_file.variables['e'][0,:,0,:].T); plt.title(r'b) $\rho$-coordinate e');
plt.subplot(223); plt.plot( sigma_file.variables['e'][0,:,0,:].T); plt.title(r'c) $\sigma$-coordinate e');
plt.subplot(224); plt.plot( z_file.variables['e'][0,:,0,:].T); plt.title('d) z*-coordinate e');


# You can now begin to discern the nature of the coordinates in each mode. If we zoom in on the shelf-break region we will be able to more clearly see what each coordinate mode is doing.
# 

plt.figure(figsize=(12,6))
xl=5,12; yl=-1000,10
plt.subplot(221); plt.plot( layer_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title('a) Layer mode e');
plt.subplot(222); plt.plot( rho_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title(r'b) $\rho$-coordinate e');
plt.subplot(223); plt.plot( sigma_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title(r'c) $\sigma$-coordinate e');
plt.subplot(224); plt.plot( z_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title('d) z*-coordinate e');


# The $\sigma$-coordinate (c) always has `nk` layers with finite thickness (uniformly distributed) in each column. The $z$*-coordinate model (d) seems to have a variable number of layers but in fact the layers thicknesses vanish wherever the layer would be below the topography. The isopycnal coordinates, both in layer-mode (a) and ALE-mode (b), have on one thick layer on the shelf and fewer finite-thickness layers off-shelf than the other models. In these cases, there are vanished layers at both the top and bottom of the column.
# 
# So now we know the location of the interfaces we can presume the center of the layer is in between at $(e[k,j,i]+e[k+1,j,i])/2$. Let's use `contourf` to shade salinity at the layer centers. Note how we have to create a 2D "x" coordinate to pass to `contourf` since `contourf` expects both coordinate arrays to be 2D if either one of them is 2D. We do this by using an expression `x=xh+0*z` which uses `numpy`'s "broadcasting" feature (see http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html for explanation of rules). We will also plot the interface positions on top of the shaded contours:
# 

plt.figure(figsize=(12,6))
xxl=50,120 # This is the zoomed-in region around the shelf break in model coordinates
plt.subplot(221)
z = ( layer_file.variables['e'][0,:-1,0,:] + layer_file.variables['e'][0,1:,0,:] ) / 2
x = xh + 0*z
plt.contourf( x, z, layer_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title('a) Layer mode S');
plt.plot( xh, layer_file.variables['e'][0,:,0,:].T, 'k');
plt.subplot(222)
z = ( rho_file.variables['e'][0,:-1,0,:] + rho_file.variables['e'][0,1:,0,:] ) / 2
plt.contourf( x, z, rho_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title(r'b) $\rho$ coordinate S');
plt.plot( xh, rho_file.variables['e'][0,:,0,:].T, 'k');
plt.subplot(223)
z = ( sigma_file.variables['e'][0,:-1,0,:] + sigma_file.variables['e'][0,1:,0,:] ) / 2
plt.contourf( x, z, sigma_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title(r'c) $\sigma$ coordinate S');
plt.plot( xh, sigma_file.variables['e'][0,:,0,:].T, 'k');
plt.subplot(224)
z = ( z_file.variables['e'][0,:-1,0,:] + z_file.variables['e'][0,1:,0,:] ) / 2
plt.contourf( x, z, z_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title('d) z* coordinate S');
plt.plot( xh, z_file.variables['e'][0,:,0,:].T, 'k');


# The above looks closer to what one imagines things look like but **there are some very big problems with the above plots**.
# 
# 1) The apparent topography (white regions at bottom of plots) is quite different between the panels. This happens because `contourf` only shades between cell centers and so only half of the edge cells are plotted. `contourf` does not extrapolate beyond the coordinate provided for the data location. We loose a half cell of shading at the top and bottom of the column and also at the left and right of the plot. In the isopycnal-like coordinates, the bottom layer is thick and so we loose a lot.
# 
# 2) The shading within layers and between columns is interpolated which is introducing interior features and gradients which should not be there. The overlaid interface positions make this apparent for the layer mode for which salinity is absolutely constant along a layer (recall first plot of salinity).
# 
# To get the plot we want we essentially need to insert a layer of extra data at the top and bottom of the model and an extra column at both ends. To illustrate lets see how one might do this first for the "layer" output. We'll define a little function to help:
# 

def fix_contourf(nc_object, record, xh, variable='salt', clim=None, xl=None, yl=None, plot_grid=True):
    e = nc_object.variables['e'][record,:,0,:] # Interface positions
    z = ( e[:-1,:] + e[1:,:] ) / 2 # Layer centers
    S = nc_object.variables[variable][record,:,0,:] # Model output
    z = numpy.vstack( ( e[0,:], z, e[-1,:] ) ) # Add a layer at top and bottom
    S = numpy.vstack( ( S[0,:], S, S[-1,:] ) ) # Add layer data from top and bottom
    x = xh + 0*z
    plt.contourf( x, z, S );
    if clim is not None: plt.clim(clim);
    if plot_grid: plt.plot( xh, e.T, 'k');
    if xl is not None: plt.xlim(xl);
    if yl is not None: plt.ylim(yl);

plt.figure(figsize=(12,3))
# Same plot as above
plt.subplot(121)
z = ( layer_file.variables['e'][0,:-1,0,:] + layer_file.variables['e'][0,1:,0,:] ) / 2
x = xh + 0*z
plt.contourf( x, z, layer_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl);
plt.title('a) Layer mode S, as above');
plt.plot( xh, layer_file.variables['e'][0,:,0,:].T, 'k');
plt.clim(34,35)
# Now with an extra layer above and below
plt.subplot(122)
fix_contourf(layer_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title('b) Layer mode S, plotted with extra layers');


# So not the data appears to be plotted from the surface down to the topography. Using this approach for all the coordinates:
# 

plt.figure(figsize=(12,6))
xxl=50,120 # This is the zoomed-in region around the shelf break in model coordinates
plt.subplot(221); fix_contourf(layer_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title('a) Layer mode S')
plt.subplot(222); fix_contourf(rho_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'b) $\rho$ coordinate S');
plt.subplot(223); fix_contourf(sigma_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'c) $\sigma$ coordinate S');
plt.subplot(224); fix_contourf(z_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title('d) z* coordinate S');


# A remaining issue is why does there appear to be a salinity inversion in the z*-coordinate model. Technically there is (see salinity plots in i,k-space) but the layers are vanished so we should not be seeing them. This is because `contourf` is interpolating between thick and vanished layers. The bottom-line is that `contourf` is assuming the data is smooth and interpreting data inconsistent with the model formulation which considers the data to be piecewise.
# 
# ### Use `pcolormesh()` to visualize
# 
# The most consistent tool for visualizing piecewise data is `pcolormesh`. An important distinction between `contourf` and `pcolormesh` is that the latter takes the coordinates of the corners of cells when shading cell-centered values. Recall we loaded the coordinate `yq` and inserted an extra value on the left edge - we'll use that for the horizontal coordinate of cell edges. We will horizontally average to get an approximate position for the cell corner heights:
# 

def plot_with_pcolormesh(nc_object, record, xq, variable='salt', clim=None, xl=None, yl=None, plot_grid=True):
    e = nc_object.variables['e'][record,:,0,:] # Interface positions for h-columns
    ea = numpy.vstack( ( e[:,0].T, (e[:,:-1].T+e[:,1:].T)/2, e[:,-1].T ) ).T # Interface positions averaged to u-columns
    plt.pcolormesh( xq+0*ea, ea, nc_object.variables[variable][record,:,0,:] )
    if clim is not None: plt.clim(clim);
    if plot_grid: plt.plot( xq, ea.T, 'k');
    if xl is not None: plt.xlim(xl);
    if yl is not None: plt.ylim(yl);

plt.figure(figsize=(12,6))
xxl=50,120 # This is the zoomed-in region around the shelf break in model coordinates
plt.subplot(221); plot_with_pcolormesh(layer_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title('a) Layer mode S')
plt.subplot(222); plot_with_pcolormesh(rho_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'b) $\rho$ coordinate S');
plt.subplot(223); plot_with_pcolormesh(sigma_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'c) $\sigma$ coordinate S');
plt.subplot(224); plot_with_pcolormesh(z_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title('d) z* coordinate S');


# In the above plots, the vanished layers are reasonably hidden and the overall shading for salinity more similar between the plots.
# 
# The above method treats each cell as a trapezoid with corners shared between neighboring cells. It does not preserve the mean depth of the cell boundaries. To give more faithful rendering of the what the model can do, a tool is provided in `MOM6-examples/tools/analysis/m6toolbox` that returns arguments that can be passed straight to `pcolormesh` consistent with various interpretations of the grid structure, e.g. pcm (piecewise constant thicknesses), plm (piecewise linear), linear (as described above).
# 

# These next two lines add the MOM6-examples/tools/analysis/ directory to the search path for python packages
import sys
sys.path.append('../../tools/analysis/')
# m6toolbox is a python package that has a function that helps visualize vertical sections
import m6toolbox


# Define a function to plot a section
def plot_section(file_handle, record, xq, variable='salt', clim=None, xl=None, yl=None,  plot_grid=True, rep='pcm'):
    """Plots a section by reading vertical grid and scalar variable and super-sampling
    both in order to plot vertical and horizontal reconstructions.
    
    Optional arguments have defaults for plotting salinity and overlaying the grid.
    """
    e = file_handle.variables['e'][record,:,0,:] # Vertical grid positions
    s = file_handle.variables[variable][record,:,0,:] # Scalar field to color
    x,z,q = m6toolbox.section2quadmesh(xq, e, s, representation=rep) # This yields three areas at twice the model resolution
    plt.pcolormesh(x, z, q);
    if clim is not None: plt.clim(clim)
    if plot_grid: plt.plot(x, z.T, 'k', hold=True);
    if xl is not None: plt.xlim(xl)
    if yl is not None: plt.ylim(yl)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1); plot_section(layer_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='plm'); plt.title('a) Layer S');
plt.subplot(2,2,2); plot_section(rho_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='plm'); plt.title(r'b) $\rho$-coordinate S');
plt.subplot(2,2,3); plot_section(sigma_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='linear'); plt.title(r'c) $\sigma$-coordinate S');
plt.subplot(2,2,4); plot_section(z_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='pcm'); plt.title('d) z*-coordinate S');


# We will use `matplotlib.pyplot` for plotting and scipy's netcdf package for reading the model output. The `%pylab inline` causes figures to appear in the page and conveniently alias pyplot to `plt` (which is becoming a widely used alias).
# 
# This analysis assumes you changed `DAYMAX` to some multiple of 5 so that there are multiple time records in the model output.
# 
# To see this notebook with figures, see https://gist.github.com/adcroft/2a2b91d66625fd534372.
# 

get_ipython().magic('pylab inline')
import scipy.io.netcdf


# We first create a netcdf object, or "handle", to the netcdf file. We'll also list all the objects in the netcdf object.
# 

prog_file = scipy.io.netcdf_file('prog__0001_006.nc')
prog_file.variables


# Now we will create a variable object for the "e" variable in the file. Again, I'm labelling it as a handle to distinguish it from a numpy array or raw data.
# 
# We'll also look at an "attribute" and print the shape of the data.
# 

e_handle = prog_file.variables['e']
print('Description =', e_handle.long_name)
print('Shape =',e_handle.shape)


# "e" is 4-dimensional. netcdf files and objects are index [n,k,j,i] for the time-, vertical-, meridional-, zonal-axes.
# 
# Let's take a quick look at the first record [n=0] of the top interface [k=0]. 
# 

plt.pcolormesh( e_handle[0,0] )


# The data looks OKish. No scale! And see that "`<matplotlib...>`" line? That's a handle returned by the matplotlib function. Hide it with a semicolon. Let's add a scale and change the colormap.
# 

plt.pcolormesh( e_handle[0,0], cmap=cm.seismic ); plt.colorbar();


# We have 4D data but can only visualize by projecting on a 2D medium (the page). Let's solve that by going interactive!
# 

import ipywidgets


# We'll need to know the range to fix the color scale...
# 

[e_handle[:,0].min(), e_handle[:,0].max()]


# We define a simple function that takes the record number as an argument and then plots the top interface (k=0) for that record. We then use the `interact()` function to do some magic!
# 

def plot_ssh(record):
    plt.pcolormesh( e_handle[record,0], cmap=cm.spectral )
    plt.clim(-.5,.8) # Fixed scale here
    plt.colorbar()

ipywidgets.interact(plot_ssh, record=(0,e_handle.shape[0]-1,1));


# Unable to scroll the slider steadily enough? We'll use a loop to redraw for us...
# 

from IPython import display


for n in range( e_handle.shape[0]):
    display.display(plt.gcf())
    plt.clf()
    plot_ssh(n)
    display.clear_output(wait=True)


# <h1 align="center">MOM6 diagnostics for EPBL single column wind+warming test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of EPBL boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/CVMix SCM tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=40cm,1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'skin_warming_wind_EPBL'

fname_A1 = './EPBL/visc_40cm.nc'
fname_B1 = './EPBL/visc_1m.nc'
fname_C1 = './EPBL/visc_10m.nc'
fname_D1 = './EPBL/visc_CM4.nc'

fname_A2 = './EPBL/prog_40cm.nc'
fname_B2 = './EPBL/prog_1m.nc'
fname_C2 = './EPBL/prog_10m.nc'
fname_D2 = './EPBL/prog_CM4.nc'


deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -45


dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')
for v in visc.variables: print(v)
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_A = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_A2,'r')
print(' ')
for v in prog.variables: print(v)

    
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_B = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_B2,'r')
    
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_C = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
    
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_D = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
    
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.04)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.04)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.04)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.04)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_EPBL_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-45,-15))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'Boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-45,-15))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((15,16.8))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);





















# This is a quick illustration of the output from the four "flow_downslope" experiments which simulate a dense plug of water flowing down off a raised shelf in an ambiently stratified ocean. The four experiments use different vertical coordianes: a) Layer model, or stacked shallow water, b) Continuous isopycnal coordinates using the ALE-method, c) Terrain-following sigma-coordinates using the ALE-method, and d) Geopotential z*-coordinates using the ALE method.
# 

get_ipython().magic('pylab inline')
import scipy.io.netcdf


# These next two lines add the MOM6-examples/tools/analysis/ directory to the search path for python packages
import sys
sys.path.append('../../tools/analysis/')
# m6toolbox is a python package that has a function that helps visualize vertical sections
import m6toolbox


# Open the output from the four experiments
layer_file = scipy.io.netcdf_file('layer/prog.nc')
rho_file = scipy.io.netcdf_file('rho/prog.nc')
sigma_file = scipy.io.netcdf_file('sigma/prog.nc')
z_file = scipy.io.netcdf_file('z/prog.nc')


# Read the horizontal coordinate which is the same for all configurations 
xq = layer_file.variables['xq'][:] # This is the coordinate of the cell corners (u-points in 1D)
xq = numpy.concatenate(([0],xq)) # Inserts left most edge of domain in to coordinates


# Define a function to plot a section
def plot_section(file_handle, record, variable='salt', clim=(34.,35.), plot_grid=True, rep='linear'):
    """Plots a section of by reading vertical grid and scalar variable and super-sampling
    both in order to plot vertical and horizontal reconstructions.
    
    Optional arguments have defaults for plotting salinity and overlaying the grid.
    """
    e = file_handle.variables['e'][record,:,0,:] # Vertical grid positions
    s = file_handle.variables[variable][record,:,0,:] # Scalar field to color
    x,z,q = m6toolbox.section2quadmesh(xq, e, s, representation=rep) # This yields three areas at twice the model resolution
    plt.pcolormesh(x, z, q);
    #plt.clim(clim)
    if plot_grid: plt.plot(x, z.T, 'k', hold=True);
    plt.ylim(-4000,1)
    #plt.xlim(400,600)

record = -1 # Last record
plt.figure(figsize=(10,5))
plt.subplot(2,2,1); plot_section(layer_file, record); plt.title('Layer');
plt.subplot(2,2,2); plot_section(rho_file, record); plt.title(r'$\rho$');
plt.subplot(2,2,3); plot_section(sigma_file, record, plot_grid=True); plt.title(r'$\sigma$');
plt.subplot(2,2,4); plot_section(z_file, record); plt.title(r'$z^*$');


# <h1 align="center">MOM6 diagnostics for KPP single column BATS test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of KPP boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/single_column tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'BATS_KPP'

fname_A1 = './KPP/visc_40cm.nc'
fname_B1 = './KPP/visc_1m.nc'
fname_C1 = './KPP/visc_10m.nc'
fname_D1 = './KPP/visc_CM4.nc'

fname_A2 = './KPP/prog_40cm.nc'
fname_B2 = './KPP/prog_1m.nc'
fname_C2 = './KPP/prog_10m.nc'
fname_D2 = './KPP/prog_CM4.nc'

deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -250
secday = 86400

dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_A = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_A = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_A = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_A = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_A = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_A2,'r')
   
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_B = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_B = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_B = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_B = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_B = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_B2,'r')
   
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_C = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_C = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_C = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_C = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_C = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
   
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_D = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_D = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_D = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_D = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_D = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
   
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# KPP diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_KPP_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# Total diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kd_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kd_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kd_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kd_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_Kd_interface.png'
fig.savefig(fname,dpi=dpi);



# KPP non-local transport  


figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = secday*KPP_dTdt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = secday*KPP_dTdt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = secday*KPP_dTdt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = secday*KPP_dTdt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.25, vmax=.25)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/day$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_nonlocal_temp_tendency.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='lower right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-350,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'KPP boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='lower right')


ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-350,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((18,29))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);






# This notebook uses data generated on an x64 workstation using the gfdl-ws site files and intel compiler,
# using
# ```
# module load ifort/11.1.073
# module load intel_compilers
# module use /home/sdu/publicmodules
# module load netcdf/4.2
# module load mpich2/1.5b1
# ```
# for the `build/intel/env` file and run-time environment.
# 
# Use
# ```
# module swap python python/3.4.3
# ipython notebook
# ```
# to see/use this notebook.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)


bml_prog_z=scipy.io.netcdf_file('BML/prog_z.nc','r')
kpp_prog_z=scipy.io.netcdf_file('KPP/prog_z.nc','r')
epbl_prog_z=scipy.io.netcdf_file('EPBL/prog_z.nc','r')
bml_visc=scipy.io.netcdf_file('BML/visc.nc','r')
kpp_visc=scipy.io.netcdf_file('KPP/visc.nc','r')
epbl_visc=scipy.io.netcdf_file('EPBL/visc.nc','r')
bml_prog=scipy.io.netcdf_file('BML/prog.nc','r')


t = bml_prog_z.variables['Time'][:]
zw = -bml_prog_z.variables['zw'][:]
zt = -bml_prog_z.variables['zt'][:]


k=44; clevs=numpy.arange(16.5,28.25,.25)
plt.subplot(131);
plt.contourf(t[:], zt[:k], bml_prog_z.variables['temp'][:,:k,0,0].T, levels=clevs);
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $\theta(z,t)$');
plt.subplot(132);
plt.contourf(t[:], zt[:k], kpp_prog_z.variables['temp'][:,:k,0,0].T, levels=clevs);
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $\theta(z,t)$');
plt.subplot(133);
plt.contourf(t[:], zt[:k], epbl_prog_z.variables['temp'][:,:k,0,0].T, levels=clevs);
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $\theta(z,t)$');


plt.subplot(121);
plt.plot(t[:], bml_prog_z.variables['temp'][:,0,0,0].T, 'k', label='BML');
plt.plot(t[:], kpp_prog_z.variables['temp'][:,0,0,0].T, 'b', label='KPP');
plt.plot(t[:], epbl_prog_z.variables['temp'][:,0,0,0].T, 'r', label='EPBL');
plt.legend(loc='upper left'); plt.xlabel('Time (days)'); plt.ylabel('SST ($\degree$C)');
plt.subplot(122);
plt.plot(t[:], bml_prog_z.variables['salt'][:,0,0,0].T, 'k', label='BML');
plt.plot(t[:], kpp_prog_z.variables['salt'][:,0,0,0].T, 'b', label='KPP');
plt.plot(t[:], epbl_prog_z.variables['salt'][:,0,0,0].T, 'r', label='EPBL');
plt.legend(loc='lower left'); plt.xlabel('Time (days)'); plt.ylabel('SSS (ppt)');


plt.subplot(121);
plt.plot(t[:], bml_visc.variables['MLD_003'][:,0,0].T, 'k', label='BML');
plt.plot(t[:], kpp_visc.variables['MLD_003'][:,0,0].T, 'b', label='KPP');
plt.plot(t[:], epbl_visc.variables['MLD_003'][:,0,0].T, 'r', label='EPBL');
plt.legend(loc='upper right'); plt.xlabel('Time (days)'); plt.ylabel('MLD$_{0.03}$ (m)'); plt.title('Diagnosed MLD');
plt.subplot(122); k=20
plt.plot(t[:], bml_prog.variables['h_ML'][:,0,0].T, 'k', label='BML');
plt.plot(t[:], kpp_visc.variables['KPP_OBLdepth'][:,0,0], 'b', label='KPP');
plt.plot(t[:], epbl_visc.variables['ePBL_h_ML'][:,0,0].T, 'r', label='EPBL');
plt.ylim(0,350)
plt.legend(loc='upper right'); plt.xlabel('Time (days)'); plt.ylabel('h (m)'); plt.title('Active BL depths');


plt.subplot(131);n=90;dn=900;
plt.plot(bml_prog_z.variables['temp'][n::dn,:,0,0].T, zt, 'k:', label='BML');
plt.plot(kpp_prog_z.variables['temp'][n::dn,:,0,0].T, zt, 'b--', label='KPP');
plt.plot(epbl_prog_z.variables['temp'][n::dn,:,0,0].T, zt, 'r-', label='EPBL');
plt.legend(loc='upper left'); plt.xlabel(r'$\theta$ ($\degree$C)'); plt.ylabel('z (m)');
plt.subplot(132);
plt.plot(bml_prog_z.variables['salt'][n::dn,:,0,0].T, zt, 'k:', label='BML');
plt.plot(kpp_prog_z.variables['salt'][n::dn,:,0,0].T, zt, 'b--', label='KPP');
plt.plot(epbl_prog_z.variables['salt'][n::dn,:,0,0].T, zt, 'r-', label='EPBL');
plt.legend(loc='upper left'); plt.xlabel('SSS (ppt)'); plt.ylabel('z (m)');
plt.subplot(133);
plt.plot(bml_visc.variables['KPP_N2'][n::dn,:,0,0].T, bml_prog.variables['e'][n::dn,:,0,0].T, 'k:', label='BML');
plt.plot(kpp_visc.variables['KPP_N2'][n::dn,:,0,0].T, -kpp_visc.variables['zi'][:], 'b--', label='KPP');
plt.plot(epbl_visc.variables['KPP_N2'][n::dn,:,0,0].T, -epbl_visc.variables['zi'][:], 'r-', label='EPBL');
plt.xlim(-2.e-6,5.e-5); plt.ylim(-900,0)
plt.legend(loc='lower right'); plt.xlabel('$N^2$ ($s^{-2}$)'); plt.ylabel('z (m)');


get_ipython().run_cell_magic('bash', '', 'exec tcsh\ncd ../../..\nsource MOM6-examples/build/intel/env; module load git\nmake SITE=gfdl-ws FC=mpif77 CC=mpicc LD=mpif77 MPIRUN=mpirun MOM6-examples/ocean_only/single_column/{BML,KPP,EPBL}/timestats.intel -j')


# <h1 align="center">MOM6 diagnostics for EPBL single column BATS test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of EPBL boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/single_column tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'BATS_EPBL'

fname_A1 = './EPBL/visc_40cm.nc'
fname_B1 = './EPBL/visc_1m.nc'
fname_C1 = './EPBL/visc_10m.nc'
fname_D1 = './EPBL/visc_CM4.nc'

fname_A2 = './EPBL/prog_40cm.nc'
fname_B2 = './EPBL/prog_1m.nc'
fname_C2 = './EPBL/prog_10m.nc'
fname_D2 = './EPBL/prog_CM4.nc'

deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -250
secday = 86400

dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')
for v in visc.variables: print(v)
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_A = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kd_A = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_A2,'r')
print(' ')
for v in prog.variables: print(v)

    
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_B = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kd_B = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_B2,'r')
    
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_C = visc.variables['ePBL_h_ML'][:,0,0]


# tracer diffusivity as function of time and depth (m2/sec)
Kd_C = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
    
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_D = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kd_D = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
    
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.02, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# EPBL diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 EPBL $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 EPBL $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 EPBL $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 EPBL $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_KPP_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# Total diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kd_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kd_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kd_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kd_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_Kd_interface.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1, vmax=7.0)
plt.colorbar()
C = plt.contour(time, -depths, field.T, 8, linewidth=.02, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=18, vmax=27)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(10,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='lower right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-350,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'EPBL boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='lower right')


ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-350,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((18,29))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);






# This notebook uses data generated on an x64 workstation using the gfdl-ws site files and intel compiler,
# using
# ```
# module load ifort/11.1.073
# module load intel_compilers
# module use /home/sdu/publicmodules
# module load netcdf/4.2
# module load mpich2/1.5b1
# ```
# for the `build/intel/env` file and run-time environment.
# 

# This experiment has isothermal initial conditions and a non-penatrative warming buoyancy flux with constant wind stress. 
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)


bml_prog_z=scipy.io.netcdf_file('BML/prog_z.nc','r')
kpp_prog_z=scipy.io.netcdf_file('KPP/prog_z.nc','r')
epbl_prog_z=scipy.io.netcdf_file('EPBL/prog_z.nc','r')
bml_visc=scipy.io.netcdf_file('BML/visc.nc','r')
kpp_visc=scipy.io.netcdf_file('KPP/visc.nc','r')
epbl_visc=scipy.io.netcdf_file('EPBL/visc.nc','r')


t = bml_prog_z.variables['Time'][:]
zw = -bml_prog_z.variables['zw'][:]
zt = -bml_prog_z.variables['zt'][:]


plt.subplot(131);
plt.contourf(t[1:], zt[:16], bml_prog_z.variables['temp'][1:,:16,1,1].T, levels=numpy.arange(15,16.6,.1));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $\theta(z,t)$');
plt.subplot(132);
plt.contourf(t[1:], zt[:16], kpp_prog_z.variables['temp'][1:,:16,1,1].T, levels=numpy.arange(15,16.6,.1));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $\theta(z,t)$');
plt.subplot(133);
plt.contourf(t[1:], zt[:16], epbl_prog_z.variables['temp'][1:,:16,1,1].T, levels=numpy.arange(15,16.6,.1)); 
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $\theta(z,t)$');


plt.subplot(131);
plt.contourf(t[1:], zt[:19], bml_prog_z.variables['u'][1:,:19,1,1].T);
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $u(z,t)$');
plt.subplot(132);
plt.contourf(t[1:], zt[:19], kpp_prog_z.variables['u'][1:,:19,1,1].T);
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $u(z,t)$');
plt.subplot(133);
plt.contourf(t[1:], zt[:19], epbl_prog_z.variables['u'][1:,:19,1,1].T);
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $u(z,t)$');


plt.subplot(121);
plt.plot(t[1:], bml_prog_z.variables['temp'][1:,0,1,1].T, label='BML');
plt.plot(t[1:], kpp_prog_z.variables['temp'][1:,0,1,1].T, label='KPP');
plt.plot(t[1:], epbl_prog_z.variables['temp'][1:,0,1,1].T, label='EPBL');
plt.legend(loc='upper left'); plt.xlabel('Time (days)'); plt.ylabel('SST ($\degree$C)');
plt.subplot(122);
plt.plot(t[:], bml_visc.variables['MLD_003'][:,1,1].T, label='BML');
plt.plot(t[:], kpp_visc.variables['MLD_003'][:,1,1].T, label='KPP');
plt.plot(t[:], epbl_visc.variables['MLD_003'][:,1,1].T, label='EPBL');
plt.legend(loc='upper right'); plt.xlabel('Time (days)'); plt.ylabel('MLD$_{0.03}$ (m)');


# This notebook uses data generated on an x64 workstation using the gfdl-ws site files and intel compiler,
# using
# ```
# module load ifort/11.1.073
# module load intel_compilers
# module use /home/sdu/publicmodules
# module load netcdf/4.2
# module load mpich2/1.5b1
# ```
# for the `build/intel/env` file and run-time environment.
# 

# The experiment has a linear stratification for initial conditions and a fixed wind-stress and zero buoyancy fluxes.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)


bml_prog_z=scipy.io.netcdf_file('BML/prog_z.nc','r')
kpp_prog_z=scipy.io.netcdf_file('KPP/prog_z.nc','r')
epbl_prog_z=scipy.io.netcdf_file('EPBL/prog_z.nc','r')
bml_visc=scipy.io.netcdf_file('BML/visc.nc','r')
kpp_visc=scipy.io.netcdf_file('KPP/visc.nc','r')
epbl_visc=scipy.io.netcdf_file('EPBL/visc.nc','r')


t = bml_prog_z.variables['Time'][:]
zw = -bml_prog_z.variables['zw'][:]
zt = -bml_prog_z.variables['zt'][:]


plt.subplot(131);
plt.contourf(t[1:], zt[:20], bml_prog_z.variables['temp'][1:,:20,1,1].T, levels=numpy.arange(14.3,15.0,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $\theta(z,t)$');
plt.subplot(132);
plt.contourf(t[1:], zt[:20], kpp_prog_z.variables['temp'][1:,:20,1,1].T, levels=numpy.arange(14.3,15.0,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $\theta(z,t)$');
plt.subplot(133);
plt.contourf(t[1:], zt[:20], epbl_prog_z.variables['temp'][1:,:20,1,1].T, levels=numpy.arange(14.3,15.0,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $\theta(z,t)$');


plt.subplot(121);
plt.plot(t[1:], bml_prog_z.variables['temp'][1:,0,1,1].T, label='BML');
plt.plot(t[1:], kpp_prog_z.variables['temp'][1:,0,1,1].T, label='KPP');
plt.plot(t[1:], epbl_prog_z.variables['temp'][1:,0,1,1].T, label='EPBL');
plt.legend(loc='lower left'); plt.xlabel('Time (days)'); plt.ylabel('SST ($\degree$C)');
plt.subplot(122);
plt.plot(t[:], bml_visc.variables['MLD_003'][:,1,1].T, label='BML');
plt.plot(t[:], kpp_visc.variables['MLD_003'][:,1,1].T, label='KPP');
plt.plot(t[:], epbl_visc.variables['MLD_003'][:,1,1].T, label='EPBL');
plt.legend(loc='upper left'); plt.xlabel('Time (days)'); plt.ylabel('MLD$_{0.03}$ (m)');


# <h1 align="center">MOM6 diagnostics for KPP single column wind-only test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of KPP boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/CVMix SCM tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=40cm,1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'wind_only_KPP'

fname_A1 = './KPP/visc_40cm.nc'
fname_B1 = './KPP/visc_1m.nc'
fname_C1 = './KPP/visc_10m.nc'
fname_D1 = './KPP/visc_CM4.nc'

fname_A2 = './KPP/prog_40cm.nc'
fname_B2 = './KPP/prog_1m.nc'
fname_C2 = './KPP/prog_10m.nc'
fname_D2 = './KPP/prog_CM4.nc'


deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -70


dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_A = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_A = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_A = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_A = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_A = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_A2,'r')
   
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_B = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_B = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_B = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_B = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_B = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_B2,'r')
   
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_C = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_C = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_C = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_C = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_C = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
   
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_D = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_D = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_D = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_D = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_D = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
   
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# KPP diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_KPP_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# Total diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kd_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kd_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kd_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kd_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_Kd_interface.png'
fig.savefig(fname,dpi=dpi);



# KPP non-local transport  


figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = KPP_dTdt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = KPP_dTdt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = KPP_dTdt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = KPP_dTdt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_nonlocal_temp_tendency.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.2, vmax=.2)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.2, vmax=.2)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.2, vmax=.2)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.2, vmax=.2)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-70,-5))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'KPP boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-70,-5))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((14.65,15))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);





















# <h1 align="center">MOM6 diagnostics for EPBL single column wind-only test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of EPBL boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/CVMix SCM tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=40cm,1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'wind_only_EPBL'

fname_A1 = './EPBL/visc_40cm.nc'
fname_B1 = './EPBL/visc_1m.nc'
fname_C1 = './EPBL/visc_10m.nc'
fname_D1 = './EPBL/visc_CM4.nc'

fname_A2 = './EPBL/prog_40cm.nc'
fname_B2 = './EPBL/prog_1m.nc'
fname_C2 = './EPBL/prog_10m.nc'
fname_D2 = './EPBL/prog_CM4.nc'


deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -70


dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')
for v in visc.variables: print(v)
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_A = visc.variables['ePBL_h_ML'][:,0,0]

# tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_A2,'r')
print(' ')
for v in prog.variables: print(v)

    
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_B = visc.variables['ePBL_h_ML'][:,0,0]


# tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['Kd_interface'][:,:,0,0]

prog  = scipy.io.netcdf_file(fname_B2,'r')
    
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_C = visc.variables['ePBL_h_ML'][:,0,0]


# tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
    
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')
    
# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# mixed layer depth as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# boundary layer depth as function of time (metre)
h_D = visc.variables['ePBL_h_ML'][:,0,0]


# tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['Kd_interface'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
    
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.08)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.08)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.08)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.08)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_EPBL_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.3, vmax=.3)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.3, vmax=.3)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.3, vmax=.3)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.3, vmax=.3)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=14, vmax=15)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(13.5,15,.02), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-70,0))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'Boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)
ax.legend(fontsize=24)

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-70,-5))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((14.65,15))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);





















# <h1 align="center">MOM6 diagnostics for KPP single column skin warming+wind test case</h1> 
# 
#  Results from this notebook: 
# 1. Basic diagnostics of KPP boundary layer and prognostic fields, comparing various vertical resolution results. 
# 
# Assumptions regarding this notebook:
# 0. Use of Python 3 or more recent. 
# 1. This notebook is written for the MOM6-examples/ocean_only/CVMix SCM tests.  
# 2. This notebook makes use of four simulations, each with differing vertical grid spacing.
#    The uniform grid spacings are dz=40cm,1m,10m, enabled via NK=1000,400,40 inside MOM_inputs.
#    The nonuniform grid is based on the OM4 grid, enabled via setting NK=75, MAXIMUM_DEPTH=6500.0,
#    and ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz", where vgrid_75_2m.nc is located in 
#    MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT.
# 
# Hopes for the use of this notebook: 
# 1. To provide a starting point to document single column model tests;
# 2. To illustrate a self-contained iPython notebook of use for MOM6 analysis.  
# 
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

expt_name = 'skin_warming_wind_KPP'

fname_A1 = './KPP/visc_40cm.nc'
fname_B1 = './KPP/visc_1m.nc'
fname_C1 = './KPP/visc_10m.nc'
fname_D1 = './KPP/visc_CM4.nc'

fname_A2 = './KPP/prog_40cm.nc'
fname_B2 = './KPP/prog_1m.nc'
fname_C2 = './KPP/prog_10m.nc'
fname_D2 = './KPP/prog_CM4.nc'


deltaz_A = '40cm'
deltaz_B = '1m'
deltaz_C = '10m'
deltaz_D = 'CM4'

fname_deltaz_A = '_40cm'
fname_deltaz_B = '_1m'
fname_deltaz_C = '_10m'
fname_deltaz_D = '_CM4'

ymin = -45


dpi=200


visc = scipy.io.netcdf_file(fname_A1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_A = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_A = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_A = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_A = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_A = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_A = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_A = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_A2,'r')
   
# depth of cell interface     
zi_A = prog.variables['zi'][:]

# depth of cell center 
zl_A = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_A  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_A  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_A = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_B1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_B = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_B = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_B = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_B = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_B = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_B = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_B = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_B2,'r')
   
# depth of cell interface     
zi_B = prog.variables['zi'][:]

# depth of cell center 
zl_B = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_B  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_B  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_B = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_C1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_C = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_C = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_C = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_C = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_C = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_C = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_C = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_C2,'r')
   
# depth of cell interface     
zi_C = prog.variables['zi'][:]

# depth of cell center 
zl_C = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_C  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_C  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_C = prog.variables['temp'][:,:,0,0]


visc = scipy.io.netcdf_file(fname_D1,'r')

# recall data layout is (t,z,y,x)

time = visc.variables['Time'][:]

# KPP boundary layer depth as function of time (metre)
h_D = visc.variables['KPP_OBLdepth'][:,0,0]

# MLD as function of time (metre)
mld_D = visc.variables['MLD_003'][:,0,0]

# KPP tracer diffusivity as function of time and depth (m2/sec)
Kt_D = visc.variables['KPP_Kheat'][:,:,0,0]

# total tracer diffusivity as function of time and depth (m2/sec)
Kd_D = visc.variables['Kd_interface'][:,:,0,0]

# KPP velocity diffusivity as function of time and depth (m2/sec)
Ku_D = visc.variables['KPP_Kv'][:,:,0,0]

# surface (and penetrating) buoyancy flux, as used by [CVmix] KPP (m2/s3)
KPP_buoyFlux_D = visc.variables['KPP_buoyFlux'][:,:,0,0]

# Temperature tendency due to non-local transport of heat, as calculated by KPP (K/s)
KPP_dTdt_D = visc.variables['KPP_NLT_dTdt'][:,:,0,0]


prog  = scipy.io.netcdf_file(fname_D2,'r')
   
# depth of cell interface     
zi_D = prog.variables['zi'][:]

# depth of cell center 
zl_D = prog.variables['zl'][:]

# zonal velocity as function of time and depth
u_D  = prog.variables['u'][:,:,0,0]

# zonal velocity as function of time and depth
v_D  = prog.variables['v'][:,:,0,0]

# temperature as function of time and depth 
temp_D = prog.variables['temp'][:,:,0,0]


fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

plt.subplot(221)
data   = u_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = u_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = u_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = u_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-.1, vmax=0.1)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $u$ ($m/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_zonal_velocity.png'
fig.savefig(fname,dpi=dpi);


# KPP diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 KPP $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_KPP_diffusivity.png'
fig.savefig(fname,dpi=dpi);



# Total diffusivity  

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = Kd_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = Kd_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = Kd_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = Kd_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0.0, vmax=0.02)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 total $\kappa_{\Theta}$ ($m^2/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_Kd_interface.png'
fig.savefig(fname,dpi=dpi);



# KPP non-local transport  


figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = KPP_dTdt_A
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_A
deltaz = deltaz_A 
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
data   = KPP_dTdt_B
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
data   = KPP_dTdt_C
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
data   = KPP_dTdt_D
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zi_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=-1e-5, vmax=1e-5)
plt.colorbar()
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'Non-local($K/s$) with $\Delta z$ ='+deltaz,fontsize=24)
plt.xlabel('Time (days)',fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.tight_layout()
fname = expt_name+'_MOM6_nonlocal_temp_tendency.png'
fig.savefig(fname,dpi=dpi);



# temperature drift

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
data   = temp_A[:,:] - temp_A[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(222)
data   = temp_B[:,:] - temp_B[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(223)
data   = temp_C[:,:] - temp_C[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.subplot(224)
data   = temp_D[:,:] - temp_D[0,:]
field  = np.ma.masked_array(data, mask=[data==0.])
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=0, vmax=1.1)
plt.colorbar()
#C = plt.contour(time, -depths, field.T, 8, linewidth=.05, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.xlabel('Time (days)',fontsize=24)
plt.title(r'MOM6 $\Theta-\Theta(t=0)$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)

plt.tight_layout()
fname = expt_name+'_MOM6_temp_drift.png'
fig.savefig(fname,dpi=dpi);



# temperature 

figure(1)
fig = plt.figure(figsize=(16,10), dpi=dpi)

plt.subplot(221)
field = temp_A[:,:]
depths = zl_A
deltaz = deltaz_A
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(15,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(221)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(222)
field = temp_B[:,:]
depths = zl_B
deltaz = deltaz_B
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(15,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(222)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(223)
field = temp_C[:,:]
depths = zl_C
deltaz = deltaz_C
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
#CS = plt.pcolormesh(time, -depths, field.T)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(15,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(223)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)


plt.subplot(224)
field = temp_D[:,:]
depths = zl_D
deltaz = deltaz_D
CS = plt.pcolormesh(time, -depths, field.T, vmin=15, vmax=16.2)
plt.colorbar()
C = plt.contour(time, -depths, field.T, levels=numpy.arange(15,17,.05), linewidth=.5, colors='black')
plt.ylim((ymin,0))
plt.xlabel('Time (days)',fontsize=24)
plt.ylabel('z (m)',fontsize=24)
plt.title(r'MOM6 $\Theta$ ($\degree$C) w/ $\Delta z$ ='+deltaz,fontsize=24);
plot = fig.add_subplot(224)
plot.tick_params(axis='both', which='major', labelsize=18)
plot.tick_params(axis='both', which='minor', labelsize=18)



plt.tight_layout()
fname = expt_name+'_MOM6_temp.png'
fig.savefig(fname,dpi=dpi);


# Boundary layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = h_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = h_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = h_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = h_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='lower right')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-45,-15))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'KPP boundary layer depth from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_bldepth.png'
fig.savefig(fname,dpi=dpi);



# Mixed layer depth

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = mld_A
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_A)
field = mld_B
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_B)
field = mld_C
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_C)
field = mld_D
CS = plt.plot(time, -field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='lower right')


ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((-45,-15))

plt.grid()
plt.xlabel('Time (days)',fontsize=30)
plt.ylabel('z (m)',fontsize=30)
plt.title(r'MLD from MOM6',fontsize=30)
#ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=30)
#ax.set_xticklabels(["$%.1f$" % x for x in xticks], fontsize=30);

plt.tight_layout()
fname = expt_name+'_MOM6_mld.png'
fig.savefig(fname,dpi=dpi);



# SST

figure(1)
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

field = temp_A[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_A)
field = temp_B[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_B)
field = temp_C[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_C)
field = temp_D[:,0]
CS = plt.plot(time, field,'-',linewidth=3, label=deltaz_D)

ax.legend(fontsize=24,loc='upper left')

ax.tick_params('both', length=10, width=2, which='major',labelsize=20)
ax.tick_params('both', length=10, width=2, which='minor',labelsize=20)
plt.ylim((15,16.8))

plt.grid()
plt.ylabel(r'SST ($\degree$C)',fontsize=30)
plt.xlabel('Time (days)',fontsize=30)
plt.title(r'SST from MOM6',fontsize=30)

plt.tight_layout()
fname = expt_name+'_MOM6_SST.png'
fig.savefig(fname,dpi=dpi);





















# This notebook uses data generated on an x64 workstation using the gfdl-ws site files and intel compiler,
# using
# ```
# module load ifort/11.1.073
# module load intel_compilers
# module use /home/sdu/publicmodules
# module load netcdf/4.2
# module load mpich2/1.5b1
# ```
# for the `build/intel/env` file and run-time environment.
# 

# The experiment has a linear stratification for initial conditions and a fixed mechanical energy input (u*) with zero mean wind-stress and zero buoyancy fluxes.
# 

import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)


bml_prog_z=scipy.io.netcdf_file('BML/prog_z.nc','r')
kpp_prog_z=scipy.io.netcdf_file('KPP/prog_z.nc','r')
epbl_prog_z=scipy.io.netcdf_file('EPBL/prog_z.nc','r')
bml_visc=scipy.io.netcdf_file('BML/visc.nc','r')
kpp_visc=scipy.io.netcdf_file('KPP/visc.nc','r')
epbl_visc=scipy.io.netcdf_file('EPBL/visc.nc','r')


t = bml_prog_z.variables['Time'][:]
zw = -bml_prog_z.variables['zw'][:]
zt = -bml_prog_z.variables['zt'][:]


plt.subplot(131);
plt.contourf(t[1:], zt[:19], bml_prog_z.variables['temp'][1:,:19,1,1].T, levels=numpy.arange(14.4,15.06,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $\theta(z,t)$');
plt.subplot(132);
plt.contourf(t[1:], zt[:19], kpp_prog_z.variables['temp'][1:,:19,1,1].T, levels=numpy.arange(14.4,15.06,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $\theta(z,t)$');
plt.subplot(133);
plt.contourf(t[1:], zt[:19], epbl_prog_z.variables['temp'][1:,:19,1,1].T, levels=numpy.arange(14.4,15.06,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $\theta(z,t)$');


# Equation 10.53 from "Modeling an dprediciton of the upper layers of the ocean" by E.B. Kraus, 1977, is:
# $$h(t) = \left( \frac{12 {u^*}^3 m^* t}{N^2} \right)^{1/3}$$
# 

rho0 = 1000.; dRhodT=-0.255; dTdz= 0.01; g=9.80616; N2= -g/rho0*dRhodT*dTdz; print('N2 =',N2)
ustar = sqrt(0.05/rho0); print('u* = ',ustar)
mstar=0.3
plt.subplot(121);
plt.plot(t[1:], bml_prog_z.variables['temp'][1:,0,1,1].T, label='BML');
plt.plot(t[1:], kpp_prog_z.variables['temp'][1:,0,1,1].T, label='KPP');
plt.plot(t[1:], epbl_prog_z.variables['temp'][1:,0,1,1].T, label='EPBL');
plt.legend(loc='lower left'); plt.xlabel('Time (days)'); plt.ylabel('SST ($\degree$C)');
plt.subplot(122);
plt.plot(t[:], bml_visc.variables['MLD_003'][:,1,1].T, label='BML');
plt.plot(t[:], kpp_visc.variables['MLD_003'][:,1,1].T, label='KPP');
plt.plot(t[:], epbl_visc.variables['MLD_003'][:,1,1].T, label='EPBL');
plt.plot(t, (12.*(ustar**3.)*mstar*(86400.*t)/N2)**(1./3.), label='Niiler and Krauss, 1977');
plt.legend(loc='upper left'); plt.xlabel('Time (days)'); plt.ylabel('MLD$_{0.03}$ (m)');


