import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import abspy as ap
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


NSIDE = 128
freq_list = [95,150,200]  # measurements
nfreq = len(freq_list)

mapcmb = hp.read_map('./data/TQU_CMB_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
map30 = hp.read_map('./data/TQU_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
map353 = hp.read_map('./data/TQU_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
mask = hp.read_map('./data/ali_mask_r7.fits',dtype=bool,verbose=0)

fig = plt.figure(figsize=(5,5))
hp.mollview(mask,title='AliCPT mask map',hold=True)

est = ap.pstimator(nside=NSIDE,mask=mask.reshape(1,-1), aposcale=5.0, psbin=100)
modes, dustamp = est.auto_t(map353[0].reshape(1,-1))[:2]
modes, syncamp = est.auto_t(map30[0].reshape(1,-1))[:2]
modes, cmbamp = est.auto_t(mapcmb[0].reshape(1,-1))[:2]

# mock foreground from model
from abspy.tools.fg_models import syncdustmodel
c = syncdustmodel(modes=list(modes),refs=[30.,353.])
# update dust PS template to model parameters
for i in range(len(modes)):
    c.reset({c.param_list[i] : syncamp[i]})
    c.reset({c.param_list[i+len(modes)] : dustamp[i]})
c.reset({'beta_d':1.})
c.reset({'beta_s':-2.})
c.reset({'rho':0.3})
c.params
hybrid_ps = c.bandpower(freq_list)

# foreground + CMB
signal = np.zeros((len(modes),nfreq,nfreq))
for i in range(len(modes)):
    for j in range(nfreq):
        for k in range(nfreq):
            signal[i,j,k] = cmbamp[i]
signal += hybrid_ps

fig,ax = plt.subplots(figsize=(5,5))
for j in range(len(freq_list)):
    ax.scatter(modes,hybrid_ps[:,j,j],label=str(freq_list[j])+' GHz')
    ax.plot(modes,signal[:,j,j],label=str(freq_list[j])+' GHz')
    
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$\ell$',fontsize=15)
ax.set_ylabel(r'$D_\ell$ ($\mu K^2$)',fontsize=15)
ax.tick_params(axis='both', labelsize=15)
ax.set_title('auto-corr. at each frequency band')
ax.legend(loc=2)

fig,ax = plt.subplots(figsize=(5,5))
for i in range(len(modes)):
    ax.scatter(freq_list,np.diag(hybrid_ps[i]),label=str(modes[i]))
    ax.scatter(freq_list,np.diag(signal[i]),marker='^')
    
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$\nu$',fontsize=15)
ax.set_ylabel(r'$D_\ell$ ($\mu K^2$)',fontsize=15)
ax.tick_params(axis='both', labelsize=15)
ax.set_title('auto-corr. at angular mode')
ax.legend(loc=2)

# noise and template
noise = np.ones_like(signal)*1.e-2

template = np.zeros((len(modes),2))
template[:,0] = syncamp
template[:,1] = dustamp

from abspy.methods.tpfit import tpfit
p1 = tpfit(signal=signal,noise=noise,freqs=freq_list,modes=list(modes),template=template,refs=[30.,353.])
p1.sampling_opt = {'n_iter_before_update': 10,
                   'n_live_points': 4000,
                   'verbose': True,
                   'max_iter': 10000,;
                   'resume': False}
results = p1.run()


