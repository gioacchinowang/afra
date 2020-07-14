import afra as af
from afra.tools.ps_estimator import pstimator
import healpy as hp
import numpy as np
import pysm
import pysm.units as u
import camb


def main():
    """non-MPI
    
    record your input parameters
    prepare your singal, variance and mask maps
    """
    NSIDE = 128
    LMAX = 2*NSIDE
    NSAMP = 500  # global sampling size
    FREQ = [30, 95, 150, 353]
    NFREQ = len(FREQ)  # frequency band
    FWHMS = [0.5*np.pi/180., 0.3*np.pi/180., 0.2*np.pi/180., 0.08*np.pi/180.]  # FWHM in rad for beam effect
    APOSCALE = 6.
    PSBIN = 40  # number of angular modes per band power bin
    SHIFT_T = 10.  # CMB bandpower shift
    SHIFT_EB = 10.
    CUT_T = 1.  # CMB bandpower extraction threshold
    CUT_EB = 1.
    MODEL = ['s1','d1']  # pysm3 model name list
    
    # generate background
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.05)
    pars.set_for_lmax(4*NSIDE, lens_potential_accuracy=1);
    pars.WantTensors = True
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']
    ls = np.arange(totCL.shape[0])
    coefficient = (2.*np.pi)/ls/(ls+1)
    totCL[0] = 0.
    for i in range(1,len(ls)):
        totCL[i] *= coefficient[i]
    cmb_cl = np.transpose(totCL)
    cmbmap = hp.synfast(cmb_cl,nside=NSIDE,new=True,verbose=False)
    # generate foreground (+ background)
    sky = pysm.Sky(nside=NSIDE, preset_strings=MODEL)
    signalmaps = np.zeros((NFREQ,3,12*NSIDE**2))
    for i in range(NFREQ):
        maps = sky.get_emission(FREQ[i]*u.GHz)
        maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(FREQ[i]*u.GHz))
        signalmaps[i] = hp.smoothing(maps.value + cmbmap,fwhm=FWHMS[i],verbose=False)
    # import variance maps
    varmaps = np.zeros((NFREQ,3,12*NSIDE**2))
    varmaps[0] = hp.read_map('./data/plkvar_30.fits',field=[0,1,2],dtype=np.float32,verbose=0)
    varmaps[1] = hp.read_map('./data/alivar_95.fits',field=[0,1,2],dtype=np.float32,verbose=0)
    varmaps[2] = hp.read_map('./data/alivar_150.fits',field=[0,1,2],dtype=np.float32,verbose=0)
    varmaps[3] = hp.read_map('./data/plkvar_353.fits',field=[0,1,2],dtype=np.float32,verbose=0)
    # import mask map
    maskmap = hp.read_map('./data/ali_mask.fits',dtype=np.float32,verbose=0)
    
    """---------- don't worry about the following ----------"""
    
    fullmap = signalmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    fullvar = varmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    
    pipeline1 = af.abspipe(fullmap,mask=maskmap.reshape(1,-1),variances=fullvar,fwhms=FWHMS,lmax=LMAX)
    pipeline1.nsamp = NSAMP
    pipeline1.debug = True
    rslt_t = pipeline1.run(aposcale=APOSCALE,psbin=PSBIN,shift=SHIFT_T,threshold=CUT_T)
    
    fullmap = signalmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    fullvar = varmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    
    pipeline2 = af.abspipe(fullmap,mask=maskmap.reshape(1,-1),variances=fullvar,fwhms=FWHMS,lmax=LMAX)
    pipeline2.nsamp = NSAMP
    pipeline2.debug = True
    rslt_eb = pipeline2.run(aposcale=APOSCALE,psbin=PSBIN,shift=SHIFT_EB,threshold=CUT_EB)
    
    output = list()
    output.append(rslt_t[0])
    output.append(rslt_t[1])
    output.append(rslt_eb[1])
    output.append(rslt_eb[2])
    
    # calculate binned CMB band-power from input Cl
    est = pstimator(nside=NSIDE,mask=maskmap.reshape(1,-1),aposcale=APOSCALE,psbin=PSBIN)
    cmb_dl = np.zeros((3,len(est.modes)))
    for i in range(len(est.modes)):
        lrange = np.array(est._b.get_ell_list(i))
        factor = 0.5*lrange*(lrange+1)/np.pi
        w = np.array(est._b.get_weight_list(i))
        for j in range(3):
            cmb_dl[j,i] = np.sum(w*cmb_cl[j,lrange]*factor)
    output.append(cmb_dl[0])
    output.append(cmb_dl[1])
    output.append(cmb_dl[2])
    
    np.save('abs_example.npy',output)


if __name__ == '__main__':
    main()
