import abspy as ap
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


def main():
    """non-MPI version of tutorial 04
    
    record your input parameters
    prepare your singal, variance and mask maps
    """
    NSIDE = 128
    NSAMP = 200  # global sampling size
    NSAMP = max(2,NSAMP//mpisize)  # local sampling size
    NFREQ = 4  # frequency band
    FWHMS = [1.e-3]*NFREQ # beam FWHM each frequencies
    PSBIN = 40  # number of angular modes per band power bin
    SHIFT_T = 1.  # CMB bandpower shift
    SHIFT_EB = 1.
    CUT_T = 1.  # CMB bandpower extraction threshold
    CUT_EB = 1.
    
    # import data
    signalmaps = np.zeros((NFREQ,3,12*NSIDE**2))
    signalmaps[0] = hp.read_map('./data/TQU_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    signalmaps[1] = hp.read_map('./data/TQU_95GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    signalmaps[2] = hp.read_map('./data/TQU_150GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    signalmaps[3] = hp.read_map('./data/TQU_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    varmaps = np.zeros((NFREQ,3,12*NSIDE**2))
    varmaps[0] = hp.read_map('./data/TQU_var_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    varmaps[1] = hp.read_map('./data/TQU_var_95GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    varmaps[2] = hp.read_map('./data/TQU_var_150GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    varmaps[3] = hp.read_map('./data/TQU_var_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    maskmap = hp.read_map('./data/ali_mask_r7.fits',dtype=bool,verbose=False)
    
    cmbmap = hp.read_map('./data/TQU_CMB_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    """---------- don't worry about the following ----------"""
    
    fullmap = signalmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    fullvar = varmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    
    pipeline1 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=1,nside=NSIDE,mask=maskmap.reshape(1,-1),variance=fullvar,fwhms=FWHMS)
    pipeline1.nsamp = NSAMP
    rslt_t = pipeline1.run(psbin=PSBIN)
    
    fullmap = signalmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    fullvar = varmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    
    pipeline2 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=2,nside=NSIDE,mask=maskmap.reshape(1,-1),variance=fullvar,fwhms=FWHMS)
    pipeline2.nsamp = NSAMP
    rslt_eb = pipeline2.run(psbin=PSBIN)
    
    output = np.zeros((7,len(rslt_t[0])))
    output[0] = rslt_t[0]
    output[1] = np.mean(rslt_t[1],axis=0)
    output[2] = np.std(rslt_t[2],axis=0)
    output[3] = np.mean(rslt_eb[1],axis=0)
    output[4] = np.std(rslt_eb[2],axis=0)
    output[5] = np.mean(rslt_eb[3],axis=0)
    output[6] = np.std(rslt_eb[4],axis=0)
    
    np.save('abs_example.npy',output)
    
    est = ap.pstimator(nside=NSIDE,mask=maskmap.reshape(1,-1), aposcale=5.0, psbin=PSBIN)
    auto_cmb_t = est.auto_t(cmbmap[0].reshape(1,-1),fwhms=FWHMS[0])
    auto_cmb_eb = est.auto_eb(cmbmap[1:].reshape(2,-1),fwhms=FWHMS[0])
    
    output[0] = binned_ell
    output[1] = list(auto_cmb_t[0])
    output[2] = list(auto_cmb_eb[1])
    output[3] = list(auto_cmb_eb[2])
    
    np.save('abs_cmb.npy',output[:4])
        

if __name__ == '__main__':
    main()
