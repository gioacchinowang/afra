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
    NSAMP = 500  # global sampling size
    NFREQ = 4  # frequency band
    FWHMS = [1.e-3]*NFREQ # beam FWHM each frequencies
    APOSCALE = 6.
    PSBIN = 40  # number of angular modes per band power bin
    SHIFT_T = 30.  # CMB bandpower shift
    SHIFT_EB = 30.
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
    
    maskmap = hp.read_map('./data/ali_mask_r7.fits',dtype=np.float64,verbose=False)
    
    cmbmap = hp.read_map('./data/TQU_CMB_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    """---------- don't worry about the following ----------"""
    
    fullmap = signalmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    fullvar = varmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    
    pipeline1 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=1,nside=NSIDE,mask=maskmap.reshape(1,-1),variances=fullvar,fwhms=FWHMS)
    pipeline1.nsamp = NSAMP
    rslt_t = pipeline1.run(aposcale=APOSCALE,psbin=PSBIN,shift=SHIFT_T,threshold=CUT_T)
    
    fullmap = signalmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    fullvar = varmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    
    pipeline2 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=2,nside=NSIDE,mask=maskmap.reshape(1,-1),variances=fullvar,fwhms=FWHMS)
    pipeline2.nsamp = NSAMP
    rslt_eb = pipeline2.run(aposcale=APOSCALE,psbin=PSBIN,shift=SHIFT_EB,threshold=CUT_EB)
    
    output = np.zeros((10,len(rslt_t[0])))
    output[0] = rslt_t[0]
    output[1] = rslt_t[1]
    output[2] = rslt_t[2]
    output[3] = rslt_eb[1]
    output[4] = rslt_eb[2]
    output[5] = rslt_eb[3]
    output[6] = rslt_eb[4]
    
    est = ap.pstimator(nside=NSIDE,mask=maskmap.reshape(1,-1),aposcale=APOSCALE,psbin=PSBIN)
    auto_cmb_t = est.auto_t(cmbmap[0].reshape(1,-1),fwhms=FWHMS[0])
    auto_cmb_eb = est.auto_eb(cmbmap[1:].reshape(2,-1),fwhms=FWHMS[0])
    
    output[7] = auto_cmb_t[1]
    output[8] = auto_cmb_eb[1]
    output[9] = auto_cmb_eb[2]
    
    np.save('abs_example.npy',output)
        

if __name__ == '__main__':
    main()
