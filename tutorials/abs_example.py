import abspy as ap
import healpy as hp
import numpy as np


def main():
    """non-MPI
    
    record your input parameters
    prepare your singal, variance and mask maps
    """
    NSIDE = 128
    LMAX = 200
    NSAMP = 500  # global sampling size
    NFREQ = 4  # frequency band
    FWHMS = [0]*NFREQ # beam FWHM each frequencies
    APOSCALE = 6.
    PSBIN = 40  # number of angular modes per band power bin
    SHIFT_T = 10.  # CMB bandpower shift
    SHIFT_EB = 10.
    CUT_T = 1.  # CMB bandpower extraction threshold
    CUT_EB = 1.
    
    # import data
    signalmaps = np.zeros((NFREQ,3,12*NSIDE**2))
    signalmaps[0] = hp.read_map('./data/TQU_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    signalmaps[1] = hp.read_map('./data/TQU_95GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    signalmaps[2] = hp.read_map('./data/TQU_150GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    signalmaps[3] = hp.read_map('./data/TQU_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    varmaps = np.zeros((NFREQ,3,12*NSIDE**2))
    varmaps[0] = hp.read_map('./data/TQU_plkvar_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    varmaps[1] = hp.read_map('./data/TQU_alivar_95GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    varmaps[2] = hp.read_map('./data/TQU_alivar_150GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    varmaps[3] = hp.read_map('./data/TQU_plkvar_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    maskmap = hp.read_map('./data/ali_mask_r7.fits',dtype=np.float64,verbose=0)
    
    cmbmap = hp.read_map('./data/TQU_CMB_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
    
    """---------- don't worry about the following ----------"""
    
    fullmap = signalmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    fullvar = varmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    
    pipeline1 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=1,nside=NSIDE,mask=maskmap.reshape(1,-1),variances=fullvar,fwhms=FWHMS,lmax=LMAX)
    pipeline1.nsamp = NSAMP
    pipeline1.debug = True
    rslt_t = pipeline1.run(aposcale=APOSCALE,psbin=PSBIN,shift=SHIFT_T,threshold=CUT_T)
    
    fullmap = signalmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    fullvar = varmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    
    pipeline2 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=2,nside=NSIDE,mask=maskmap.reshape(1,-1),variances=fullvar,fwhms=FWHMS,lmax=LMAX)
    pipeline2.nsamp = NSAMP
    pipeline2.debug = True
    rslt_eb = pipeline2.run(aposcale=APOSCALE,psbin=PSBIN,shift=SHIFT_EB,threshold=CUT_EB)
    
    output = list()
    output.append(rslt_t[0])
    output.append(rslt_t[1])
    output.append(rslt_eb[1])
    output.append(rslt_eb[2])
    
    cmbmap[:,maskmap<1] = 0.
    est = ap.pstimator(nside=NSIDE,mask=maskmap.reshape(1,-1),aposcale=APOSCALE,psbin=PSBIN,lmax=LMAX)
    auto_cmb_t = est.auto_t(cmbmap[0].reshape(1,-1),fwhms=FWHMS[0])
    auto_cmb_eb = est.auto_eb(cmbmap[1:].reshape(2,-1),fwhms=FWHMS[0])
    
    output.append(auto_cmb_t[1])
    output.append(auto_cmb_eb[1])
    output.append(auto_cmb_eb[2])
    
    np.save('abs_example.npy',output)
        

if __name__ == '__main__':
    main()
