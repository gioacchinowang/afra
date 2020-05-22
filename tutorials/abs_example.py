import abspy as ap
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

def main():
    """MPI supported version of tutorial 04
    
    record your input parameters
    prepare your singal, variance and mask maps
    """
    NSIDE = 128
    NSAMP = 200  # global sampling size
    NSAMP = max(2,NSAMP//mpisize)  # local sampling size
    NFREQ = 4  # frequency band
    FWHMS = [1.e-3]*NFREQ # beam FWHM each frequencies
    ABSBIN = 100  # angular mode bin number
    SHIFT_T = 10.  # CMB bandpower shift
    SHIFT_EB = 10.
    CUT_T = 1.  # CMB bandpower extraction threshold
    CUT_EB = 0.1
    
    # import data
    if not mpirank:
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
    else:
        signalmaps = None
        varmaps = None
        cmbmap = None
        maskmap = None
    # bcast data
    signalmaps = comm.bcast(signalmaps, root=0)
    varmaps = comm.bcast(varmaps, root=0)
    cmbmap = comm.bcast(cmbmap, root=0)
    maskmap = comm.bcast(maskmap, root=0)
    
    """---------- don't worry about the following ----------"""
    
    fullmap = signalmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    fullvar = varmaps[:,0,:].reshape(NFREQ,1,12*NSIDE**2)
    
    pipeline1 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=1,nside=NSIDE,mask=maskmap.reshape(1,-1),variance=fullvar,fwhms=FWHMS)
    pipeline1.nsamp = NSAMP
    raw_t = pipeline1.method_noisyT_raw(psbin=20)
    
    fullmap = signalmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    fullvar = varmaps[:,1:,:].reshape(NFREQ,2,12*NSIDE**2)
    
    pipeline2 = ap.abspipe(fullmap,nfreq=NFREQ,nmap=2,nside=NSIDE,mask=maskmap.reshape(1,-1),variance=fullvar,fwhms=FWHMS)
    pipeline2.nsamp = NSAMP
    raw_eb = pipeline2.method_noisyEB_raw(psbin=20)
    
    # collect data
    ellist = raw_t[0]  # angular mode position
    raw_t_noise = None
    raw_t_signal = None
    raw_e_noise = None
    raw_e_signal = None
    raw_b_noise = None
    raw_b_signal = None
    if not mpirank:
        raw_t_noise = np.empty([mpisize*NSAMP, NFREQ, len(ellist), len(ellist)], dtype=raw_t[1].dtype)
        raw_t_signal = np.empty([mpisize*NSAMP, NFREQ, len(ellist), len(ellist)], dtype=raw_t[1].dtype)
        raw_e_noise = np.empty([mpisize*NSAMP, NFREQ, len(ellist), len(ellist)], dtype=raw_eb[1].dtype)
        raw_e_signal = np.empty([mpisize*NSAMP, NFREQ, len(ellist), len(ellist)], dtype=raw_eb[1].dtype)
        raw_b_noise = np.empty([mpisize*NSAMP, NFREQ, len(ellist), len(ellist)], dtype=raw_eb[1].dtype)
        raw_b_signal = np.empty([mpisize*NSAMP, NFREQ, len(ellist), len(ellist)], dtype=raw_eb[1].dtype)
    comm.Gather(raw_t[1], raw_t_noise, root=0)
    comm.Gather(raw_t[2], raw_t_signal, root=0)
    comm.Gather(raw_eb[1], raw_e_noise, root=0)
    comm.Gather(raw_eb[2], raw_b_noise, root=0)
    comm.Gather(raw_eb[3], raw_e_signal, root=0)
    comm.Gather(raw_eb[4], raw_b_signal, root=0)
    
    # analyze raw output
    if not mpirank:
        
        # analyze TT
        # get noise PS mean and rms
        t_noise_mean = np.mean(raw_t_noise,axis=0)
        t_noise_std = np.std(raw_t_signal,axis=0)
        t_noise_std_diag = np.zeros((len(ellist),NFREQ))
        for l in range(len(ellist)):
            t_noise_std_diag[l] = np.diag(t_noise_std[l])
        # add signal map
        rslt_ell = list()
        rslt_Dt = list()
        for s in range(mpisize*NSAMP):
            # send PS to ABS method
            safe_absbin = min(len(ellist), ABSBIN)
            spt_t = abssep(raw_t_signal[s],t_noise_mean,t_noise_std_diag,modes=ellist,bins=safe_absbin,shift=SHIFT_T,threshold=CUT_T)
            rslt_t = spt_t()
            if (s==0):
                rslt_ell = rslt_t[0]
            rslt_Dt += rslt_t[1]
        rslt_Dt_array = np.reshape(rslt_Dt, (mpisize*NSAMP,-1))
        
        # analyze EE and BB
        # get noise PS mean and rms
        e_noise_mean = np.mean(raw_e_noise,axis=0)
        e_noise_std = np.std(raw_e_noise,axis=0)
        b_noise_mean = np.mean(raw_b_noise,axis=0)
        b_noise_std = np.std(raw_b_noise,axis=0)
        e_noise_std_diag = np.zeros((len(ellist),NFREQ))
        b_noise_std_diag = np.zeros((len(ellist),NFREQ))
        for l in range(len(ellist)):
            e_noise_std_diag[l] = np.diag(e_noise_std[l])
            b_noise_std_diag[l] = np.diag(b_noise_std[l])
        # add signal map
        rslt_ell = list()
        rslt_De = list()
        rslt_Db = list()
        for s in range(mpisize*NSAMP):
            # send PS to ABS method
            safe_absbin = min(len(ellist), ABSBIN)
            spt_e = abssep(raw_e_signal[s],e_noise_mean,e_noise_std_diag,modes=ellist,bins=safe_absbin,shift=SHIFT_EB,threshold=CUT_EB)
            spt_b = abssep(raw_b_signal[s],b_noise_mean,b_noise_std_diag,modes=ellist,bins=safe_absbin,shift=SHIFT_EB,threshold=CUT_EB)
            rslt_e = spt_e()
            rslt_b = spt_b()
            if (s==0):
                rslt_ell = rslt_e[0]
            rslt_De += rslt_e[1]
            rslt_Db += rslt_b[1]
        # get result
        rslt_De_array = np.reshape(rslt_De, (mpisize*NSAMP,-1))
        rslt_Db_array = np.reshape(rslt_Db, (mpisize*NSAMP,-1))
        
        output = np.zeros((7,len(ellist)))
        output[0] = ellist
        output[1] = np.mean(rslt_Dt_array,axis=0)
        output[2] = np.std(rslt_Dt_array,axis=0)
        output[3] = np.mean(rslt_De_array,axis=0)
        output[4] = np.std(rslt_De_array,axis=0)
        output[5] = np.mean(rslt_Db_array,axis=0)
        output[6] = np.std(rslt_Db_array,axis=0)
        
        np.save('abs_example.npy',output)
        
        est = ap.pstimator(nside=NSIDE,mask=mask.reshape(1,-1), aposcale=5.0, psbin=20)
        auto_cmb_t = est.auto_t(mapcmb[0].reshape(1,-1),fwhms=1.e-3)
        auto_cmb_eb = est.auto_eb(mapcmb[1:].reshape(2,-1),fwhms=1.e-3)
        output[0] = auto_cmb_t[0]
        output[1] = auto_cmb_t[1]
        output[2] = auto_cmb_eb[1]
        output[3] = auto_cmb_eb[2]
        
        np.save('abs_cmb.npy',output[:4])
        

if __name__ == '__main__':
    main()
