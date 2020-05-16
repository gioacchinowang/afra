import abspy as ap
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

def main(cut):
    """compact script of tutorial 04"""
    NSIDE = 128
    NSCAL = 1.
    NSAMP = 200  # global sampling size
    NSAMP = NSAMP//mpisize  # local sampling size
    
    if not mpirank:
        # import data
        map30 = hp.read_map('./data/TQU_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
        map95 = hp.read_map('./data/TQU_95GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
        map150 = hp.read_map('./data/TQU_150GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
        map353 = hp.read_map('./data/TQU_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
        mapcmb = hp.read_map('./data/TQU_CMB_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)
        vmap30 = hp.read_map('./data/TQU_var_30GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)*NSCAL
        vmap95 = hp.read_map('./data/TQU_var_95GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)*NSCAL
        vmap150 = hp.read_map('./data/TQU_var_150GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)*NSCAL
        vmap353 = hp.read_map('./data/TQU_var_353GHz_r7.fits',field=[0,1,2],dtype=np.float64,verbose=0)*NSCAL
        mask = hp.read_map('./data/ali_mask_r7.fits',dtype=bool,verbose=False)
    else:
        map30 = None
        map95 = None
        map150 = None
        map353 = None
        mapcmb = None
        vmap30 = None
        vmap95 = None
        vmap150 = None
        vmap353 = None
        mask = None
    # bcast
    map30 = comm.bcast(map30, root=0)
    map95 = comm.bcast(map95, root=0)
    map150 = comm.bcast(map150, root=0)
    map353 = comm.bcast(map353, root=0)
    mapcmb = comm.bcast(mapcmb, root=0)
    vmap30 = comm.bcast(vmap30, root=0)
    vmap95 = comm.bcast(vmap95, root=0)
    vmap150 = comm.bcast(vmap150, root=0)
    vmap353 = comm.bcast(vmap353, root=0)
    mask = comm.bcast(mask, root=0)
    
    fullmap = np.zeros((4,1,12*NSIDE**2))
    fullmap[0] = map30[0]
    fullmap[1] = map95[0]
    fullmap[2] = map150[0]
    fullmap[3] = map353[0]
    
    fullvar = np.zeros((4,1,12*NSIDE**2))
    fullvar[0] = vmap30[0]
    fullvar[1] = vmap95[0]
    fullvar[2] = vmap150[0]
    fullvar[3] = vmap353[0]
    
    pipeline1 = ap.abspipe(fullmap,nfreq=4,nmap=1,nside=NSIDE,mask=mask.reshape(1,-1),variance=fullvar,fwhms=[1.e-3,1.e-3,1.e-3,1.e-3])
    pipeline1.nsamp = NSAMP
    rslt_t = pipeline1.method_noisyT_raw(psbin=20,absbin=100,shift=10.,threshold=cut)
    #rslt_t = pipeline1(psbin=20,absbin=100,shift=10.,threshold=cut)
    
    fullmap = np.zeros((4,2,12*NSIDE**2))
    fullmap[0] = map30[1:]
    fullmap[1] = map95[1:]
    fullmap[2] = map150[1:]
    fullmap[3] = map353[1:]
    
    fullvar = np.zeros((4,2,12*NSIDE**2))
    fullvar[0] = vmap30[1:]
    fullvar[1] = vmap95[1:]
    fullvar[2] = vmap150[1:]
    fullvar[3] = vmap353[1:]
    
    pipeline2 = ap.abspipe(fullmap,nfreq=4,nmap=2,nside=NSIDE,mask=mask.reshape(1,-1),variance=fullvar,fwhms=[1.e-3,1.e-3,1.e-3,1.e-3])
    pipeline2.nsamp = NSAMP
    rslt_eb = pipeline2.method_noisyEB_raw(psbin=20,absbin=100,shift=0.,threshold=cut)
    #rslt_eb = pipeline2(psbin=20,absbin=100,shift=0.,threshold=cut)
    
    # collect data
    rslt_t_dl = None
    rslt_e_dl = None
    rslt_b_dl = None
    if not mpirank:
        rslt_t_dl = np.empty([mpisize*NSAMP, len(rslt_t[0])], dtype=rslt_t[1].dtype)
        rslt_e_dl = np.empty([mpisize*NSAMP, len(rslt_eb[0])], dtype=rslt_eb[1].dtype)
        rslt_b_dl = np.empty([mpisize*NSAMP, len(rslt_eb[0])], dtype=rslt_eb[2].dtype)
    comm.Gather(rslt_t[1], rslt_t_dl, root=0)
    comm.Gather(rslt_eb[1], rslt_e_dl, root=0)
    comm.Gather(rslt_eb[2], rslt_b_dl, root=0)
    
    # plot
    if not mpirank:
        
        output = np.zeros((7,len(rslt_t[0])))
        output[0] = rslt_t[0]
        output[1] = np.mean(rslt_t_dl,axis=0)
        output[2] = np.std(rslt_t_dl,axis=0)
        output[3] = np.mean(rslt_e_dl,axis=0)
        output[4] = np.std(rslt_e_dl,axis=0)
        output[5] = np.mean(rslt_b_dl,axis=0)
        output[6] = np.std(rslt_b_dl,axis=0)
        
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
    main(0.1)
