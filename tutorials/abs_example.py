import abspy as ap
import healpy as hp
import numpy as np
import matplotlib
matplotlib.use('Agg')


def main():
    """compact script of tutorial 04"""
    NSIDE = 128
    NSCAL = 1.e-3
    NSAMP = 10
    
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
    
    pipeline1 = ap.abspipe(fullmap,nfreq=4,nmap=1,nside=NSIDE,mask=mask.reshape(1,-1),variance=fullvar)
    pipeline1.nsamp = NSAMP
    rslt_t = pipeline1(psbin=10,absbin=10,shift=10.,threshold=1.)
    
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
    
    pipeline2 = ap.abspipe(fullmap,nfreq=4,nmap=2,nside=NSIDE,mask=mask.reshape(1,-1),variance=fullvar)
    pipeline2.nsamp = NSAMP
    rslt_eb = pipeline2(psbin=10,absbin=10,shift=0.,threshold=1.)
    
    fig,ax = matplotlib.pyplot.subplots(figsize=(10,10))
    
    cmb_cl = hp.anafast(mapcmb)
    ell = np.arange(len(cmb_cl[0]))
    ax.plot((0.5/np.pi)*ell*(ell+1)*cmb_cl[0],color='black',linestyle='-',label=r'CMB TT')
    ax.plot((0.5/np.pi)*ell*(ell+1)*cmb_cl[1],color='black',linestyle='-',label=r'CMB EE')
    ax.plot((0.5/np.pi)*ell*(ell+1)*cmb_cl[2],color='black',linestyle='-',label=r'CMB BB')
    ax.errorbar(rslt_t[0],rslt_t[1],yerr=rslt_t[2],color='red',fmt='o',label=r'ABS TT')
    ax.errorbar(rslt_eb[0],rslt_eb[1],yerr=rslt_eb[2],color='green',fmt='s',label=r'ABS EE')
    ax.errorbar(rslt_eb[0],rslt_eb[3],yerr=rslt_eb[4],color='orange',fmt='X',label=r'ABS BB')
    ax.fill_between((2*NSIDE,3*NSIDE),(1e-7,1e-7),(1e7,1e7),color='gray',linestyle='--',alpha=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc=2)
    ax.set_xlabel(r'$\ell$',fontsize=20)
    ax.set_ylabel(r'$D_\ell$ ($\mu K^2$)',fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim((1.e-7,1.e7))
    ax.set_xlim((1,3*NSIDE))
    matplotlib.pyplot.savefig('abs_example.pdf', bbox_inches='tight')
    

if __name__ == '__main__':
    main()
