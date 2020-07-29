import logging as log
import numpy as np
import healpy as hp
from copy import deepcopy
from afra.methods.abs import abssep
from afra.tools.ps_estimator import pstimator
from afra.tools.icy_decorator import icy


@icy
class abspipe(object):
    """
    The ABS pipeline class.
    
    Author:
    - Jian Yao (STJU)
    - Jiaxin Wang (SJTU) jiaxin.wang@sjtu.edu.cn
    """
    def __init__(self, signals, variances=None, mask=None, lmax=None, fwhms=None):
        """
        The ABS pipeline for extracting CMB power-spectrum band power,
        according to given measured sky maps at various frequency bands.
    
        Parameters
        ----------
    
        signals : dict
            Measured signal maps,
            should be arranged in form: {frequency (GHz) : array(map #, pixel #)}.
        
        variances : dict
            Measured noise variance maps,
            should be arranged in form: {frequency (GHz): array(map #, pixel #)}.
            By default, no variance maps required.
        
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (1, pixel #).
    
        lmax : integer
            Maximal multiple.
            
        fwhms : list, tuple
            FWHM (in rad) of gaussian beams for each frequency
        """
        log.debug('@ abspipe::__init__')
        #
        self.signals = signals
        self.variances = variances
        self.mask = mask
        #
        self.lmax = lmax
        self.fwhms = fwhms
        # method select dict with keys defined by (self._noise_flag, self._nmap)
        self._methodict = {(True,1): self.method_noisyT,
                           (True,2): self.method_noisyEB,
                           (False,1): self.method_pureT,
                           (False,2): self.method_pureEB}
        # resampling size
        self.nsamp = 1000
        # debug mode
        self.debug = False
        # ps estimator
        self._est = None

    @property
    def signals(self):
        return self._signals
        
    @property
    def variances(self):
        return self._variances
        
    @property
    def mask(self):
        return self._mask

    @property
    def nfreq(self):
        return self._nfreq
        
    @property
    def nside(self):
        return self._nside
    
    @property
    def nmap(self):
        return self._nmap
        
    @property
    def nsamp(self):
        return self._nsamp
        
    @property
    def debug(self):
        return self._debug
        
    @property
    def lmax(self):
        return self._lmax
        
    @property
    def fwhms(self):
        return self._fwhms
    
    @lmax.setter
    def lmax(self, lmax):
        if lmax is None:
            self._lmax = 2*self._nside
        else:
            assert isinstance(lmax, int)
            assert (lmax < 3*self._nside)
            self._lmax = lmax
        
    @signals.setter
    def signals(self, signals):
        """detect and register nfreq, nmap, npix and nside automatically
        """
        assert isinstance(signals, dict)
        self._nfreq = len(signals)
        log.debug('number of frequencies: %s' % str(self._nfreq))
        assert (len(signals[next(iter(signals))].shape) == 2)
        self._nmap = signals[next(iter(signals))].shape[0]
        log.debug('number of maps: %s' % str(self._nmap))
        self._npix = signals[next(iter(signals))].shape[1]
        self._nside = int(np.sqrt(self._npix//12))
        log.debug('HEALPix Nside: %s' % str(self._nside))
        self._signals = np.r_[[signals[x] for x in sorted(signals.keys())]]
        log.debug('signal maps loaded')
        
    @variances.setter
    def variances(self, variances):
        if variances is not None:
            assert isinstance(variances, dict)
            assert (variances[next(iter(variances))].shape[0] == self._nmap)
            assert (variances[next(iter(variances))].shape[1] == self._npix)
            self._noise_flag = True
            self._variances = np.r_[[variances[x] for x in sorted(variances.keys())]]
        else:
            self._noise_flag = False
            self._variances = None
        log.debug('variance maps loaded')
        
    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones((1,self._npix),dtype=np.float32)
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = mask.copy()
        # clean up input maps with mask
        self._mask[:,self._mask[0]==0.] = 0.
        self._signals[:,:,self._mask[0]==0.] = 0.
        if self._variances is not None:
            self._variances[:,:,self._mask[0]==0.] = 0.
        log.debug('mask map loaded')
        
    @nsamp.setter
    def nsamp(self, nsamp):
        assert isinstance(nsamp, int)
        self._nsamp = nsamp
        log.debug('resampling size set: %s' % str(self._nsamp))
        
    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug
        log.debug('debug mode: %s' % str(self._debug))
        
    @fwhms.setter
    def fwhms(self, fwhms):
        if fwhms is None:
            self._fwhms = [fwhms]*self._nfreq
        else:
            assert isinstance(fwhms, (list,tuple))
            assert (len(fwhms) == self._nfreq)
            self._fwhms = deepcopy(fwhms)
        log.debug('fwhms loaded')
        
    def __call__(self, aposcale, psbin, shift=None, threshold=None, verbose=False):
        """
        ABS pipeline class call function.
        
        Parameters
        ----------
        
        aposcale : float
            Apodization scale.
        
        psbin : integer
            Number of angular modes in each bin,
            for conducting pseudo-PS estimation.
            
        shift : float
            ABS method shift parameter.
            
        threshold : float or None
            ABS method threshold parameter.
        
        Returns
        -------
        
        angular modes, target angular power spectrum : tuple of lists
        """
        log.debug('@ abspipe::__call__')
        assert isinstance(aposcale, float)
        assert isinstance(psbin, int)
        assert (psbin > 0)
        assert (aposcale > 0)
        return self.run(aposcale, psbin, shift, threshold, verbose)
        
    def run(self, aposcale, psbin, shift, threshold, verbose=False):
        # init PS estimator
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmax=self._lmax)
        # method selection
        return self._methodict[(self._noise_flag,self._nmap)](shift, threshold, verbose)
        
    def run_bmode(self, aposcale, psbin, shift, threshold, verbose=False):
        """alternative routine for B mode only,
        with EB-leakage corrected before PS estimation,
        for pipeline test only"""
        # init PS estimator
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmax=self._lmax)
        # method selection
        if self._noise_flag:
            return self.method_noisyB(shift, threshold, verbose)
        else:
            return self.method_pureB(shift, threshold, verbose)
    
    def method_pureT(self, shift, threshold, verbose):
        """
        CMB T mode (band power) extraction without noise.
        
        Returns
        -------
        angular modes, T-mode PS
        """
        modes = self._est.modes
        # prepare total signals PS in the shape required by ABS method
        signal_ps_t = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto correlation
            stmp = self._est.auto_t(self._signals[i],fwhms=self._fwhms[i])
            # assign results
            for k in range(len(modes)):
                signal_ps_t[k,i,i] = stmp[1][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                stmp = self._est.cross_t(np.r_[self._signals[i],self._signals[j]],fwhms=[self._fwhms[i],self._fwhms[j]])
                for k in range(len(modes)):
                    signal_ps_t[k,i,j] = stmp[1][k]
                    signal_ps_t[k,j,i] = stmp[1][k]
        # send PS to ABS method, noiseless case requires no shift nor threshold
        spt_t = abssep(signal_ps_t,shift=None,threshold=None)
        if verbose:
            return (modes, spt_t.run(), spt_t.run_info())
        return (modes, spt_t.run())
        
    def method_noisyT_raw(self):
        """
        CMB T mode (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, T-mode PS, T-mode PS std
        """
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        modes = self._est.modes
        for i in range(self._nfreq):
            tmp = self._est.auto_t(self._signals[0],fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]  # register workspace
            for j in range(i+1,self._nfreq):
                tmp = self._est.cross_t(self._signals[:2,0],fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]  # register workspace
        # allocate
        noise_ps_t = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_t = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        noise_map = np.zeros((2,self._npix),dtype=np.float64)
        signal_map = np.zeros((2,self._npix),dtype=np.float64)
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                noise_map[0] = np.random.normal(size=self._npix)*np.sqrt(self._variances[i,0])
                signal_map[0] = self._signals[i,0] + noise_map[0]
                # auto correlation
                ntmp = self._est.auto_t(noise_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = self._est.auto_t(signal_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(len(modes)):
                    noise_ps_t[s,k,i,i] = ntmp[1][k]
                    signal_ps_t[s,k,i,i] = stmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    noise_map[1] = np.random.normal(size=self._npix)*np.sqrt(self._variances[j,0])
                    signal_map[1] = self._signals[j,0] + noise_map[1]
                    # cross correlation
                    ntmp = self._est.cross_t(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = self._est.cross_t(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        noise_ps_t[s,k,i,j] = ntmp[1][k]
                        noise_ps_t[s,k,j,i] = ntmp[1][k]
                        signal_ps_t[s,k,i,j] = stmp[1][k]
                        signal_ps_t[s,k,j,i] = stmp[1][k]
        return (modes, noise_ps_t, signal_ps_t)
        
    def method_noisyT(self, shift, threshold, verbose):
        modes, noise_ps_t, signal_ps_t = self.method_noisyT_raw()
        # get noise PS mean and rms
        noise_ps_t_mean = np.mean(noise_ps_t,axis=0)
        noise_ps_t_std = np.std(noise_ps_t,axis=0)
        noise_ps_t_std_diag = np.zeros((len(modes),self._nfreq),dtype=np.float64)
        for l in range(len(modes)):
            noise_ps_t_std_diag[l] = np.diag(noise_ps_t_std[l])
        # shift for each angular mode independently
        safe_shift = shift*np.mean(noise_ps_t_std_diag,axis=1)
        rslt_Dt = np.zeros((self._nsamp,len(modes)),dtype=np.float64)
        for s in range(self._nsamp):
            # send PS to ABS method
            spt_t = abssep(signal_ps_t[s]-noise_ps_t_mean,noise_ps_t_mean,noise_ps_t_std_diag,shift=safe_shift,threshold=threshold)
            rslt_Dt[s] = spt_t.run()
        if verbose:
            spt_t = abssep(np.mean(signal_ps_t,axis=0)-noise_ps_t_mean,noise_ps_t_mean,noise_ps_t_std_diag,shift=safe_shift,threshold=threshold)
            if self._debug:
                return (modes, rslt_Dt, spt_t.run_info())
            return (modes, np.mean(rslt_Dt,axis=0), np.std(rslt_Dt,axis=0), spt_t.run_info())
        if self._debug:
            return (modes, rslt_Dt)
        return (modes, np.mean(rslt_Dt,axis=0), np.std(rslt_Dt,axis=0))
    
    def method_pureEB(self, shift, threshold, verbose):
        """
        CMB E and B modes (band power) extraction without noise.
        
        Returns
        -------
        
        angular modes, E-mode PS, B-mode
        """
        modes = self._est.modes  # register angular modes
        signal_ps_e = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_b = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto corr
            stmp = self._est.auto_eb(self._signals[i],fwhms=self._fwhms[i])
            # assign results
            for k in range(len(modes)):
                signal_ps_e[k,i,i] = stmp[1][k]
                signal_ps_b[k,i,i] = stmp[2][k]
            # cross corr
            for j in range(i+1,self._nfreq):
                stmp = self._est.cross_eb(np.r_[self._signals[i],self._signals[j]],fwhms=[self._fwhms[i],self._fwhms[j]])
                for k in range(len(modes)):
                    signal_ps_e[k,i,j] = stmp[1][k]
                    signal_ps_b[k,i,j] = stmp[2][k]
                    signal_ps_e[k,j,i] = stmp[1][k]
                    signal_ps_b[k,j,i] = stmp[2][k]
        # send PS to ABS method, noiseless case requires no shift nor threshold
        spt_e = abssep(signal_ps_e,shift=None,threshold=None)
        spt_b = abssep(signal_ps_b,shift=None,threshold=None)
        if verbose:
            return (modes, spt_e.run(), spt_b.run(), spt_e.run_info(), spt_b.run_info())
        return (modes, spt_e.run(), spt_b.run())
        
    def method_noisyEB_raw(self):
        """
        CMB E and B modes (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, E-mode PS, E-mode PS std, B-mode PS, B-mode PS std
        """
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        modes = self._est.modes
        for i in range(self._nfreq):
            tmp = self._est.auto_eb(self._signals[0],fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]  # register workspace
            for j in range(i+1,self._nfreq):
                tmp = self._est.cross_eb(self._signals[:2].reshape(4,-1),fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]  # register workspace
        # allocate
        noise_ps_e = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        noise_ps_b = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_e = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_b = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        noise_map = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
        signal_map = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                noise_map[0] = np.random.normal(size=self._npix)*np.sqrt(self._variances[i,0])
                signal_map[0] = self._signals[i,0] + noise_map[0]
                noise_map[1] = np.random.normal(size=self._npix)*np.sqrt(self._variances[i,1])
                signal_map[1] = self._signals[i,1] + noise_map[1]
                # auto correlation
                ntmp = self._est.auto_eb(noise_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = self._est.auto_eb(signal_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(len(modes)):
                    noise_ps_e[s,k,i,i] = ntmp[1][k]
                    noise_ps_b[s,k,i,i] = ntmp[2][k]
                    signal_ps_e[s,k,i,i] = stmp[1][k]
                    signal_ps_b[s,k,i,i] = stmp[2][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    noise_map[2] = np.random.normal(size=self._npix)*np.sqrt(self._variances[j,0])
                    signal_map[2] = self._signals[j,0] + noise_map[2]
                    noise_map[3] = np.random.normal(size=self._npix)*np.sqrt(self._variances[j,1])
                    signal_map[3] = self._signals[j,1] + noise_map[3]
                    # cross correlation
                    ntmp = self._est.cross_eb(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = self._est.cross_eb(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        noise_ps_e[s,k,i,j] = ntmp[1][k]
                        noise_ps_b[s,k,i,j] = ntmp[2][k]
                        noise_ps_e[s,k,j,i] = ntmp[1][k]
                        noise_ps_b[s,k,j,i] = ntmp[2][k]
                        signal_ps_e[s,k,i,j] = stmp[1][k]
                        signal_ps_e[s,k,j,i] = stmp[1][k]
                        signal_ps_b[s,k,i,j] = stmp[2][k]
                        signal_ps_b[s,k,j,i] = stmp[2][k]
        return (modes, noise_ps_e, noise_ps_b, signal_ps_e, signal_ps_b)
        
    def method_noisyEB(self, shift, threshold, verbose):
        modes, noise_ps_e, noise_ps_b, signal_ps_e, signal_ps_b = self.method_noisyEB_raw()
        # get noise PS mean and rms
        noise_ps_e_mean = np.mean(noise_ps_e,axis=0)
        noise_ps_e_std = np.std(noise_ps_e,axis=0)
        noise_ps_b_mean = np.mean(noise_ps_b,axis=0)
        noise_ps_b_std = np.std(noise_ps_b,axis=0)
        noise_ps_e_std_diag = np.zeros((len(modes),self._nfreq),dtype=np.float64)
        noise_ps_b_std_diag = np.zeros((len(modes),self._nfreq),dtype=np.float64)
        for l in range(len(modes)):
            noise_ps_e_std_diag[l] = np.diag(noise_ps_e_std[l])
            noise_ps_b_std_diag[l] = np.diag(noise_ps_b_std[l])
        safe_shift_e = shift*np.mean(noise_ps_e_std_diag,axis=1)
        safe_shift_b = shift*np.mean(noise_ps_b_std_diag,axis=1)
        # add signal map
        rslt_De = np.zeros((self._nsamp,len(modes)),dtype=np.float64)
        rslt_Db = np.zeros((self._nsamp,len(modes)),dtype=np.float64)
        for s in range(self._nsamp):
            # send PS to ABS method
            spt_e = abssep(signal_ps_e[s]-noise_ps_e_mean,noise_ps_e_mean,noise_ps_e_std_diag,shift=safe_shift_e,threshold=threshold)
            spt_b = abssep(signal_ps_b[s]-noise_ps_b_mean,noise_ps_b_mean,noise_ps_b_std_diag,shift=safe_shift_b,threshold=threshold)
            rslt_De[s] = spt_e.run()
            rslt_Db[s] = spt_b.run()
        # get result
        if verbose:
            spt_e = abssep(np.mean(signal_ps_e,axis=0)-noise_ps_e_mean,noise_ps_e_mean,noise_ps_e_std_diag,shift=safe_shift_e,threshold=threshold)
            spt_b = abssep(np.mean(signal_ps_b,axis=0)-noise_ps_b_mean,noise_ps_b_mean,noise_ps_b_std_diag,shift=safe_shift_b,threshold=threshold)
            if self._debug:
                return (modes, rslt_De, rslt_Db, spt_e.run_info(), spt_b.run_info())
            return (modes, np.mean(rslt_De,axis=0), np.std(rslt_De,axis=0), np.mean(rslt_Db,axis=0), np.std(rslt_Db,axis=0), spt_e.run_info(), spt_b.run_info())
        if self._debug:
            return (modes, rslt_De, rslt_Db)
        return (modes, np.mean(rslt_De,axis=0), np.std(rslt_De,axis=0), np.mean(rslt_Db,axis=0), np.std(rslt_Db,axis=0))
        
    def method_pureB(self, shift, threshold, verbose):
        """
        CMB B mode (band power) extration without noise
        
        Returns
        -------
        angular mdoes, B-mode PS
        """
        # get B mode maps from self._signals
        bmaps = self.purify()
        # run a trial PS estimation
        modes = self._est.modes
        # prepare total signals PS in the shape required by ABS method
        signal_ps_t = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto correlation
            stmp = self._est.auto_t(bmaps[i],fwhms=self._fwhms[i])
            # assign results
            for k in range(len(modes)):
                signal_ps_t[k,i,i] = stmp[1][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                stmp = self._est.cross_t(np.vstack([bmaps[i],bmaps[j]]),fwhms=[self._fwhms[i],self._fwhms[j]])
                for k in range(len(modes)):
                    signal_ps_t[k,i,j] = stmp[1][k]
                    signal_ps_t[k,j,i] = stmp[1][k]
        # send PS to ABS method, noiseless case requires no shift nor threshold
        spt_t = abssep(signal_ps_t,shift=None,threshold=None)
        if verbose:
            return (modes, spt_t.run(), spt_t.run_info())
        return (modes, spt_t.run())
        
    def method_noisyB_raw(self):
        """
        CMB B mode (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, B-mode PS, B-mode PS std
        """
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        modes = self._est.modes
        for i in range(self._nfreq):
            tmp = self._est.auto_t(self._signals[0,0].reshape(1,-1),fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]  # register workspace
            for j in range(i+1,self._nfreq):
                tmp = self._est.cross_t(self._signals[:2,0],fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]  # register workspace
        # allocate
        noise_ps_t = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_t = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float64)
        noise_map = np.zeros((2,self._npix),dtype=np.float64)
        signal_map = np.zeros((2,self._npix),dtype=np.float64)
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                mtmp = np.random.normal(size=(3,self._npix))*np.sqrt(self._variances[i])
                noise_map[0] = self.purify_mono(mtmp)
                signal_map[0] = self.purify_mono(self._signals[i] + mtmp)
                # auto correlation
                ntmp = self._est.auto_t(noise_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = self._est.auto_t(signal_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(len(modes)):
                    noise_ps_t[s,k,i,i] = ntmp[1][k]
                    signal_ps_t[s,k,i,i] = stmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    mtmp = np.random.normal(size=(3,self._npix))*np.sqrt(self._variances[j])
                    noise_map[1] = self.purify_mono(mtmp)
                    signal_map[1] = self.purify_mono(self._signals[j] + mtmp)
                    # cross correlation
                    ntmp = self._est.cross_t(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = self._est.cross_t(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        noise_ps_t[s,k,i,j] = ntmp[1][k]
                        noise_ps_t[s,k,j,i] = ntmp[1][k]
                        signal_ps_t[s,k,i,j] = stmp[1][k]
                        signal_ps_t[s,k,j,i] = stmp[1][k]
        return (modes, noise_ps_t, signal_ps_t)
        
    def method_noisyB(self, shift, threshold, verbose):
        modes, noise_ps_t, signal_ps_t = self.method_noisyB_raw()
        # get noise PS mean and rms
        noise_ps_t_mean = np.mean(noise_ps_t,axis=0)
        noise_ps_t_std = np.std(noise_ps_t,axis=0)
        noise_ps_t_std_diag = np.zeros((len(modes),self._nfreq),dtype=np.float64)
        for l in range(len(modes)):
            noise_ps_t_std_diag[l] = np.diag(noise_ps_t_std[l])
        safe_shift = shift*np.mean(noise_ps_t_std_diag,axis=1)  # shift for each angular mode independently
        # add signal map
        rslt_Dt = np.zeros((self._nsamp,len(modes)),dtype=np.float64)
        for s in range(self._nsamp):
            # send PS to ABS method
            spt_t = abssep(signal_ps_t[s]-noise_ps_t_mean,noise_ps_t_mean,noise_ps_t_std_diag,shift=safe_shift,threshold=threshold)
            rslt_Dt[s] = spt_t.run()
        if verbose:
            spt_t = abssep(np.mean(signal_ps_t,axis=0)-noise_ps_t_mean,noise_ps_t_mean,noise_ps_t_std_diag,shift=safe_shift,threshold=threshold)
            if self._debug:
                return (modes, rslt_Dt, spt_t.run_info())
            return (modes, np.mean(rslt_Dt,axis=0), np.std(rslt_Dt,axis=0), spt_t.run_info())
        if self._debug:
            return (modes, rslt_Dt)
        return (modes, np.mean(rslt_Dt,axis=0), np.std(rslt_Dt,axis=0))

    def purify(self):
        """Get pure B mode maps with EB leakage corrected
        
        Parameters
        ----------
        
        map : numpy.ndarray
            TQU maps at various frequencies,
            in shape (N_freq, 3, N_pix).
            
        mask : numpy.ndarray
            mask map, in shape (N_pix, ).
            
        Returns
        -------
        B mode maps : numpy.ndarray
            B mode maps at various frequencies,
            in shape (N_freq, 1, N_pix).
        """
        mask_sum = np.sum(self._mask[0])
        pix_list = np.arange(self._npix)
        fill = np.arange(self._npix)[np.where(self._mask[0]==1.)] # the pixel index of the available index
        rslt = np.zeros((self._nfreq,1,self._npix),dtype=np.float64)
        # process at each frequency
        for i in range(self._nfreq):
            # get the template of E to B leakage
            Alm0 = hp.map2alm(self._signals[i]) #alms of the masked maps
            B0 = hp.alm2map(Alm0[2],nside=self._nside,verbose=0)  # corrupted B map
            Alm0[0] = 0.
            Alm0[2] = 0.
            E0 = hp.alm2map(Alm0,nside=self._nside,verbose=0)  # TQU of corrupted E mode only
            E0[:,self._mask[0]==0.] = 0.  # re-mask
            Alm1 = hp.map2alm(E0)  # Alms of the TUQ from E-mode only
            B1 = hp.alm2map(Alm1[2],nside=self._nside,verbose=0)  # template of EB leakage
            # compute the residual of linear fit
            x = B1[fill]
            y = B0[fill]
            mx  = np.sum(x)/mask_sum
            my  = np.sum(y)/mask_sum
            cxx = np.sum((x-mx)*(x-mx))
            cxy = np.sum((y-my)*(x-mx))
            a1  = cxy/cxx
            a0  = my - mx*a1
            resi  = y - a0 - a1*x
            rslt[i,0,fill] = resi
        return rslt
        
    def purify_mono(self, maps):
        """purify function with customized maps input at single frequency
        
        Parameters
        ----------
        
        map : numpy.ndarray
            TQU maps at single frequency,
            in shape (3, N_pix).
            
        mask : numpy.ndarray
            mask map, in shape (N_pix, ).
            
        Returns
        -------
        B mode maps : numpy.ndarray
            B mode maps at various frequencies,
            in shape (N_pix, ).
        """
        assert (maps.shape == (3,self._npix))
        mask_sum = np.sum(self._mask[0])
        pix_list = np.arange(self._npix)
        fill = np.arange(self._npix)[np.where(self._mask[0]==1.)] # the pixel index of the available index
        rslt = np.zeros(self._npix,dtype=np.float64)
        # process at each frequency
        # get the template of E to B leakage
        Alm0 = hp.map2alm(maps) #alms of the masked maps
        B0 = hp.alm2map(Alm0[2],nside=self._nside,verbose=0)  # corrupted B map
        Alm0[0] = 0.
        Alm0[2] = 0.
        E0 = hp.alm2map(Alm0,nside=self._nside,verbose=0)  # TQU of corrupted E mode only
        E0[:,self._mask[0]==0.] = 0.  # re-mask
        Alm1 = hp.map2alm(E0)  # Alms of the TUQ from E-mode only
        B1 = hp.alm2map(Alm1[2],nside=self._nside,verbose=0)  # template of EB leakage
        # compute the residual of linear fit
        x = B1[fill]
        y = B0[fill]
        mx  = np.sum(x)/mask_sum
        my  = np.sum(y)/mask_sum
        cxx = np.sum((x-mx)*(x-mx))
        cxy = np.sum((y-my)*(x-mx))
        a1  = cxy/cxx
        a0  = my - mx*a1
        resi  = y - a0 - a1*x
        rslt[fill] = resi
        return rslt
