import logging as log
import numpy as np
from abspy.methods.abs import abssep
from abspy.tools.ps_estimator import pstimator
from abspy.tools.icy_decorator import icy


@icy
class abspipe(object):
    """
    The ABS pipeline class.
    
    Author:
    - Jian Yao (STJU)
    - Jiaxin Wang (SJTU) jiaxin.wang@sjtu.edu.cn
    """
    
    def __init__(self, signal, nfreq, nmap, nside, variance=None, mask=None, fwhms=None):
        """
        The ABS pipeline for extracting CMB power-spectrum band power,
        according to given measured sky maps at various frequency bands.
    
        Parameters
        ----------
    
        signal : numpy.ndarray
            Measured signal maps,
            should be arranged in shape: (frequency #, map #, pixel #).
        
        variance : numpy.ndarray
            Measured noise variance maps,
            should be arranged in shape: (frequency #, map #, pixel #).
            By default, no variance maps required.
        
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (1, pixel #).
    
        nfreq : integer
            Number of frequencies.
        
        nmap : integer
            Number of maps,
            if 1, taken as T maps only,
            if 2, taken as Q,U maps only,
        
        nside : integer
            HEALPix Nside of inputs.
            
        nsamp : integer
            Noise resampling size (hidden parameter, by default is 100).
            
        fwhms : list, tuple
            FWHM of gaussian beams for each frequency
        """
        log.debug('@ abspipe::__init__')
        #
        self.nfreq = nfreq
        self.nmap = nmap
        self.nside = nside
        self.fwhms = fwhms
        #
        self.signal = signal
        self.variance = variance
        self.mask = mask
        # method select dict with keys defined by (self._noise_flag, self._nmap)
        self._methodict = {(True,1): self.method_noisyT,
                           (True,2): self.method_noisyEB,
                           (False,1): self.method_pureT,
                           (False,2): self.method_pureEB}
        # resampling size
        self.nsamp = 100

    @property
    def signal(self):
        return self._signal
        
    @property
    def variance(self):
        return self._variance
        
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
    def fwhms(self):
        return self._fwhms
    
    @nfreq.setter
    def nfreq(self, nfreq):
        assert isinstance(nfreq, int)
        assert (nfreq > 0)
        self._nfreq = nfreq
        log.debug('number of frequencies'+str(self._nfreq))
        
    @nmap.setter
    def nmap(self, nmap):
        assert isinstance(nmap, int)
        assert (nmap > 0)
        self._nmap = nmap
        log.debug('number of maps'+str(self._nmap))
        
    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        self._npix = 12*nside**2
        log.debug('HEALPix Nside'+str(self._nside))
        
    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        assert (signal.shape == (self._nfreq,self._nmap,self._npix))
        self._signal = signal
        log.debug('singal maps loaded')
        
    @variance.setter
    def variance(self, variance):
        if variance is not None:
            assert isinstance(variance, np.ndarray)
            assert (variance.shape == (self._nfreq,self._nmap,self._npix))
            self._noise_flag = True
        else:
            self._noise_flag = False
        self._variance = variance
        log.debug('variance maps loaded')
        
    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones((1,self._npix),dtype=bool)
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = mask
        log.debug('mask map loaded')
        
    @nsamp.setter
    def nsamp(self, nsamp):
        assert isinstance(nsamp, int)
        self._nsamp = nsamp
        log.debug('resampling size set')
        
    @fwhms.setter
    def fwhms(self, fwhms):
        if fwhms is None:
            self._fwhms = [fwhms]*self._nfreq
        else:
            assert isinstance(fwhms, (list,tuple))
            assert (len(fwhms) == self._nfreq)
            self._fwhms = fwhms
        log.debug('fwhms loaded')
        
    def __call__(self, psbin, absbin, shift=0.0, threshold=0.0):
        """
        ABS pipeline class call function.
        
        Parameters
        ----------
        
        psbin : integer
            Number of angular modes in each bin,
            for conducting pseudo-PS estimation.
            
        absbin : integer
            Number of angular mode bins,
            for ABS method.
            
        shift : float
            ABS method shift parameter.
            
        threshold : float
            ABS method threshold parameter.
        
        Returns
        -------
        
        angular modes, target angular power spectrum : tuple of lists
        """
        log.debug('@ abspipe::__call__')
        assert isinstance(psbin, int)
        assert isinstance(absbin, int)
        assert (psbin > 0)
        assert (absbin > 0)
        # method selection
        return self._methodict[(self._noise_flag,self._nmap)](psbin, absbin, shift, threshold)
        
    def method_pureT(self, psbin, absbin, shift, threshold):
        """
        CMB T mode (band power) extraction without noise.
        
        Returns
        -------
        angular modes, T-mode PS
        """
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=5.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        trial = est.auto_t(self._signal[0,0].reshape(1,-1))
        ellist = list(trial[0])  # register angular modes
        #wsp = trial[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        # prepare total singal PS in the shape required by ABS method
        signal_ps_t = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto correlation
            stmp = est.auto_t(self._signal[i],fwhms=self._fwhms[i])
            # assign results
            for k in range(nell):
                signal_ps_t[k,i,i] = stmp[1][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                stmp = est.cross_t(np.vstack([self._signal[i],self._signal[j]]),fwhms=[self._fwhms[i],self._fwhms[j]])
                for k in range(nell):
                    signal_ps_t[k,i,j] = stmp[1][k]
                    signal_ps_t[k,j,i] = signal_ps_t[k,i,j]
        # send PS to ABS method
        safe_absbin = min(nell, absbin)
        spt_t = abssep(signal_ps_t,modes=ellist,bins=safe_absbin,shift=shift,threshold=threshold)
        return spt_t()
        
    def method_noisyT(self, psbin, absbin, shift, threshold):
        raw_rslt = self.method_noisyT_raw(psbin,absbin,shift,threshold)
        return (raw_rslt[0], list(np.mean(raw_rslt[1],axis=0)), list(np.std(raw_rslt[1],axis=0)))
        
    def method_noisyT_raw(self, psbin, absbin, shift, threshold):
        """
        CMB T mode (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, T-mode PS, T-mode PS std
        """
        # estimate noise PS and noise RMS
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=5.0,psbin=psbin)  # init PS estimator
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        ellist = list()
        for i in range(self._nfreq):
            tmp = est.auto_t(self._signal[0,0].reshape(1,-1),fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]  # register workspace
            ellist = list(tmp[0])  # register angular modes
            for j in range(i+1,self._nfreq):
                tmp = est.cross_t(self._signal[:2,0],fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        # allocate
        noise_ps_t = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_t = np.zeros((self._nsamp,nell,self._nfreq,self._nfreq),dtype=np.float64)
        noise_rms_t = np.zeros((nell,self._nfreq),dtype=np.float64)
        noise_map = np.zeros((2,self._npix),dtype=np.float64)
        signal_map = np.zeros((2,self._npix),dtype=np.float64)
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                mtmp = self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variance[i,0])
                noise_map[0] = mtmp
                signal_map[0] = self._signal[i,0] + mtmp
                # auto correlation
                ntmp = est.auto_t(noise_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = est.auto_t(signal_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(nell):
                    noise_ps_t[k,i,i] += ntmp[1][k]
                    noise_rms_t[k,i] += (tmp[1][k])**2
                    signal_ps_t[s,k,i,i] += stmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    mtmp = self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variance[j,0])
                    noise_map[1] = mtmp
                    signal_map[1] = self._signal[j,0] + mtmp
                    # cross correlation
                    ntmp = est.cross_t(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = est.cross_t(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(nell):
                        noise_ps_t[k,i,j] += ntmp[1][k]
                        noise_ps_t[k,j,i] += ntmp[1][k]
                        signal_ps_t[s,k,i,j] = stmp[1][k]
                        signal_ps_t[s,k,j,i] = stmp[1][k]
        # get noise PS mean and rms
        for l in range(nell):
            for i in range(self._nfreq):
                noise_ps_t[l,i,:] /= self._nsamp
                noise_rms_t[l,i] = np.sqrt(noise_rms_t[l,i]/self._nsamp - noise_ps_t[l,i,i]**2)
        # add signal map
        rslt_ell = list()
        rslt_Dt = list()
        for s in range(self._nsamp):
            # send PS to ABS method
            safe_absbin = min(nell, absbin)
            spt_t = abssep(signal_ps_t[s],noise_ps_t,noise_rms_t,modes=ellist,bins=safe_absbin,shift=shift,threshold=threshold)
            rslt_t = spt_t()
            rslt_ell = rslt_t[0]
            rslt_Dt += rslt_t[1]
        # get result
        return (rslt_ell, np.reshape(rslt_Dt, (self._nsamp,-1)))
    
    def method_pureEB(self, psbin, absbin, shift, threshold):
        """
        CMB E and B modes (band power) extraction without noise.
        
        Returns
        -------
        
        angular modes, E-mode PS, B-mode
        """
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=5.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        trial = est.auto_eb(self._signal[0])
        ellist = list(trial[0])  # register angular modes
        #wsp = trial[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        signal_ps_e = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_b = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto corr
            stmp = est.auto_eb(self._signal[i],fwhms=self._fwhms[i])
            # assign results
            for k in range(nell):
                signal_ps_e[k,i,i] = stmp[1][k]
                signal_ps_b[k,i,i] = stmp[2][k]
            # cross corr
            for j in range(i+1,self._nfreq):
                stmp = est.cross_eb(np.vstack([self._signal[i],self._signal[j]]),fwhms=[self._fwhms[i],self._fwhms[j]])
                for k in range(nell):
                    signal_ps_e[k,i,j] = stmp[1][k]
                    signal_ps_b[k,i,j] = stmp[2][k]
                    signal_ps_e[k,j,i] = stmp[1][k]
                    signal_ps_b[k,j,i] = stmp[2][k]
        # send PS to ABS method
        safe_absbin = min(nell, absbin)
        spt_e = abssep(signal_ps_e,modes=ellist,bins=safe_absbin,shift=shift,threshold=threshold)
        spt_b = abssep(signal_ps_b,modes=ellist,bins=safe_absbin,shift=shift,threshold=threshold)
        rslt_e = spt_e()
        rslt_b = spt_b()
        return (rslt_e[0], rslt_e[1], rslt_b[1])
        
    def method_noisyEB(self, psbin, absbin, shift, threshold):
        raw_rslt = self.method_noisyEB_raw(psbin,absbin,shift,threshold)
        return (raw_rslt[0], list(np.mean(raw_rslt[1],axis=0)), list(np.std(raw_rslt[1],axis=0)), list(np.mean(raw_rslt[2],axis=0)), list(np.std(raw_rslt[2],axis=0)))
        
    def method_noisyEB_raw(self, psbin, absbin, shift, threshold):
        """
        CMB E and B modes (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, E-mode PS, E-mode PS std, B-mode PS, B-mode PS std
        """
        # estimate noise PS and noise RMS
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=5.0,psbin=psbin)  # init PS estimator
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        ellist = list()
        for i in range(self._nfreq):
            tmp = est.auto_eb(self._signal[0],fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]  # register workspace
            ellist = list(tmp[0])  # register angular modes
            for j in range(i+1,self._nfreq):
                tmp = est.cross_eb(self._signal[:2].reshape(4,-1),fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        # allocate
        noise_ps_e = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        noise_ps_b = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_e = np.zeros((self._nsamp,nell,self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_b = np.zeros((self._nsamp,nell,self._nfreq,self._nfreq),dtype=np.float64)
        noise_rms_e = np.zeros((nell,self._nfreq),dtype=np.float64)
        noise_rms_b = np.zeros((nell,self._nfreq),dtype=np.float64)
        noise_map = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
        signal_map = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                mtmp = self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variance[i,0])
                noise_map[0] = mtmp
                signal_map[0] = self._signal[i,0] + mtmp
                mtmp = self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variance[i,1])
                noise_map[1] = mtmp
                signal_map[1] = self._signal[i,1] + mtmp
                # auto correlation
                ntmp = est.auto_eb(noise_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = est.auto_eb(signal_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(nell):
                    noise_ps_e[k,i,i] += ntmp[1][k]
                    noise_ps_b[k,i,i] += ntmp[2][k]
                    noise_rms_e[k,i] += (ntmp[1][k])**2
                    noise_rms_b[k,i] += (ntmp[2][k])**2
                    signal_ps_e[s,k,i,i] = stmp[1][k]
                    signal_ps_b[s,k,i,i] = stmp[2][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    mtmp = self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variance[j,0])
                    noise_map[2] = mtmp
                    signal_map[2] = self._signal[j,0] + mtmp
                    mtmp = self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variance[j,1])
                    noise_map[3] = mtmp
                    signal_map[3] = self._signal[j,1] + mtmp
                    # cross correlation
                    ntmp = est.cross_eb(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = est.cross_eb(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(nell):
                        noise_ps_e[k,i,j] += ntmp[1][k]
                        noise_ps_b[k,i,j] += ntmp[2][k]
                        noise_ps_e[k,j,i] += ntmp[1][k]
                        noise_ps_b[k,j,i] += ntmp[2][k]
                        signal_ps_e[s,k,i,j] = stmp[1][k]
                        signal_ps_e[s,k,j,i] = stmp[1][k]
                        signal_ps_b[s,k,i,j] = stmp[2][k]
                        signal_ps_b[s,k,j,i] = stmp[2][k]
        # get noise PS mean and rms
        for l in range(nell):
            for i in range(self._nfreq):
                noise_ps_e[l,i,:] /= self._nsamp
                noise_ps_b[l,i,:] /= self._nsamp
                noise_rms_e[l,i] = np.sqrt(noise_rms_e[l,i]/self._nsamp - noise_ps_e[l,i,i]**2)
                noise_rms_b[l,i] = np.sqrt(noise_rms_b[l,i]/self._nsamp - noise_ps_b[l,i,i]**2)
        # add signal map
        rslt_ell = list()
        rslt_De = list()
        rslt_Db = list()
        for s in range(self._nsamp):
            # send PS to ABS method
            safe_absbin = min(nell, absbin)
            spt_e = abssep(signal_ps_e[s],noise_ps_e,noise_rms_e,modes=ellist,bins=safe_absbin,shift=shift,threshold=threshold)
            spt_b = abssep(signal_ps_b[s],noise_ps_b,noise_rms_b,modes=ellist,bins=safe_absbin,shift=shift,threshold=threshold)
            rslt_e = spt_e()
            rslt_b = spt_b()
            rslt_ell = rslt_e[0]
            rslt_De += rslt_e[1]
            rslt_Db += rslt_b[1]
        # get result
        return (rslt_ell, np.reshape(rslt_De, (self._nsamp,-1)), np.reshape(rslt_Db, (self._nsamp,-1)))
