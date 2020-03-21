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
    
    def __init__(self, signal, nfreq, nmap, nside, variance=None, mask=None):
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
        """
        log.debug('@ abspipe::__init__')
        #
        self.nfreq = nfreq
        self.nmap = nmap
        self.nside = nside
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
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        trial = est.auto_t(self._signal[0,0].reshape(1,-1))
        ellist = list(trial[0])  # register angular modes
        wsp = trial[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        # prepare total singal PS in the shape required by ABS method
        signal_ps_t = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto correlation
            tmp = est.auto_t(self._signal[i],wsp)
            # assign results
            for k in range(nell):
                signal_ps_t[k,i,i] = tmp[1][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                tmp = est.cross_t(np.vstack([self._signal[i],self._signal[j]]),wsp)
                for k in range(nell):
                    signal_ps_t[k,i,j] = tmp[1][k]
                    signal_ps_t[k,j,i] = signal_ps_t[k,i,j]
        # send PS to ABS method
        spt_t = abssep(signal_ps_t,modes=ellist,bins=absbin,shift=shift,threshold=threshold)
        return spt_t()
        
    def method_noisyT(self, psbin, absbin, shift, threshold):
        """
        CMB T mode (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, T-mode PS, T-mode PS std
        """
        # estimate noise PS and noise RMS
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        trial = est.auto_t(self._signal[0,0].reshape(1,-1))
        ellist = list(trial[0])  # register angular modes
        wsp = trial[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        # allocate
        noise_ps_t = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        noise_rms_t = np.zeros((nell,self._nfreq),dtype=np.float64)
        noise = np.zeros((2,self._npix),dtype=np.float64)
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # 1st noise realization
                for p in range(self._npix):
                    if self._mask[0,p]:
                        noise[0,p] = np.random.normal(0,np.sqrt(self._variance[i,0,p]))
                # auto correlation
                tmp = est.auto_t(noise[0].reshape(1,-1),wsp)
                # assign results
                for k in range(nell):
                    noise_ps_t[k,i,i] += tmp[1][k]
                    noise_rms_t[k,i] += (tmp[1][k])**2
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # 2nd noise realization
                    for p in range(self._npix):
                        if self._mask[0,p]:
                            noise[1,p] = np.random.normal(0,np.sqrt(self._variance[j,0,p]))
                    tmp = est.cross_t(noise,wsp)
                    for k in range(nell):
                        noise_ps_t[k,i,j] += tmp[1][k]
                        noise_ps_t[k,j,i] += noise_ps_t[k,i,j]
        # get noise PS mean and rms
        for l in range(nell):
            for i in range(self._nfreq):
                noise_ps_t[l,i,:] /= self._nsamp
                noise_rms_t[l,i] = np.sqrt(noise_rms_t[l,i]/self._nsamp - noise_ps_t[l,i,i]**2)
        # add signal map
        rslt_ell = list()
        rslt_Dt = list()
        for s in range(self._nsamp):
            signal_ps_t = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            for i in range(self._nfreq):
                # 1st noise realization
                for p in range(self._npix):
                    if self._mask[0,p]:
                        noise[0,p] = self._signal[i,0,p] + np.random.normal(0,np.sqrt(self._variance[i,0,p]))
                # auto correlation
                tmp = est.auto_t(noise[0].reshape(1,-1),wsp)  # noise += signal
                # assign results
                for k in range(nell):
                    signal_ps_t[k,i,i] = tmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # 2nd noise realization
                    for p in range(self._npix):
                        if self._mask[0,p]:
                            noise[1,p] = self._signal[j,0,p] + np.random.normal(0,np.sqrt(self._variance[j,0,p]))
                    tmp = est.cross_t(noise,wsp)  # noise += signal
                    for k in range(nell):
                        signal_ps_t[k,i,j] = tmp[1][k]
                        signal_ps_t[k,j,i] = signal_ps_t[k,i,j]
            # send PS to ABS method
            spt_t = abssep(signal_ps_t,noise_ps_t,noise_rms_t,modes=ellist,bins=absbin,shift=shift,threshold=threshold)
            rslt_t = spt_t()
            rslt_ell = rslt_t[0]
            rslt_Dt += rslt_t[1]
        # get result mean and std
        Dt_array = np.reshape(rslt_Dt, (self._nsamp,-1))
        return (rslt_ell, list(np.mean(Dt_array,axis=0)), list(np.std(Dt_array,axis=0)))
    
    def method_pureEB(self, psbin, absbin, shift, threshold):
        """
        CMB E and B modes (band power) extraction without noise.
        
        Returns
        -------
        
        angular modes, E-mode PS, B-mode
        """
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        trial = est.auto_eb(self._signal[0])
        ellist = list(trial[0])  # register angular modes
        wsp = trial[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        signal_ps_e = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        signal_ps_b = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto corr
            tmp = est.auto_eb(self._signal[i],wsp)
            # assign results
            for k in range(nell):
                signal_ps_e[k,i,i] = tmp[1][k]
                signal_ps_b[k,i,i] = tmp[2][k]
            # cross corr
            for j in range(i+1,self._nfreq):
                tmp = est.cross_eb(np.vstack([self._signal[i],self._signal[j]]),wsp)
                for k in range(nell):
                    signal_ps_e[k,i,j] = tmp[1][k]
                    signal_ps_b[k,i,j] = tmp[2][k]
                    signal_ps_e[k,j,i] = signal_ps_e[k,i,j]
                    signal_ps_b[k,j,i] = signal_ps_b[k,i,j]
        # send PS to ABS method
        spt_e = abssep(signal_ps_e,modes=ellist,bins=absbin,shift=shift,threshold=threshold)
        spt_b = abssep(signal_ps_b,modes=ellist,bins=absbin,shift=shift,threshold=threshold)
        rslt_e = spt_e()
        rslt_b = spt_b()
        return (rslt_e[0], rslt_e[1], rslt_b[1])
        
    def method_noisyEB(self, psbin, absbin, shift, threshold):
        """
        CMB E and B modes (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, E-mode PS, E-mode PS std, B-mode PS, B-mode PS std
        """
        # estimate noise PS and noise RMS
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        trial = est.auto_eb(self._signal[0])
        ellist = list(trial[0])  # register angular modes
        wsp = trial[-1]  # register workspace
        nell = len(ellist)  # know the number of angular modes
        # allocate
        noise_ps_e = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        noise_ps_b = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
        noise_rms_e = np.zeros((nell,self._nfreq),dtype=np.float64)
        noise_rms_b = np.zeros((nell,self._nfreq),dtype=np.float64)
        noise = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # 1st noise realization
                for p in range(self._npix):
                    if self._mask[0,p]:
                        noise[0,p] = np.random.normal(0,np.sqrt(self._variance[i,0,p]))
                        noise[1,p] = np.random.normal(0,np.sqrt(self._variance[i,1,p]))
                # auto correlation
                tmp = est.auto_eb(noise[:2],wsp)
                # assign results
                for k in range(nell):
                    noise_ps_e[k,i,i] += tmp[1][k]
                    noise_ps_b[k,i,i] += tmp[2][k]
                    noise_rms_e[k,i] += (tmp[1][k])**2
                    noise_rms_b[k,i] += (tmp[2][k])**2
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # 2nd noise realization
                    for p in range(self._npix):
                        if self._mask[0,p]:
                            noise[2,p] = np.random.normal(0,np.sqrt(self._variance[j,0,p]))
                            noise[3,p] = np.random.normal(0,np.sqrt(self._variance[j,1,p]))
                    tmp = est.cross_eb(noise,wsp)
                    for k in range(nell):
                        noise_ps_e[k,i,j] += tmp[1][k]
                        noise_ps_b[k,i,j] += tmp[2][k]
                        noise_ps_e[k,j,i] += noise_ps_e[k,i,j]
                        noise_ps_b[k,j,i] += noise_ps_b[k,i,j]
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
            signal_ps_e = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            signal_ps_b = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            for i in range(self._nfreq):
                # 1st noise realization
                for p in range(self._npix):
                    if self._mask[0,p]:
                        noise[0,p] = self._signal[i,0,p] + np.random.normal(0,np.sqrt(self._variance[i,0,p]))
                        noise[1,p] = self._signal[i,1,p] + np.random.normal(0,np.sqrt(self._variance[i,1,p]))
                # auto correlation
                tmp = est.auto_eb(noise[:2],wsp)  # noise += signal
                # assign results
                for k in range(nell):
                    signal_ps_e[k,i,i] = tmp[1][k]
                    signal_ps_b[k,i,i] = tmp[2][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # 2nd noise realization
                    for p in range(self._npix):
                        if self._mask[0,p]:
                            noise[2,p] = self._signal[j,0,p] + np.random.normal(0,np.sqrt(self._variance[j,0,p]))
                            noise[3,p] = self._signal[j,1,p] + np.random.normal(0,np.sqrt(self._variance[j,1,p]))
                    tmp = est.cross_eb(noise,wsp)  # noise += signal
                    for k in range(nell):
                        signal_ps_e[k,i,j] = tmp[1][k]
                        signal_ps_b[k,i,j] = tmp[2][k]
                        signal_ps_e[k,j,i] = signal_ps_e[k,i,j]
                        signal_ps_b[k,j,i] = signal_ps_b[k,i,j]
            # send PS to ABS method
            spt_e = abssep(signal_ps_e,noise_ps_e,noise_rms_e,modes=ellist,bins=absbin,shift=shift,threshold=threshold)
            spt_b = abssep(signal_ps_b,noise_ps_b,noise_rms_b,modes=ellist,bins=absbin,shift=shift,threshold=threshold)
            rslt_e = spt_e()
            rslt_b = spt_b()
            rslt_ell = rslt_e[0]
            rslt_De += rslt_e[1]
            rslt_Db += rslt_b[1]
        # get result mean and std
        De_array = np.reshape(rslt_De, (self._nsamp,-1))
        Db_array = np.reshape(rslt_Db, (self._nsamp,-1))
        return (rslt_ell, list(np.mean(De_array,axis=0)), list(np.std(De_array,axis=0)), list(np.mean(Db_array,axis=0)), list(np.std(Db_array,axis=0)))
