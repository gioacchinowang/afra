"""
The ABS pipeline class.

Author:
- Jian Yao (STJU)
- Jiaxin Wang (SJTU) jiaxin.wang@sjtu.edu.cn
"""

import logging as log
import numpy as np
from copy import deepcopy
from abspy.methods.abs import abssep
from abspy.tools.ps_estimator import pstimator
from abspy.tools.icy_decorator import icy


@icy
class abspipe(object):
    
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
            if 3, taken as T,Q,U maps.
        
        nside : integer
            HEALPix Nside of inputs.
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
        # method select dict with keys defined by (self._noise_flat, self._nmap)
        self._methodict = {(True,1): self.method_noisyT,
                           (True,2): self.method_noisyEB,
                           (True,3): self.method_noisyTEB,
                           (False,1): self.method_pureT,
                           (False,2): self.method_pureEB,
                           (False,3): self.method_pureTEB}

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
        
    def __call__(self, psbin, absbin, shift=0.0, threshold=0.0):
        log.debug('@ abspipe::__call__')
        return self.run(psbin, absbin, shift, threshold)
        
    def run(self, psbin, absbin, shift, threshold):
        """
        ABS pipeline class call function
        
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
        angular modes, target angular power spectrum : (list, list)
        """
        log.debug('@ abspipe::run')
        assert isinstance(psbin, int)
        assert isinstance(absbin, int)
        assert (psbin > 0)
        assert (absbin > 0)
        # method selection
        return self._methodict[(self._noise_flag,self._nmap)](psbin, absbin, shift, threshold)
        
    def method_pureT(self, psbin, absbin, shift, threshold):
        _est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        _trial = _est.auto_t(self._signal[0,0].reshape(1,-1))
        _ellist = list(_trial[0])  # register angular modes
        _nell = len(_ellist)  # know the number of angular modes
        # prepare total singal PS in the shape required by ABS method
        _signal_ps_t = np.zeros((_nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto correlation
            _tmp = _est.auto_t(self._signal[i])
            # assign results
            for k in range(_nell):
                _signal_ps_t[k,i,i] = _tmp[1][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                _tmp = _est.cross_t(np.vstack([self._signal[i],self._signal[j]]))
                for k in range(_nell):
                    _signal_ps_t[k,i,j] = _tmp[1][k]
                    _signal_ps_t[k,j,i] = _signal_ps_t[k,i,j]
        # send PS to ABS method
        _spt_t = abssep(_signal_ps_t,modes=_ellist,bins=absbin,shift=shift,threshold=threshold)
        return _spt_t()
    
    def method_pureEB(self, psbin, absbin, shift, threshold):
        _est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        _trial = _est.auto_t(self._signal[0,0].reshape(1,-1))
        _ellist = list(_trial[0])  # register angular modes
        _nell = len(_ellist)  # know the number of angular modes
        _signal_ps_e = np.zeros((_nell,self._nfreq,self._nfreq),dtype=np.float64)
        _signal_ps_b = np.zeros((_nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto corr
            _tmp = _est.auto_eb(self._signal[i])
            # assign results
            for k in range(_nell):
                _signal_ps_e[k,i,i] = _tmp[1][k]
                _signal_ps_b[k,i,i] = _tmp[2][k]
            # cross corr
            for j in range(i+1,self._nfreq):
                _tmp = _est.cross_eb(np.vstack([self._signal[i],self._signal[j]]))
                for k in range(_nell):
                    _signal_ps_e[k,i,j] = _tmp[1][k]
                    _signal_ps_b[k,i,j] = _tmp[2][k]
                    _signal_ps_e[k,j,i] = _signal_ps_e[k,i,j]
                    _signal_ps_b[k,j,i] = _signal_ps_b[k,i,j]
        # send PS to ABS method
        _spt_e = abssep(_signal_ps_e,modes=_ellist,bins=absbin,shift=shift,threshold=threshold)
        _spt_b = abssep(_signal_ps_b,modes=_ellist,bins=absbin,shift=shift,threshold=threshold)
        _rslt_e = _spt_e()
        _rslt_b = _spt_b()
        return (_rslt_e[0], _rslt_e[1], _rslt_b[1])
    
    def method_pureTEB(self, psbin, absbin, shift, threshold):
        _est = pstimator(nside=self._nside,mask=self._mask,aposcale=1.0,psbin=psbin)  # init PS estimator
        # run a trial PS estimation
        _trial = _est.auto_t(self._signal[0,0].reshape(1,-1))
        _ellist = list(_trial[0])  # register angular modes
        _nell = len(_ellist)  # know the number of angular modes
        _signal_ps_t = np.zeros((_nell,self._nfreq,self._nfreq),dtype=np.float64)
        _signal_ps_e = np.zeros((_nell,self._nfreq,self._nfreq),dtype=np.float64)
        _signal_ps_b = np.zeros((_nell,self._nfreq,self._nfreq),dtype=np.float64)
        for i in range(self._nfreq):
            # auto corr
            _tmp = _est.auto_teb(self._signal[i])
            # assign results
            for k in range(_nell):
                _signal_ps_t[k,i,i] = _tmp[1][k]
                _signal_ps_e[k,i,i] = _tmp[2][k]
                _signal_ps_b[k,i,i] = _tmp[3][k]
            # cross corr
            for j in range(i+1,self._nfreq):
                _tmp = _est.cross_teb(np.vstack([self._signal[i],self._signal[j]]))
                for k in range(_nell):
                    _signal_ps_t[k,i,j] = _tmp[1][k]
                    _signal_ps_e[k,i,j] = _tmp[2][k]
                    _signal_ps_b[k,i,j] = _tmp[3][k]
                    _signal_ps_t[k,j,i] = _signal_ps_t[k,i,j]
                    _signal_ps_e[k,j,i] = _signal_ps_e[k,i,j]
                    _signal_ps_b[k,j,i] = _signal_ps_b[k,i,j]
        # send PS to ABS method
        _spt_t = abssep(_signal_ps_t,modes=_ellist,bins=absbin,shift=shift,threshold=threshold)
        _spt_e = abssep(_signal_ps_e,modes=_ellist,bins=absbin,shift=shift,threshold=threshold)
        _spt_b = abssep(_signal_ps_b,modes=_ellist,bins=absbin,shift=shift,threshold=threshold)
        _rslt_t = _spt_t()
        _rslt_e = _spt_e()
        _rslt_b = _spt_b()
        return (_rslt_t[0], _rslt_t[1], _rslt_e[1], _rslt_b[1])
        
    def method_noisyT(self, psbin, absbin, shift, threshold):
        pass
        
    def method_noisyEB(self, psbin, absbin, shift, threshold):
        pass

    def method_noisyTEB(self, psbin, absbin, shift, threshold):
        pass
