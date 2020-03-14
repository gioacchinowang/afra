"""
The ABS separator class.

Author:
- Jian Yao (STJU)
- Jiaxin Wang (SJTU) jiaxin.wang@sjtu.edu.cn
"""

import logging as log
import numpy as np
from copy import deepcopy
from abspy.tools.icy_decorator import icy


@icy
class abssep(object):
    
    def __init__(self, signal, noise=None, sigma=None, bins=None, modes=None, shift=0.0, threshold=0.0):
        """
        ABS separator class initialization function.
        
        Parameters:
        -----------
        
        singal : numpy.ndarray
            The total CROSS power-sepctrum matrix,
            with global size (N_modes, N_freq, N_freq).
            * N_freq: number of frequency bands
            * N_modes: number of angular modes
            
        noise : numpy.ndarray
            The ensemble averaged (instrumental) noise CROSS power-sepctrum,
            with global size (N_modes, N_freq, N_freq).
            * N_freq: number of frequency bands
            * N_modes: number of angular modes
            
        nrms : numpy.ndarray
            The RMS of ensemble (instrumental) noise AUTO power-spectrum,
            with global size (N_modes, N_freq).
            * N_freq: number of frequency bands
            * N_modes: number of angular modes
            
        bins : (positive) integer
            The bin width of angular modes.
            
        modes : list, tuple
            The list of angular modes of given power spectra.
            
        shift : (positive) float
            Global shift to the target power-spectrum,
            defined in Eq(3) of arXiv:1608.03707.
            
        threshold : (positive) float
            The threshold of signal to noise ratio, for information extraction.
        """
        log.debug('@ abs::__init__')
        #
        self.signal = signal
        self.noise = noise
        self.sigma = sigma
        # DO NOT CHENGE ORDER HERE
        self.modes = modes
        self.bins = bins
        #
        self.shift = shift
        self.threshold = threshold
        #
        self.noise_flag = not (self._noise is None or self._sigma is None)
        
    @property
    def signal(self):
        return self._signal
    
    @property
    def noise(self):
        return self._noise
    
    @property
    def sigma(self):
        return self._sigma
        
    @property
    def modes(self):
        return self._modes
        
    @property
    def bins(self):
        return self._bins
    
    @property
    def shift(self):
        return self._shift
    
    @property
    def threshold(self):
        return self._threshold
        
    @property
    def noise_flag(self):
        return self._noise_flag
        
    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        self._lsize = signal.shape[0]  # number of angular modes
        self._fsize = signal.shape[1]  # number of frequency bands
        assert (signal.shape[1] == signal.shape[2])
        self._signal = signal
        log.debug('signal cross-PS read')
        
    @noise.setter
    def noise(self, noise):
        if noise is None:
            log.debug('without noise cross-PS')
        else:
            assert isinstance(noise, np.ndarray)
            assert (noise.shape[0] == self._lsize)
            assert (noise.shape[1] == self._fsize)
            assert (noise.shape[1] == noise.shape[2])
            log.debug('noise cross-PS read')
        self._noise = noise
        
    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            log.debug('without noise RMS')
        else:
            assert isinstance(sigma, np.ndarray)
            assert (sigma.shape[0] == self._lsize)
            assert (sigma.shape[1] == self._fsize)
            log.debug('noise RMS auto-PS read')
        self._sigma = sigma
        
    @modes.setter
    def modes(self, modes):
        if modes is not None:
            assert isinstance(modes, (list,tuple))
            self._modes = modes
        else:  # by default modes start with 0
            self._modes = [*range(self._lsize)]
        log.debug('angular modes list set as '+str(self._modes))
        
    @bins.setter
    def bins(self, bins):
        assert isinstance(bins, int)
        assert (bins > 0 and bins <= self._lsize)
        self._bins = bins
        log.debug('angular mode bin width set as '+str(self._bins))
        
    @shift.setter
    def shift(self, shift):
        assert isinstance(shift, float)
        assert (shift >= 0)
        self._shift = shift
        log.debug('PS power shift set as '+str(self._shift))
        
    @threshold.setter
    def threshold(self, threshold):
        assert isinstance(threshold, float)
        assert (threshold >= 0)
        self._threshold = threshold
        log.debug('signal to noise threshold set as '+str(self._threshold))
        
    @noise_flag.setter
    def noise_flag(self, noise_flag):
        assert isinstance(noise_flag, bool)
        self._noise_flag = noise_flag
        log.debug('ABS with noise? '+str(self._noise_flag))
        
    @property
    def binell(self):
        """
        Central angular modes "ell" of binned average.
        
        Returns
        -------
        
        Central angular modes position : numpy.ndarray.
        """
        log.debug('@ abs::binell')
        _lnew = list()
        _lres = self._lsize%self._bins
        _lmod = self._lsize//self._bins
        # binned average for each single spectrum
        for i in range(self._bins):
            _begin = min(_lres,i)+i*_lmod
            _end = min(_lres,i) + (i+1)*_lmod + int(i < _lres)
            _lnew.append(0.5*(self._modes[_begin]+self._modes[_end-1]))
        return _lnew

    def bincps(self, cps):
        """
        Binned average of CROSS-power-spectrum and convert it into CROSS-Dl (band power).
        
        Parameters
        ----------
        
        cps : numpy.ndarray
            cross power spectrum
            
        Returns
        -------
            
        CROSS-Dl with bin average : numpy.ndarray
        """
        log.debug('@ abs::bincps')
        assert isinstance(cps, np.ndarray)
        assert (cps.shape[0] == self._lsize)
        assert (cps.shape[1] == self._fsize)
        assert (cps.shape[1] == cps.shape[2])
        _lres = self._lsize%self._bins
        _lmod = self._lsize//self._bins
        _result = np.empty((self._bins, self._fsize, self._fsize))
        _cps = deepcopy(cps)  # avoid mem issue
        # binned average for each single spectrum
        for i in range(self._bins):
            _begin = min(_lres,i)+i*_lmod
            _end = min(_lres,i) + (i+1)*_lmod + int(i < _lres)
            # convert Cl into Dl for each single spectrum
            _effl = 0.5*(self._modes[_begin]+self._modes[_end-1])
            _result[i,:,:] = np.mean(_cps[_begin:_end,:,:], axis=0)*0.5*_effl*(_effl+1)/np.pi
        return _result
    
    def binaps(self, aps):
        """
        Binned average of AUTO-power-spectrum Cl and convert it into AUTO-Dl (band power).
        
        Parameters
        ----------
        
        aps : numpy.ndarray
            auto power spectrum
        
        Returns
        -------
        
        AUTO-Dl with binned average : numpy.ndarray
        """
        log.debug('@ abs::binaps')
        assert isinstance(aps, np.ndarray)
        assert (aps.shape[0] == self._lsize)
        assert (aps.shape[1] == self._fsize)
        _lres = self._lsize%self._bins
        _lmod = self._lsize//self._bins
        # allocate results
        _result = np.empty((self._bins, self._fsize))
        _aps = deepcopy(aps)
        # binned average for each single spectrum
        for i in range(self._bins):
            _begin = min(_lres,i)+i*_lmod
            _end = min(_lres,i) + (i+1)*_lmod + int(i < _lres)
            _effl = 0.5*(self._modes[_begin]+self._modes[_end-1])
            # convert Cl into Dl for each single spectrum
            _result[i,:] = np.mean(_aps[_begin:_end,:], axis=0)*0.5*_effl*(_effl+1)/np.pi
        return _result
    
    def __call__(self):
        log.debug('@ abs::__call__')
        return self.run()
        
    def run(self):
        """
        ABS separator class call function.
        
        Returns
        -------
        angular modes, target angular power spectrum : (list, list)
        """
        log.debug('@ abs::run')
        # binned average, converted to band power
        _DL = self.bincps(self._signal)
        if (self._noise_flag):
            _NL = self.bincps(self._noise)
            _RL = self.binaps(self._sigma)
        # prepare CMB f(ell, freq)
        _f = np.ones((self._bins,self._fsize), dtype=np.float64)
        if (self._noise_flag):
            # Dl_ij = Dl_ij/sqrt(sigma_li,sigma_lj) + shift*f_li*f_lj
            for l in range(self._bins):
                for i in range(self._fsize):
                    _f[l,i] /= _RL[l,i]  # rescal f according to noise RMS
                    for j in range(self._fsize):
                        _DL[l,i,j] = (_DL[l,i,j] - _NL[l,i,j])/np.sqrt(_RL[l,i]*_RL[l,j]) + self._shift*_f[l,i]*_f[l,j]
        else:
            # Dl_ij = Dl_ij + shift*f_li*f_lj
            for l in range(self._bins):
                for i in range(self._fsize):
                    for j in range(self._fsize):
                        _DL[l,i,j] += self._shift*_f[l,i]*_f[l,j]
        # find eign at each angular mode
        _BL = list()
        for ell in range(self._bins):
            # eigvec[:,i] corresponds to eigval[i]
            # note that eigen values may be complex
            eigval, eigvec = np.linalg.eig(_DL[ell])
            log.debug('@ abs::__call__, angular mode '+str(self.binell[ell])+' with eigen vals '+str(eigval))
            for i in range(self._fsize):
                eigvec[:,i] /= np.linalg.norm(eigvec[:,i])**2
            _tmp = 0
            for i in range(self._fsize):
                if eigval[i] >= self._threshold:
                    _G = np.dot(_f[ell], eigvec[:,i])
                    _tmp += (_G**2/eigval[i])
            _BL.append(1.0/_tmp - self._shift)
        return (self.binell, _BL)
        
