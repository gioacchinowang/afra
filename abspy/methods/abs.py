import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy
from abspy.tools.binning import binell, binaps, bincps

@icy
class abssep(object):
    """
    The ABS separator class.

    Author:
    - Jian Yao (STJU)
    - Jiaxin Wang (SJTU) jiaxin.wang@sjtu.edu.cn
    """
    
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
            
        sigma : numpy.ndarray
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
    
    def __call__(self):
        """
        ABS separator class call function.
        
        Returns
        -------
        
        angular modes, target angular power spectrum : (list, list)
        """
        log.debug('@ abs::__call__')
        # binned average, converted to band power
        DL = bincps(self._signal,self._modes,self._bins)
        if (self._noise_flag):
            NL = bincps(self._noise,self._modes,self._bins)
            RL = binaps(self._sigma,self._modes,self._bins)
        # prepare CMB f(ell, freq)
        f = np.ones((self._bins,self._fsize), dtype=np.float64)
        if self._noise_flag:
            # Dl_ij = Dl_ij/sqrt(sigma_li,sigma_lj) + shift*f_li*f_lj
            for l in range(self._bins):
                for i in range(self._fsize):
                    f[l,i] /= np.sqrt(RL[l,i])  # rescal f according to noise RMS
                for i in range(self._fsize):
                    for j in range(self._fsize):
                        DL[l,i,j] = (DL[l,i,j] - NL[l,i,j])/np.sqrt(RL[l,i]*RL[l,j]) + self._shift*f[l,i]*f[l,j]
        else:
            # Dl_ij = Dl_ij + shift*f_li*f_lj
            for l in range(self._bins):
                for i in range(self._fsize):
                    for j in range(self._fsize):
                        DL[l,i,j] += self._shift*f[l,i]*f[l,j]
        # find eign at each angular mode
        BL = list()
        for ell in range(self._bins):
            # eigvec[:,i] corresponds to eigval[i]
            # note that eigen values may be complex
            eigval, eigvec = np.linalg.eig(DL[ell])
            eigval = np.abs(eigval)
            #assert (all(v > 0 for v in eigval))
            for i in range(self._fsize):
                eigvec[:,i] /= np.linalg.norm(eigvec[:,i])**2
            tmp = 0
            for i in range(self._fsize):
                if eigval[i] >= self._threshold:
                    G = np.dot(f[ell], eigvec[:,i])
                    tmp += (G**2/eigval[i])
            BL.append(1.0/tmp - self._shift)
        return (binell(self._modes, self._bins), BL)
        
