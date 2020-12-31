import numpy as np
from afra.tools.icy_decorator import icy


@icy
class abssep(object):
    
    def __init__(self, data, noise=None, sigma=None, shift=None, threshold=None):
        """
        ABS class initialization function.
        
        Parameters:
        -----------
        
        data : numpy.ndarray
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
            
        shift : (positive) numpy.ndarray or None
            Global shift to the target power-spectrum,
            defined in Eq(3) of arXiv:1608.03707.
            
        threshold : (positive) float or None
            If None, no threshold is taken.
            The threshold of signal to noise ratio, for information extraction.
        """
        self.data = data
        self.noise = noise
        self.sigma = sigma
        self.shift = shift
        self.threshold = threshold
        self.noise_flag = not (self._noise is None or self._sigma is None)

    @property
    def data(self):
        return self._data

    @property
    def noise(self):
        return self._noise

    @property
    def sigma(self):
        return self._sigma

    @property
    def shift(self):
        return self._shift

    @property
    def threshold(self):
        return self._threshold

    @property
    def noise_flag(self):
        return self._noise_flag
        
    @data.setter
    def data(self, data):
        assert isinstance(data, np.ndarray)
        self._lsize = data.shape[0]  # number of angular modes
        self._fsize = data.shape[1]  # number of frequency bands
        assert (data.shape[1] == data.shape[2])
        if (np.isnan(data).any()):
            raise ValueError('encounter nan')
        self._data = data.copy()

    @noise.setter
    def noise(self, noise):
        if noise is None:
            self._noise = None
        else:
            assert isinstance(noise, np.ndarray)
            assert (noise.shape[0] == self._lsize)
            assert (noise.shape[1] == self._fsize)
            assert (noise.shape[1] == noise.shape[2])
            if (np.isnan(noise).any()):
                raise ValueError('encounter nan')
            self._noise = noise.copy()

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            assert isinstance(sigma, np.ndarray)
            assert (sigma.shape[0] == self._lsize)
            assert (sigma.shape[1] == self._fsize)
            if (np.isnan(sigma).any()):
                raise ValueError('encounter nan')
            self._sigma = sigma.copy()

    @shift.setter
    def shift(self, shift):
        if shift is not None:
            assert isinstance(shift, np.ndarray)
            assert (len(shift) == self._lsize)
            self._shift = shift.copy()
        else:
            self._shift = np.zeros(self._lsize)

    @threshold.setter
    def threshold(self, threshold):
        if threshold is not None:
            assert isinstance(threshold, float)
        self._threshold = threshold

    @noise_flag.setter
    def noise_flag(self, noise_flag):
        assert isinstance(noise_flag, bool)
        self._noise_flag = noise_flag

    def run(self):
        # binned average, converted to band power
        DL = self._data.copy()
        RL = np.ones((self._lsize,self._fsize),dtype=np.float64)
        RL_tensor = np.ones((self._lsize,self._fsize,self._fsize),dtype=np.float64)
        if self._noise_flag:
            DL -= self._noise  # DL = DL - NL
            RL = np.sqrt(self._sigma)
            RL_tensor = np.einsum('ij,ik->ijk',RL,RL)
        # prepare CMB f(ell, freq)
        f = np.ones((self._lsize,self._fsize), dtype=np.float64)/RL
        f_tensor = np.ones((self._lsize,self._fsize,self._fsize), dtype=np.float64)/RL_tensor
        # Dl_ij = Dl_ij/sqrt(sigma_li,sigma_lj) + shift*f_li*f_lj/sqrt(sigma_li,sigma_lj)
        DL = DL/RL_tensor + np.einsum('ijk,i->ijk',f_tensor,self._shift)
        # find eign at each angular mode
        BL = np.zeros(self._lsize)
        for ell in range(self._lsize):
            # eigvec[:,i] corresponds to eigval[i]
            eigval, eigvec = np.linalg.eig(DL[ell])
            tmp = 0
            if self._threshold is None:
                for i in range(self._fsize):
                    G = np.dot(f[ell], eigvec[:,i])
                    tmp += (G**2/eigval[i])
            else:
                assert (any(eigval > self._threshold))
                for i in range(self._fsize):
                    if eigval[i] > self._threshold:
                        G = np.dot(f[ell], eigvec[:,i])
                        tmp += (G**2/eigval[i])
            BL[ell] = (1.0/tmp - self._shift[ell])
        return BL

    def run_info(self):
        # binned average, converted to band power
        DL = self._data.copy()
        RL = np.ones((self._lsize,self._fsize),dtype=np.float64)
        RL_tensor = np.ones((self._lsize,self._fsize,self._fsize),dtype=np.float64)
        if self._noise_flag:
            DL -= self._noise  # DL = DL - NL
            RL = np.sqrt(self._sigma)
            RL_tensor = np.einsum('ij,ik->ijk',RL,RL)
        # prepare CMB f(ell, freq)
        f = np.ones((self._lsize,self._fsize), dtype=np.float64)/RL
        f_tensor = np.ones((self._lsize,self._fsize,self._fsize), dtype=np.float64)/RL_tensor
        # Dl_ij = Dl_ij/sqrt(sigma_li,sigma_lj) + shift*f_li*f_lj/sqrt(sigma_li,sigma_lj)
        DL = DL/RL_tensor + np.einsum('ijk,i->ijk',f_tensor,self._shift)
        # find eign at each angular mode
        info = dict()
        for ell in range(self._lsize):
            # eigvec[:,i] corresponds to eigval[i]
            eigval, eigvec = np.linalg.eig(DL[ell])
            info[ell] = (eigval, eigvec)
        return info
