import logging as log
import numpy as np
import dynesty
from abspy.tools.fg_model import syncmodel, dustmodel, syncdustmodel
from abspy.tools.ps_estimator import pstimator
from abspy.tools.icy_decorator import icy

@icy
class tpfpipe(object):
    """
    The template fitting pipeline class.
    
    without template:
        D_b,l at effective ells
        D_s,l at effective ells
        D_d,l at effective ells
        foreground frequency scaling parameters
        foregorund cross-corr parameters
        
    with low+high frequency templates:
        D_b,l at effective ells
        foreground frequency scaling parameters
        foregorund cross-corr parameters
    """
    
    def __init__(self, singal, freq, nmap, nside, nbin=10, variance=None, mask=None, fwhms=None, sampling_opt=dict()):
        """
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
            
        freq : list, tuple
            List of frequencies.
            
        nmap : int
            Number of maps,
            if 1, taken as T maps only,
            if 2, taken as Q,U maps only,
            
        nside : int
            HEALPix Nside
            
        fwhms : list, tuple
            FWHM of gaussian beams for each frequency
            
        sampling_opt : dict
            Dynesty smapling options
        """
        log.debug('@ tpfpipeline::__init__')
        self.freq = freq
        self.nmap = nmap
        self.nside = nside
        self.fwhms = fwhms
        self.nbin = nbin
        #
        self.signal = signal
        self.variance = variance
        self.mask = mask
        # method select dict with keys defined by (self._noise_flag, self._nmap)
        self._methodict = {(1): self.method_T,
                           (2): self.method_EB}
        # sampling optinos
        self.sampling_opt = sampling_opt
                           
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
    def freq(self):
        return self._freq
        
    @property
    def nside(self):
        return self._nside
    
    @property
    def nmap(self):
        return self._nmap
        
    @property
    def fwhms(self):
        return self._fwhms
        
    @property
    def nbin(self):
        return self._nbin
        
    @property
    def sampling_opt(self):
        return self._sampling_opt
        
    @property
    def prior(self, cube):
        """flat prior"""
        return cube
        
    @property
    def nparam(self):
        
        return self._nbin
        
    @sampling_opt.setter
    def sampling_opt(self, opt):
        assert isinstance(opt, dict)
        self._sampling_opt = opt
        
    @nfreq.setter
    def freq(self, freq):
        assert isinstance(freq, (list,tuple))
        self._freq = freq
        self._nfreq = len(freq)
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
        
     def __call__(self, kwargs=dict()):
        """
        Parameters
        ----------
        kwargs : dict
            extra input argument controlling sampling process
            i.e., 'dlogz' for stopping criteria

        Returns
        -------
        Dynesty sampling results
        """
        log.debug('@ tpfit_pipeline::__call__')
        # init dynesty
        sampler = dynesty.NestedSampler(self._core_likelihood,
                                        self.prior,
                                        len(self.active_parameters),
                                        **self.sampling_opt)
        sampler.run_nested(**kwargs)
        return sampler.results
        
    @nbin.setter
    def nbin(self, nbin):
        assert isinstance(nbin, int)
        return self._nbin = nbin
        
    def _core_likelihood(self, cube):
        """
        core log-likelihood calculator
        cube remains the same on each node
        now self._simulator will work on each node and provide multiple ensemble size

        Parameters
        ----------
        cube
            list of variable values

        Returns
        -------
        log-likelihood value
        """
        log.debug('@ tpfit_pipeline::_core_likelihood')
        # security boundary check
        if np.any(cube > 1.) or np.any(cube < 0.):
            log.debug('cube %s requested. returned most negative possible number' % str(cube))
            return np.nan_to_num(-np.inf)
        # return active variables from pymultinest cube to factories
        # and then generate new field objects
        head_idx = int(0)
        tail_idx = int(0)
        field_list = tuple()
        # random seeds manipulation
        self._randomness()
        # the ordering in factory list and variable list is vital
        for factory in self._factory_list:
            variable_dict = dict()
            tail_idx = head_idx + len(factory.active_parameters)
            factory_cube = cube[head_idx:tail_idx]
            for i, av in enumerate(factory.active_parameters):
                variable_dict[av] = factory_cube[i]
            field_list += (factory.generate(variables=variable_dict,
                                            ensemble_size=self._ensemble_size,
                                            ensemble_seeds=self._ensemble_seeds),)
            log.debug('create '+factory.name+' field')
            head_idx = tail_idx
        assert(head_idx == len(self._active_parameters))
        observables = self._simulator(field_list)
        # apply mask
        observables.apply_mask(self.likelihood.mask_dict)
        # add up individual log-likelihood terms
        current_likelihood = self.likelihood(observables)
        # check likelihood value until negative (or no larger than given threshold)
        if self._check_threshold and current_likelihood > self._likelihood_threshold:
            raise ValueError('log-likelihood beyond threashould')
        return current_likelihood * self.likelihood_rescaler
