import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy
from abspy.tools.fg_models import fgmodel, syncdustmodel
from abspy.tools.bg_models import bgmodel, cmbmodel
import pymultinest


@icy
class tpfit(object):
    """
    The template fitting class.
    
    Parameters
    ----------
    
    signal : numpy.ndarray
        signal cross PS bandpower in shape (# mode, # freq, # freq)
        
    noise : numpy.ndarray
        noise cross PS bandpower in shape (# mode, # freq, # freq)
        
    template : numpy.ndarray
        template cross PS bandpower in shape (# mode, # ref)
        
    refs : list,tuple
        frequency reference for template
    
    """
    def __init__(self, signal, noise, freqs, modes, template=None, refs=[30., 353.], sampling_opt=dict()):
        """
        
        """
        log.debug('@ tpfit::__init__')
        # basic settings
        self.freqs = freqs
        self.modes = modes
        # measurements
        self.signal = signal
        self.noise = noise
        # reference frequencies
        self.refs = refs
        # adding template maps (for estimating template PS band power)
        self.template = template
        #
        self._foreground = None
        self._background = None
        # sampling optinos
        self.sampling_opt = sampling_opt
        # init active parameter list
        self.active_param_list = list()
        # init active parameter range
        self.active_param_range = dict()
        
    @property
    def freqs(self):
        return self._freqs
        
    @property
    def modes(self):
        return self._modes
        
    @property
    def signal(self):
        return self._signal
        
    @property
    def noise(self):
        return self._noise
        
    @property
    def refs(self):
        return self._refs
        
    @property
    def template(self):
        return self._template
        
    @property
    def sampling_opt(self):
        return self._sampling_opt
        
    @property
    def active_param_list(self):
        """active parameter name list"""
        return self._active_param_list
        
    @property
    def active_param_range(self):
        """active parameter range"""
        return self._active_param_range
        
    @freqs.setter
    def freqs(self, freqs):
        assert isinstance(freqs, (list,tuple))
        self._freqs = freqs
        self._nfreq = len(freqs)
        log.debug('number of frequencies %s' % str(self._nfreq))
        
    @modes.setter
    def modes(self, modes):
        assert isinstance(modes, (list,tuple))
        self._modes = modes
        self._nmode = len(modes)
        log.debug('angular modes %s' % str(self._modes))
        
    @refs.setter
    def refs(self, refs):
        assert isinstance(refs, (list,tuple))
        self._refs = refs
        self._nref = len(refs)
        log.debug('reference frequencies %s' % str(self._refs))
        
    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        assert (signal.shape[0] == self._nmode)  # number of angular modes
        assert (signal.shape[1] == self._nfreq) # number of frequency bands
        assert (signal.shape[1] == signal.shape[2])
        self._signal = signal
        log.debug('signal cross-PS read')
        
    @noise.setter
    def noise(self, noise):
        assert isinstance(noise, np.ndarray)
        assert (noise.shape[0] == self._nmode)  # number of angular modes
        assert (noise.shape[1] == self._nfreq) # number of frequency bands
        assert (noise.shape[1] == noise.shape[2])
        self._noise = noise
        log.debug('cross-PS std read')
        
    @template.setter
    def template(self, template):
        assert isinstance(template, np.ndarray)
        assert (template.shape[0] == self._nmode)  # number of angular modes
        assert (template.shape[1] == self._nref) # number of frequency bands
        self._template_flag = (template is not None)
        self._template = template
        log.debug('template cross-PS read')
        
    @sampling_opt.setter
    def sampling_opt(self, opt):
        assert isinstance(opt, dict)
        self._sampling_opt = opt
        
    @active_param_list.setter
    def active_param_list(self, active_param_list):
        assert isinstance(active_param_list, (list,tuple))
        self._active_param_list = active_param_list
        
    @active_param_range.setter
    def active_param_range(self, active_param_range):
        assert isinstance(active_param_range, dict)
        self._active_param_range = active_param_range
        
    def __call__(self, kwargs=dict()):
        log.debug('@ tpfit::__call__')
        return self.run(kwargs)
        
    @property
    def foreground(self):
        return self._foreground
        
    @foreground.setter
    def foreground(self, foreground_model):
        assert isinstance(foreground_model, fgmodel)
        self._foreground = foreground_model
        
    @property
    def background(self):
        return self._background
        
    @background.setter
    def background(self, background_model):
        assert isinstance(background_model, bgmodel)
        self._background = background_model
        
    def run(self, kwargs=dict()):
        # setup models and active param list
        self._foreground = syncdustmodel(self._modes, refs=self._refs)
        self._background = cmbmodel(self._modes)
        if self._template_flag:
            for i in range(self._nmode):
                for j in range(len(self._refs)):
                    self._foreground.reset({self._foreground.param_list[i+j*self._nmode] : self._template[i,j]})
            self._active_param_list = ['beta_s','beta_d','rho']+self._background.param_list
        else:
            self._active_param_list = self._fogreground.param_list + self._background.param_list
        # priors and ranges
        for name in self._active_param_list:
            if name in self._foreground.param_list:
                self._active_param_range[name] = self._foreground.param_range[name]
            elif name in self._background.param_list:
                self._active_param_range[name] = self._background.param_range[name]
        #
        self.info
        # start Dyensty
        results = pymultinest.solve(LogLikelihood=self._core_likelihood,
                                    Prior=self.prior,
                                    n_dims=len(self.active_param_list),
                                    **self.sampling_opt)
        return results
        
        
    def _core_likelihood(self, cube):
        """
        core log-likelihood calculator

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
        variable = cube[:]
        # map variable to parameters
        parameter = dict()
        for i in range(len(self._active_param_list)):
            _name = self._active_param_list[i]
            parameter[_name] = self.unity_mapper(variable[i], self._active_param_range[_name])
        #print ('variable', variable)
        #print ('name list', self._active_param_list)
        #print ('parameter', parameter)
        # predict signal
        for name in self._active_param_list:
            self._foreground.reset({name : parameter[name]})
            self._background.reset({name : parameter[name]})
        #print ('foreground reset', self._foreground.params)
        #print ('background reset', self._background.params)
        #import os
        #os.exit(1)
        bp = self._foreground.bandpower(self._freqs) + self._background.bandpower(self._freqs)
        #
        return self.loglikeli(bp)
        
    def loglikeli(self, predicted):
        """log-likelihood calculator"""
        assert (predicted.shape == self._signal.shape)
        return np.sum(-0.5*np.log((self._signal.reshape(1,-1) - predicted.reshape(1,-1))**2/self._noise.reshape(1,-1)**2))
        
    def prior(self, cube):
        """flat prior"""
        return cube
        
    def unity_mapper(self, x, range=[0.,1.]):
        """
        Maps x from [0, 1] into the interval [a, b]

        Parameters
        ----------
        x : float
            the variable to be mapped
        range : list,tuple
            the lower and upper parameter value limits

        Returns
        -------
        numpy.float64
            The mapped parameter value
        """
        log.debug('@ tpfit::unity_mapper')
        assert isinstance(range, (list,tuple))
        assert (len(range) == 2)
        return np.float64(x) * (np.float64(range[1])-np.float64(range[0])) + np.float64(range[0])
    
    @property
    def info(self):
        print ('sampling check list')
        print ('measurement frequency bands')
        print (self._freqs)
        print ('# of frequency bands')
        print (self._nfreq)
        print ('with template?')
        print (self._template_flag)
        print ('template reference frequency bands')
        print (self._refs)
        print ('angular modes')
        print (self._modes)
        print ('# of modes')
        print (self._nmode)
        print ('active parameter list')
        print (self._active_param_list)
        print ('active parameter range')
        print (self._active_param_range)
