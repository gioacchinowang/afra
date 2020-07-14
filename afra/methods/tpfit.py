import logging as log
import numpy as np
from afra.tools.icy_decorator import icy
from afra.tools.aux import vec_simple, g_simple
from afra.tools.fg_models import fgmodel
from afra.tools.bg_models import bgmodel
import dynesty


@icy
class tpfit_simple(object):
    """
    The template fitting class,
    symbol X stands for vectorized cross PS band power,
    which is arranged in EE, BB ordering
    
    Parameters
    ----------
    
    signal : numpy.ndarray
        vectorized signal cross PS bandpower in shape (# X,)
        
    covariance : numpy.ndarray
        covariance matrix for vectorized signal in shape (# X, # X)
        
    template : dict
        vectorized template cross PS bandpower in type {freqency: X}
        
	model : dict
		foreground and background models in type {'model_name': model},
		already prepared in pipeline.
 	
    """
    def __init__(self, signal, covariance, background, foreground):
        log.debug('@ tpfit::__init__')
        # measurements
        self.signal = signal
        self.covariance = covariance
        # parameters
        self.params = dict()
        self.param_range = dict()
        # models
        self.background = background
        self.foreground = foreground
        
    @property
    def signal(self):
        return self._signal
        
    @property
    def covariance(self):
        return self._covariance

    @property
    def foreground(self):
        return self._foreground

    @property
    def background(self):
        return self._background
        
    @property
    def params(self):
        """active parameter name list"""
        return self._params
        
    @property
    def param_range(self):
        """active parameter range"""
        return self._param_range
        
    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        assert (len(signal.shape) == 1)
        self._signal = signal.copy()  # vectorized signal matrix
        log.debug('cross-PS signal read')
        
    @covariance.setter
    def covariance(self, covariance):
        assert isinstance(covariance, np.ndarray)
        assert (len(covariance.shape) == 2)
        assert (covariance.shape[0] == covariance.shape[1])
        assert (np.linalg.matrix_rank(covariance) == covariance.shape[0])
        self._covariance = covariance.copy()  # vectorized cov matrix
        log.debug('cross-PS cov read')
        
    @params.setter
    def params(self, params):
        assert isinstance(params, dict)
        self._params = params
        
    @param_range.setter
    def param_range(self, param_range):
        assert isinstance(param_range, dict)
        self._param_range = param_range
        
    @foreground.setter
    def foreground(self, foreground):
        assert isinstance(foreground, fgmodel)
        self._foreground = foreground
        # update from model
        self._params.update(self._foreground.params)
        self._param_range.update(self._foreground.param_range)
        
    @background.setter
    def background(self, background):
        assert isinstance(background, bgmodel)
        self._background = background
        # update from model
        self._params.update(self._background.params)
        self._param_range.update(self._background.param_range)

    def rerange(self, pdict):
        assert isinstance(pdict, dict)
        for name in pdict:
            if (name in self._param_range.keys()):
                assert isinstance(pdict[name], (list,tuple))
                assert (len(pdict[name]) == 2)
                self._param_range.update({name: pdict[name]})

    def __call__(self, kwargs=dict()):
        print ('\n template fitting kernel check list \n')
        print ('# of parameters')
        print (len(self.params))
        print ('parameters')
        print (self.params.keys())
        print ('parameter range')
        print (self.param_range)
        print ('\n')
        return self.run(kwargs)
        
    def run(self, kwargs=dict()):
        sampler = dynesty.NestedSampler(self._core_likelihood,self.prior,len(self._params),**kwargs)
        sampler.run_nested()
        return sampler.results 
        """
        results = pymultinest.solve(LogLikelihood=self._core_likelihood,
                                    Prior=self.prior,
                                    n_dims=len(self._params),
                                    **kwargs)
        """
        
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
        # map variable to model parameters
        name_list = list(self._params.keys())
        for i in range(len(name_list)):
            name = name_list[i]
            tmp = self.unity_mapper(variable[i], self._param_range[name])
            self._foreground.reset({name: tmp})
            self._background.reset({name: tmp})
        # predict signal
        log.debug('@ tpfit_pipeline::foreground reset', self._foreground.params)
        log.debug('@ tpfit_pipeline::background reset', self._background.params)
        return self.loglikeli(vec_simple(self._foreground.bandpower() + self._background.bandpower()))
        
    def loglikeli(self, predicted):
        """log-likelihood calculator

        Parameters
        ----------

        predicted : numpy.ndarray
            vectorized bandpower from models
        """
        assert (predicted.shape == self._signal.shape)
        diff = predicted - self._signal
        #(sign, logdet) = np.linalg.slogdet(cov*2.*np.pi)
        return -0.5*(np.vdot(diff, np.linalg.solve(self._covariance, diff.T))) #+sign*logdet)
        
    def prior(self, cube):
        """flat prior"""
        return cube
        
    def unity_mapper(self, x, r=[0.,1.]):
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
        assert isinstance(r, (list,tuple))
        assert (len(r) == 2)
        return x * (r[1]-r[0]) + r[0]
