import logging as log
import numpy as np
from afra.tools.icy_decorator import icy
from afra.tools.aux import unity_mapper, vec_simple, vec_hl
from afra.tools.fg_models import fgmodel
from afra.tools.bg_models import bgmodel
import dynesty


@icy
class tpfit_simple(object):
    """
    The template fitting class,
    symbol X stands for vectorized cross PS band power,
    which is arranged in EE, BB ordering
    ions of cosmic acceleration for both flat and nonflat PAge universes. In the nonflat case, we find a ≳3σ tension between the spatial curvatures derive
    Parameters
    ----------
    
    signal : numpy.ndarray
        vectorized signal cross PS bandpower in shape (# mode, # freq, # freq)
        
    covariance : numpy.ndarray
        covariance matrix for vectorized signal in shape (# X, # X)
        
    model : dict
        foreground and background models in type {'model_name': model},
        already prepared in pipeline.
    """
    def __init__(self, signal, covariance, background=None, foreground=None):
        log.debug('@ tpfit::__init__')
        # measurements
        self.signal = signal
        self.covariance = covariance
        # parameters
        self.params = dict()
        self.param_range = dict()
        # debug flag
        self.debug = False
        # models
        self.background = background
        self.foreground = foreground
        if (self._foreground is None and self._background is None):
            raise ValueError('no activated model')
        
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

    @property
    def debug(self):
        return self._debug
        
    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        assert (len(signal.shape) == 3)
        self._signal = signal.copy()  # signal matrix
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
        if foreground is None:
            self._foreground = None
        else:
            assert isinstance(foreground, fgmodel)
            self._foreground = foreground
            # update from model
            self._params.update(self._foreground.params)
            self._param_range.update(self._foreground.param_range)
        
    @background.setter
    def background(self, background):
        if background is None:
            self._background = None
        else:
            assert isinstance(background, bgmodel)
            self._background = background
            # update from model
            self._params.update(self._background.params)
            self._param_range.update(self._background.param_range)

    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug

    def rerange(self, pdict):
        assert isinstance(pdict, dict)
        for name in pdict:
            if (name in self._param_range.keys()):
                assert isinstance(pdict[name], (list,tuple))
                assert (len(pdict[name]) == 2)
                self._param_range.update({name: pdict[name]})

    def __call__(self, kwargs=dict()):
        if self._debug:
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
            tmp = unity_mapper(variable[i], self._param_range[name])
            if self._foreground is not None:
                self._foreground.reset({name: tmp})
            if self._background is not None:
                self._background.reset({name: tmp})
        # predict signal
        if self._foreground is None:
            return self.loglikeli(self._background.bandpower())
        elif self._background is None:
            return self.loglikeli(self._foreground.bandpower())
        else:
            return self.loglikeli(self._foreground.bandpower() + self._background.bandpower())
        
    def loglikeli(self, predicted):
        """log-likelihood calculator

        Parameters
        ----------

        predicted : numpy.ndarray
            vectorized bandpower from models
        """
        assert (predicted.shape == self._signal.shape)
        diff = vec_simple( predicted - self._signal )
        #(sign, logdet) = np.linalg.slogdet(cov*2.*np.pi)
        return -0.5*(np.vdot(diff, np.linalg.solve(self._covariance, diff.T))) #+sign*logdet)
        
    def prior(self, cube):
        """flat prior"""
        return cube


@icy
class tpfit_hl(object):
    """
    The template fitting class,
    symbol X stands for vectorized cross PS band power,
    which is arranged in EE, BB ordering
    ions of cosmic acceleration for both flat and nonflat PAge universes. In the nonflat case, we find a ≳3σ tension between the spatial curvatures derive
    Parameters
    ----------
    
    signal : numpy.ndarray
        vectorized signal cross PS bandpower in shape (# l, # freq, # freq)
        
    covariance : numpy.ndarray
        covariance matrix for vectorized signal in shape (# X, # X)
        
    model : dict
        foreground and background models in type {'model_name': model},
        already prepared in pipeline.
    """
    def __init__(self, signal, fiducial, covariance, background=None, foreground=None):
        log.debug('@ tpfit::__init__')
        # measurements
        self.signal = signal
        self.fiducial = fiducial
        self.covariance = covariance
        # parameters
        self.params = dict()
        self.param_range = dict()
        # debug flag
        self.debug = False
        # models
        self.background = background
        self.foreground = foreground
        if (self._foreground is None and self._background is None):
            raise ValueError('no activated model')
        
    @property
    def signal(self):
        return self._signal

    @property
    def fiducial(self):
        return self._fiducial
        
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

    @property
    def debug(self):
        return self._debug
        
    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        assert (len(signal.shape) == 3)
        self._signal = signal.copy()  # vectorized signal matrix
        log.debug('cross-PS signal read')

    @fiducial.setter
    def fiducial(self, fiducial):
        assert isinstance(fiducial, np.ndarray)
        assert (len(fiducial.shape) == 3)
        self._fiducial = fiducial.copy()
        log.debug('cross-PS fiducial read')
        
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
        if foreground is None:
            self._foreground = None
        else:
            assert isinstance(foreground, fgmodel)
            self._foreground = foreground
            # update from model
            self._params.update(self._foreground.params)
            self._param_range.update(self._foreground.param_range)
        
    @background.setter
    def background(self, background):
        if background is None:
            self._background = None
        else:
            assert isinstance(background, bgmodel)
            self._background = background
            # update from model
            self._params.update(self._background.params)
            self._param_range.update(self._background.param_range)

    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug

    def rerange(self, pdict):
        assert isinstance(pdict, dict)
        for name in pdict:
            if (name in self._param_range.keys()):
                assert isinstance(pdict[name], (list,tuple))
                assert (len(pdict[name]) == 2)
                self._param_range.update({name: pdict[name]})

    def __call__(self, kwargs=dict()):
        if self._debug:
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
            tmp = unity_mapper(variable[i], self._param_range[name])
            if self._foreground is not None:
                self._foreground.reset({name: tmp})
            if self._background is not None:
                self._background.reset({name: tmp})
        # predict signal
        if self._foreground is None:
            return self.loglikeli(self._background.bandpower())
        elif self._background is None:
            return self.loglikeli(self._foreground.bandpower())
        else:
            return self.loglikeli(self._foreground.bandpower() + self._background.bandpower())
        
    def loglikeli(self, predicted):
        """log-likelihood calculator

        Parameters
        ----------

        predicted : numpy.ndarray
            vectorized bandpower from models
        """
        assert (predicted.shape == self._signal.shape)
        diff = vec_hl(predicted,self._signal,self._fiducial)
        #(sign, logdet) = np.linalg.slogdet(cov*2.*np.pi)
        return -0.5*(np.vdot(diff, np.linalg.solve(self._covariance, diff.T)))  #+sign*logdet)
        
    def prior(self, cube):
        """flat prior"""
        return cube
