import numpy as np
from afra.tools.icy_decorator import icy
from afra.tools.aux import unity_mapper, vec_gauss, vec_hl
from afra.tools.fg_models import fgmodel
from afra.tools.bg_models import bgmodel
import dynesty


@icy
class tpfit(object):

    def __init__(self, signal, fiducial, noise, covariance, background=None, foreground=None, offset=None):
        # measurements
        self.signal = signal
        self.fiducial = fiducial
        self.noise = noise
        self.covariance = covariance
        self.offset = offset
        # parameters
        self.params = dict()
        self.paramrange = dict()
        self._activelist = set()
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
    def noise(self):
        return self._noise
        
    @property
    def offset(self):
        return self._offset

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
        """parameter name list"""
        return self._params

    @property
    def paramrange(self):
        """parameter range"""
        return self._paramrange

    @property
    def activelist(self):
        return self._activelist

    @property
    def debug(self):
        return self._debug

    @signal.setter
    def signal(self, signal):
        assert isinstance(signal, np.ndarray)
        assert (len(signal.shape) == 4)
        if (np.isnan(signal).any()):
            raise ValueError('encounter nan')
        self._signal = signal.copy()  # vectorized signal matrix

    @fiducial.setter
    def fiducial(self, fiducial):
        assert isinstance(fiducial, np.ndarray)
        assert (len(fiducial.shape) == 4)
        if (np.isnan(fiducial).any()):
            raise ValueError('encounter nan')
        self._fiducial = fiducial.copy()

    @noise.setter
    def noise(self, noise):
        assert isinstance(noise, np.ndarray)
        assert (len(noise.shape) == 4)
        if (np.isnan(noise).any()):
            raise ValueError('encounter nan')
        self._noise = noise.copy()
        
    @offset.setter
    def offset(self, offset):
        if offset is None:
            self._offset = np.zeros_like(self._noise)
        else:
            assert (offset.shape == self._noise.shape)
            if (np.isnan(offset).any()):
                raise ValueError('encounter nan')
            self._offset = offset.copy()

    @covariance.setter
    def covariance(self, covariance):
        assert isinstance(covariance, np.ndarray)
        assert (len(covariance.shape) == 2)
        assert (covariance.shape[0] == covariance.shape[1])
        if (np.isnan(covariance).any()):
            raise ValueError('encounter nan')
        assert (np.linalg.matrix_rank(covariance) == covariance.shape[0])
        self._covariance = covariance.copy()  # vectorized cov matrix

    @params.setter
    def params(self, params):
        assert isinstance(params, dict)
        self._params = params

    @paramrange.setter
    def paramrange(self, paramrange):
        assert isinstance(paramrange, dict)
        self._paramrange = paramrange

    @foreground.setter
    def foreground(self, foreground):
        if foreground is None:
            self._foreground = None
        else:
            assert isinstance(foreground, fgmodel)
            self._foreground = foreground
            # update from model
            self._params.update(self._foreground.params)
            self._paramrange.update(self._foreground.paramrange)

    @background.setter
    def background(self, background):
        if background is None:
            self._background = None
        else:
            assert isinstance(background, bgmodel)
            self._background = background
            # update from model
            self._params.update(self._background.params)
            self._paramrange.update(self._background.paramrange)

    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug

    def rerange(self, pdict):
        assert isinstance(pdict, dict)
        for name in pdict:
            if (name in self._paramrange.keys()):
                assert isinstance(pdict[name], (list,tuple))
                assert (len(pdict[name]) == 2)
                self._paramrange.update({name: pdict[name]})

    def run(self, kwargs=dict()):
        if self._debug:
            print ('\n template fitting kernel check list \n')
            print ('# of parameters')
            print (len(self.params))
            print ('parameters')
            print (self.params.keys())
            print ('parameter range')
            print (self.paramrange)
            print ('\n')
        self._activelist = set(self._params.keys())
        if self._background is not None:
            self._activelist -= set(self._background.blacklist)
        if self._foreground is not None:
            self._activelist -= set(self._foreground.blacklist)
        sampler = dynesty.NestedSampler(self._core_likelihood,self.prior,len(self._activelist),**kwargs)
        sampler.run_nested()
        results = sampler.results
        names = sorted(self._activelist)
        for i in range(len(names)):
            low, high = self.paramrange[names[i]]
            for j in range(len(results.samples)):
                results.samples[j, i] = unity_mapper(results.samples[j, i], [low, high])
        return results

    def _core_likelihood(self, cube):
        """core log-likelihood calculator"""
        # security boundary check
        if np.any(cube > 1.) or np.any(cube < 0.):
            return np.nan_to_num(-np.inf)
        # map variable to model parameters
        name_list = sorted(self._activelist)  # variable matches by alphabet order
        for i in range(len(name_list)):
            name = name_list[i]
            tmp = unity_mapper(cube[i], self._paramrange[name])
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

    def prior(self, cube):
        """flat prior"""
        return cube


@icy
class tpfit_gauss(tpfit):

    def __init__(self, signal, fiducial, noise, covariance, background=None, foreground=None, offset=None):
        super(tpfit_gauss, self).__init__(signal,fiducial,noise,covariance,background,foreground,offset)

    def loglikeli(self, predicted):
        """log-likelihood function"""
        assert (predicted.shape == self._signal.shape)
        diff = vec_gauss(predicted + self._noise - self._signal)
        if (np.isnan(diff).any()):
            raise ValueError('encounter nan')
        #(sign, logdet) = np.linalg.slogdet(self._covariance*2.*np.pi)
        logl = -0.5*( np.vdot(diff, np.linalg.solve(self._covariance, diff.T)) )#+sign*logdet)
        if np.isnan(logl):
            return np.nan_to_num(-np.inf)
        return logl


@icy
class tpfit_hl(tpfit):

    def __init__(self, signal, fiducial, noise, covariance, background=None, foreground=None, offset=None):
       super(tpfit_hl, self).__init__(signal,fiducial,noise,covariance,background,foreground,offset)

    def loglikeli(self, predicted):
        """log-likelihood function"""
        assert (predicted.shape == self._signal.shape)
        diff = vec_hl(predicted+self._noise+self._offset,self._signal+self._offset,self._fiducial+self._offset)
        if (np.isnan(diff).any()):
            raise ValueError('encounter nan')
        #(sign, logdet) = np.linalg.slogdet(self._covariance*2.*np.pi)
        logl = -0.5*( np.vdot(diff, np.linalg.solve(self._covariance, diff.T)) )#+sign*logdet)
        if np.isnan(logl):
            return np.nan_to_num(-np.inf)
        return logl
