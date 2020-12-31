import numpy as np
from afra.tools.aux import umap, gvec, hvec
from afra.models.fg_models import fgmodel
from afra.models.bg_models import bgmodel
from dynesty import NestedSampler
from afra.tools.icy_decorator import icy


class fit(object):

    def __init__(self, data, fiducial, noise, covariance, background=None, foreground=None):
        """
        Parameters
        ----------
        data : numpy.ndarray
            measurements' band-power matrix

        fiducial : numpy.ndarray
            CMB+noise fiducial band-power matrix

        noise : numpy.ndarray
            noise band-power matrix

        covariance : numpy.ndarray
            covariance matrix

        background : bgmodel object
            background model instance

        foreground : fgmodel object
            foreground model instance
        """
        self.data = data
        self.fiducial = fiducial
        self.noise = noise
        self.covariance = covariance
        self.params = dict()  # initialized before back/fore-ground
        self.paramrange = dict()
        self.activelist = set()  # active Bayeisan parameters
        self.background = background
        self.foreground = foreground
        if (self._foreground is None and self._background is None):
            raise ValueError('no activated model')

    @property
    def data(self):
        return self._data

    @property
    def fiducial(self):
        return self._fiducial

    @property
    def noise(self):
        return self._noise
        
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
        return self._params

    @property
    def paramrange(self):
        return self._paramrange

    @property
    def activelist(self):
        return self._activelist

    @data.setter
    def data(self, data):
        assert isinstance(data, np.ndarray)
        assert (len(data.shape) == 4)
        if (np.isnan(data).any()):
            raise ValueError('encounter nan')
        self._data = data.copy()

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
        
    @covariance.setter
    def covariance(self, covariance):
        assert isinstance(covariance, np.ndarray)
        assert (len(covariance.shape) == 2)
        assert (covariance.shape[0] == covariance.shape[1])
        if (np.isnan(covariance).any()):
            raise ValueError('encounter nan')
        assert (np.linalg.matrix_rank(covariance) == covariance.shape[0])
        self._covariance = covariance.copy()

    @params.setter
    def params(self, params):
        assert isinstance(params, dict)
        self._params = params

    @paramrange.setter
    def paramrange(self, paramrange):
        assert isinstance(paramrange, dict)
        self._paramrange = paramrange

    @activelist.setter
    def activelist(self, activelist):
        assert isinstance(activelist, set)
        self._activelist = activelist

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

    def rerange(self, pdict):
        assert isinstance(pdict, dict)
        for name in pdict:
            if (name in self._paramrange.keys()):
                assert isinstance(pdict[name], (list,tuple))
                assert (len(pdict[name]) == 2)
                self._paramrange.update({name: pdict[name]})

    def run(self, kwargs=dict()):
        self._activelist = set(self._params.keys())
        if self._background is not None:
            self._activelist -= set(self._background.blacklist)
        if self._foreground is not None:
            self._activelist -= set(self._foreground.blacklist)
        sampler = NestedSampler(self._core_likelihood,self.prior,len(self._activelist),**kwargs)
        sampler.run_nested()
        results = sampler.results
        names = sorted(self._activelist)
        for i in range(len(names)):
            low, high = self.paramrange[names[i]]
            for j in range(len(results.samples)):
                results.samples[j, i] = umap(results.samples[j, i], [low, high])
        return results

    def _core_likelihood(self, cube):
        if np.any(cube > 1.) or np.any(cube < 0.):
            return np.nan_to_num(-np.inf)
        name_list = sorted(self._activelist)
        for i in range(len(name_list)):
            name = name_list[i]
            tmp = umap(cube[i], self._paramrange[name])
            if self._foreground is not None:
                self._foreground.reset({name: tmp})
            if self._background is not None:
                self._background.reset({name: tmp})
        # predict data
        if self._foreground is None:
            return self.loglikeli(self._background.bandpower())
        elif self._background is None:
            return self.loglikeli(self._foreground.bandpower())
        else:
            return self.loglikeli(self._foreground.bandpower() + self._background.bandpower())

    def prior(self, cube):
        return cube  # flat prior


@icy
class gaussfit(fit):

    def __init__(self, data, fiducial, noise, covariance, background=None, foreground=None):
        super(gaussfit, self).__init__(data,fiducial,noise,covariance,background,foreground)

    def loglikeli(self, predicted):
        assert (predicted.shape == self._data.shape)
        diff = gvec(predicted+self._noise-self._data)
        if (np.isnan(diff).any()):
            raise ValueError('encounter nan')
        logl = -0.5*(np.vdot(diff,np.matmul(np.linalg.pinv(self._covariance),diff)))
        if np.isnan(logl):
            return np.nan_to_num(-np.inf)
        return logl


@icy
class hlfit(fit):

    def __init__(self, data, fiducial, noise, covariance, background=None, foreground=None, offset=None):
        super(hlfit, self).__init__(data,fiducial,noise,covariance,background,foreground)
        self.offset = offset

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        if offset is None:
            self._offset = np.zeros_like(self._noise,dtype=np.float32)
        else:
            assert (offset.shape == self._noise.shape)
            if (np.isnan(offset).any()):
                raise ValueError('encounter nan')
            self._offset = offset.copy()

    def loglikeli(self, predicted):
        assert (predicted.shape == self._data.shape)
        diff = hvec(predicted+self._noise+self._offset,self._data+self._offset,self._fiducial+self._noise+self._offset)
        if (np.isnan(diff).any()):
            raise ValueError('encounter nan')
        logl = -0.5*(np.vdot(diff,np.matmul(np.linalg.pinv(self._covariance),diff)))
        if np.isnan(logl):
            return np.nan_to_num(-np.inf)
        return logl
