import numpy as np
from afra.pipelines.pipeline import pipe
from afra.methods.abs import *
from afra.methods.fit import * 
from afra.tools.aux import gvec, empcov, jkncov
from afra.tools.icy_decorator import icy

@icy
class abspipe(pipe):

    def __init__(self, data, noises=None, mask=None, beams=None, targets='T',
                 fiducials=None, fiducial_beams=None,
                 templates=None, template_noises=None, template_beams=None,
                 foreground=None,background=None,
                 likelihood='gauss', filt=None):
        super(abspipe, self).__init__(data,noises,mask,beams,targets,fiducials,fiducial_beams,templates,template_noises,template_beams,None,background,likelihood,filt)
        # analyse select dict
        self._anadict = {True:self.analyse_noisy, False:self.analyse_quiet}
        # Bayesian engine, to be assigned
        self.engine = None
        # ABS results
        self.absrslt = None
        self.absinfo = dict()

    @property
    def engine(self):
        return self._engine

    @property
    def absrslt(self):
        return self._absrslt

    @property
    def absinfo(self):
        return self._absinfo

    @engine.setter
    def engine(self, engine):
        if engine is not None:
            assert isinstance(engine, fit)
        self._engine = engine

    @absrslt.setter
    def absrslt(self, absrslt):
        if absrslt is not None:
            assert isinstance(absrslt, np.ndarray)
            if (absrslt.ndim == 2):
                assert (absrslt.shape == (self._ntarget,self._estimator.nmode))
            else:
                assert (absrslt.shape == (self._noise_nsamp,self._ntarget,self._estimator.nmode))
        self._absrslt = absrslt

    @absinfo.setter
    def absinfo(self, absinfo):
        if absinfo is not None:
            assert isinstance(absinfo, dict)
        self._absinfo = absinfo

    def run(self, aposcale, psbin, lmin=None, lmax=None, shift=None, threshold=None, kwargs=dict()):
        """
        ABS routine,
        1. run "preprocess", estimate band-powers from given maps.
        2. run "analyse", extract measurement-based CMB band-powers with ABS method.
        3. run "postprocess", fit CMB band-powers/parameters with cosmic-variance.
        
        Returns
        -------
            (Dynesty fitting result)
        """
        assert isinstance(aposcale, float)
        assert isinstance(psbin, int)
        assert (psbin > 0)
        assert (aposcale > 0)
        assert (self._background is not None)
        # preprocess
        self.preprocess(aposcale, psbin, lmin, lmax)
        # method selection
        self.analyse(shift, threshold)
        # post analysis
        return self.postprocess(kwargs)

    def run_absonly(self, aposcale, psbin, lmin=None, lmax=None, shift=None, threshold=None):
        """
        ABS analysis only, without parameteric CMB model fitting.
        
        Returns
        -------
            (angular modes, requested PS, eigen info)
        """
        assert isinstance(aposcale, float)
        assert isinstance(psbin, int)
        assert (psbin > 0)
        assert (aposcale > 0)
        # preprocess
        self.preprocess(aposcale, psbin, lmin, lmax)
        # method selection
        self.analyse(shift, threshold)
        return (self._estimator.modes, self._absrslt, self._absinfo)

    def analyse(self, shift, threshold):
        self._anadict[self._noise_flag](shift, threshold)

    def analyse_quiet(self, shift=None, threshold=None):
        """
        Noise-free CMB (band power) extraction.
        """
        # send PS to ABS method, noiseless case requires no shift nor threshold
        self.absrslt = np.zeros((self._ntarget,self._estimator.nmode),dtype=np.float64)
        for t in range(self._ntarget):
            spt = abssep(self._data_bp[t],shift=None,threshold=None)
            self._absrslt[t] = spt.run()
            if self._debug:
                self._absinfo[self._targets[t]] = spt.run_info()

    def analyse_noisy(self, shift, threshold):
        # get noise PS mean and rms
        noise_mean = np.mean(self._noise_bp,axis=0)
        noise_std = np.std(self._noise_bp,axis=0)
        noise_std_diag = np.zeros((self._ntarget,self._estimator.nmode,self._nfreq),dtype=np.float64)
        for t in range(self._ntarget):
            for l in range(self._estimator.nmode):
                noise_std_diag[t,l] = np.diag(noise_std[t,l])
        # assemble data+noise-noise_mean
        ndat = self._noise_bp.copy()
        for s in range(self._noise_nsamp):
            ndat[s] += self._data_bp - noise_mean
        # shift for each angular mode independently
        safe_shift = shift*np.mean(noise_std_diag,axis=2)  # safe_shift in shape (nmode,ntarget)
        self.absrslt = np.zeros((self._noise_nsamp,self._ntarget,self._estimator.nmode),dtype=np.float64)
        for s in range(self._noise_nsamp):
            for t in range(self._ntarget):
                # send PS to ABS method
                spt = abssep(ndat[s,t],noise_mean[t],noise_std_diag[t],shift=safe_shift[t],threshold=threshold)
                self.absrslt[s,t] = spt.run()
                if self._debug:
                    self.absinfo[self._targets[t]] = spt.run_info()

    def postprocess(self, kwargs=dict()):
        # force nfreq=1
        self._nfreq = 1
        self._background_obj._nfreq = 1
        # assemble fiducial+abs
        abs_bp = self._absrslt.reshape(self._noise_nsamp,self._ntarget,self._estimator.nmode,1,1)
        abs_fid = self._fiducial_bp[:,:,:,0,0].reshape(self._fiducial_nsamp,self._ntarget,self._estimator.nmode,1,1)
        # empirical cov
        xbp = gvec(abs_bp)
        xfid = gvec(abs_fid)
        self.covmat = 2.*jkncov(xbp) + jkncov(xfid)
        # null noise
        null_noise = np.zeros((self._ntarget,self._estimator.nmode,1,1),dtype=np.float64)
        if (self._likelihood == 'gauss'):
            self._engine = gaussfit(np.mean(abs_bp,axis=0),np.mean(abs_fid,axis=0),null_noise,self._covmat,self._background_obj)
        elif (self._likelihood == 'hl'):
            self._engine = hlfit(np.mean(abs_bp,axis=0),np.mean(abs_fid,axis=0),null_noise,self._covmat,self._background_obj)
        if (len(self._paramrange)):
            self._engine.rerange(self._paramrange)
        rslt = self._engine.run(kwargs)
        self._paramlist = sorted(self._engine.activelist)
        return rslt
