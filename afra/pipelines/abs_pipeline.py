import numpy as np
import healpy as hp
from afra.methods.abs import abssep
from afra.tools.bg_models import *
from afra.tools.ps_estimator import pstimator
from afra.tools.aux import vec_gauss, oas_cov, unity_mapper
from afra.methods.tpfit import tpfit_gauss, tpfit_hl
from afra.tools.icy_decorator import icy

@icy
class abspipe(object):
    """The ABS pipeline for extracting CMB power-spectrum band power,
        according to given measured sky maps at various frequency bands.
    """

    def __init__(self, data, noises=None, fiducials=None, mask=None, fwhms=None, fiducial_fwhms=None, targets='T', filt=None, background=None, likelihood='gauss'):
        """
        ABS pipeline init.
        
        Parameters
        ----------
        
        data : dict
            Measured data maps,
            should be arranged in form: {frequency (GHz) : array(map #, pixel #)}.
        
        noises : dict
            Simulated noise map samples,
            should be arranged in form: {frequency (GHz): (sample #, map #, pixel #)}.
        
        fiducials: dict
            Fiducial CMB maps (prepared),
            should be arranged in form: {frequency (GHz): (sample #, map #, pixel #)}.
        
        mask : numpy.ndarray
            Universal mask map, if None, assume full-sky coverage, otherwise,
            should be arranged in shape: (pixel #,).
        
        fwhms : dict
            FWHM (in rad) of (measurements') gaussian beams for each frequency,
            if None, assume no observational beam.
        
        fiducial_fwhms: dict
            Fiducial CMB fwhms.
        
        targets : str
            Chosen among 'T', 'E', 'B', 'EB', 'TEB',
            to instruct analyzing mode combination.
        
        filt : dict
            Filtering-correction matrix for CMB (from filted to original),
            entry name should at least contain "targets".
        
        background: str
            CMB model type, chosen among "ncmb" and "acmb",
            can be None if post-ABS analysis is not required.
        """
        self.data = data
        self.noises = noises
        self.mask = mask
        self.fwhms = fwhms
        self.targets = targets
        # analyse select dict with keys defined by self._noise_flag
        self._anadict = {(True): self.analyse_noisy,
                           (False): self.analyse_quiet}
        # debug mode
        self.debug = False
        # ps estimator
        self._estimator = None
        # filtering matrix dict
        self.filt = filt
        # background fiducial
        self.fiducials = fiducials
        self.fiducial_fwhms = fiducial_fwhms
        # background model
        self.background = background
        self._background_obj = None
        # Bayesian engine, to be assigned
        self._engine = None
        self.likelihood = likelihood
        # init parameter list
        self.paramlist = list()
        self.paramrange = dict()

    @property
    def data(self):
        return self._data

    @property
    def noises(self):
        return self._noises

    @property
    def mask(self):
        return self._mask
 
    @property
    def freqlist(self):
        return self._freqlist

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def nside(self):
        return self._nside

    @property
    def targets(self):
        return self._targets

    @property
    def ntarget(self):
        return self._ntarget

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def debug(self):
        return self._debug

    @property
    def fwhms(self):
        return self._fwhms

    @property
    def filt(self):
        return self._filt

    @property
    def fiducials(self):
        return self._fiducials

    @property
    def fiducial_fwhms(self):
        return self._fiducial_fwhms

    @property
    def background(self):
        return self._background

    @property
    def engine(self):
        return self._engine

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def paramlist(self):
        return self._paramlist

    @property
    def paramrange(self):
        return self._paramrange

    @data.setter
    def data(self, data):
        """catch and register nfreq, nmap, npix and nside automatically.
        """
        assert isinstance(data, dict)
        self._nfreq = len(data)
        self._freqlist = sorted(data.keys())
        assert (len(data[next(iter(data))].shape) == 2)
        assert (data[next(iter(data))].shape[0] == 3)
        self._npix = data[next(iter(data))].shape[1]
        self._nside = int(np.sqrt(self._npix//12))
        self._data = data.copy()

    @noises.setter
    def noises(self, noises):
        if noises is not None:
            assert isinstance(noises, dict)
            assert (len(noises) == self._nfreq)
            assert (len(noises[next(iter(noises))].shape) == 3)
            self._nsamp = noises[next(iter(noises))].shape[0]
            assert (noises[next(iter(noises))].shape[1] == 3)
            assert (noises[next(iter(noises))].shape[2] == self._npix)
            self._noises = noises.copy()
            self._noise_flag = True
        else:
            self._noises = None
            self._noise_flag = False

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones(self._npix,dtype=np.float32)
        else:
            assert isinstance(mask, np.ndarray)
            assert (len(mask) == self._npix)
            self._mask = mask.copy()
        # clean up input maps with mask
        for f in self._freqlist:
            self._data[f][:,self._mask==0.] = 0.
            if self._noises is not None:
                    self._noises[f][:,:,self._mask==0.] = 0.

    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug

    @fwhms.setter
    def fwhms(self, fwhms):
        if fwhms is not None:
            assert isinstance(fwhms, dict)
            assert (len(fwhms) == self._nfreq)
            self._fwhms = fwhms.copy()
        else:
            self._fwhms = dict()
            for f in self._freqlist:
                self._fwhms[f] = None

    @targets.setter
    def targets(self, targets):
        assert isinstance(targets, str)
        self._targets = targets
        self._ntarget = len(targets)

    @filt.setter
    def filt(self, filt):
        if filt is not None:
            assert isinstance(filt, dict)
            assert (self._targets in filt)
        self._filt = filt

    @fiducials.setter
    def fiducials(self, fiducials):
        if self._noise_flag:
            assert isinstance(fiducials, dict)
            assert (len(fiducials) == 1)
            assert (fiducials[next(iter(fiducials))].shape == (self._nsamp,3,self._npix))
            self._fiducials = fiducials.copy()
        else:
            self._fiducials = None

    @fiducial_fwhms.setter
    def fiducial_fwhms(self, fiducial_fwhms):
        if fiducial_fwhms is not None:
            assert isinstance(fiducial_fwhms, dict)
            assert (fiducial_fwhms.keys() == self._fiducials.keys())
            self._fiducial_fwhms = fiducial_fwhms.copy()
        elif self._noise_flag:
            self._fiducial_fwhms = dict()
            for name in list(self._fiducials.keys()):
                self._fiducial_fwhms[name] = None
        else:
            self._fiducial_fwhms = None

    @likelihood.setter
    def likelihood(self, likelihood):
        assert isinstance(likelihood, str)
        self._likelihood = likelihood

    @background.setter
    def background(self, background):
        if background is None:
            self._background = None
        else:
            assert isinstance(background, str)
            if (background == 'ncmb'):
                self._background = ncmbmodel
            elif (background == 'acmb'):
                self._background = acmbmodel

    @paramlist.setter
    def paramlist(self, paramlist):
        assert isinstance(paramlist, list)
        self._paramlist = paramlist

    @paramrange.setter
    def paramrange(self, paramrange):
        assert isinstance(paramrange, dict)
        self._paramrange = paramrange

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
        bp_s, bp_f, bp_n = self.preprocess(aposcale, psbin, lmin, lmax)
        # method selection
        bp_cmb = self.analyse(bp_s, bp_n, shift, threshold)[1]
        # post analysis
        return self.postprocess(bp_cmb, bp_f, kwargs)

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
        bp_s, bp_f, bp_n = self.preprocess(aposcale, psbin, lmin, lmax)
        # method selection
        return self.analyse(bp_s, bp_n, shift, threshold)

    def preprocess(self, aposcale, psbin, lmin=None, lmax=None):
        """
        ABS preprocess routine, converts maps into band-powers.

        Parameters
        ----------

        aposcale : float
            Apodization scale.

        psbin : integer
            Number of angular modes in each bin,
            for conducting pseudo-PS estimation.

        lmin/lmax : integer
            Lower/Upper multipole limit.

        Returns
        -------
            (data bp, fiducial bp, noise bp)
        """
        assert isinstance(aposcale, float)
        assert isinstance(psbin, int)
        assert (psbin > 0)
        assert (aposcale > 0)
        # init PS estimator
        self._estimator = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmin=lmin,lmax=lmax,targets=self._targets,filt=self._filt)
        # run trial PS estimations for workspace template
        if not self._noise_flag:
            # prepare total data PS in the shape required by ABS method
            data_bp = np.zeros((self._ntarget,self._estimator.nmode,self._nfreq,self._nfreq),dtype=np.float32)
            for i in range(self._nfreq):
                _fi = self._freqlist[i]
                # auto correlation
                stmp = self._estimator.autoBP(self._data[_fi],fwhms=self._fwhms[_fi])
                # assign results
                for t in range(self._ntarget):
                    for k in range(self._estimator.nmode):
                        data_bp[t,k,i,i] = stmp[1+t][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    _fj = self._freqlist[j]
                    stmp = self._estimator.crosBP(np.r_[self._data[_fi],self._data[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                    for t in range(self._ntarget):
                        for k in range(self._estimator.nmode):
                            data_bp[t,k,i,j] = stmp[1+t][k]
                            data_bp[t,k,j,i] = stmp[1+t][k]
            return (data_bp, None, None)
        else:
            # run trial PS estimations for workspace template
            wsp_dict = dict()
            for i in range(self._nfreq):
                _fi = self._freqlist[i]
                wsp_dict[(i,i)] = self._estimator.autoWSP(self._data[_fi],fwhms=self._fwhms[_fi])
                for j in range(i+1,self._nfreq):
                    _fj = self._freqlist[j]
                    wsp_dict[(i,j)] = self._estimator.crosWSP(np.r_[self._data[_fi],self._data[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
            # allocate
            noise_bp = np.zeros((self._nsamp,self._ntarget,self._estimator.nmode,self._nfreq,self._nfreq),dtype=np.float32)
            data_bp = np.zeros((self._nsamp,self._ntarget,self._estimator.nmode,self._nfreq,self._nfreq),dtype=np.float32)
            fiducial_bp = np.zeros((self._nsamp,self._ntarget,self._estimator.nmode),dtype=np.float32)
            for s in range(self._nsamp):
                # fiducial
                _ff = list(self._fiducials.keys())[0]
                ftmp = self._estimator.autoBP(self._fiducials[_ff][s],fwhms=self._fiducial_fwhms[_ff])
                for t in range(self._ntarget):
                    fiducial_bp[s,t] = ftmp[1+t]
                # prepare noise samples on-fly
                for i in range(self._nfreq):
                    _fi = self._freqlist[i]
                    # auto correlation
                    ntmp = self._estimator.autoBP(self._noises[_fi][s],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[_fi])
                    stmp = self._estimator.autoBP(self._data[_fi]+self._noises[_fi][s],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[_fi])
                    # assign results
                    for t in range(self._ntarget):
                        for k in range(self._estimator.nmode):
                            noise_bp[s,t,k,i,i] = ntmp[1+t][k]
                            data_bp[s,t,k,i,i] = stmp[1+t][k]
                    # cross correlation
                    for j in range(i+1,self._nfreq):
                        _fj = self._freqlist[j]
                        # cross correlation
                        ntmp = self._estimator.crosBP(np.r_[self._noises[_fi][s],self._noises[_fj][s]],wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                        stmp = self._estimator.crosBP(np.r_[self._data[_fi]+self._noises[_fi][s],self._data[_fj]+self._noises[_fj][s]],wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                        for t in range(self._ntarget):
                            for k in range(self._estimator.nmode):
                                noise_bp[s,t,k,i,j] = ntmp[1+t][k]
                                noise_bp[s,t,k,j,i] = ntmp[1+t][k]
                                data_bp[s,t,k,i,j] = stmp[1+t][k]
                                data_bp[s,t,k,j,i] = stmp[1+t][k]
            return (data_bp, fiducial_bp, noise_bp)

    def analyse(self, bp_data, bp_noise, shift, threshold):
        return self._anadict[self._noise_flag](bp_data, bp_noise, shift, threshold)

    def analyse_quiet(self, data, noise=None, shift=None, threshold=None):
        """
        Noise-free CMB (band power) extraction.
        
        Returns
        -------
            (angular modes, requested PS, eigen info)
        """
        # send PS to ABS method, noiseless case requires no shift nor threshold
        rslt = np.empty((self._ntarget,self._estimator.nmode),dtype=np.float32)
        info = dict()
        for t in range(self._ntarget):
            spt = abssep(data[t],shift=None,threshold=None)
            rslt[t] = spt.run()
            if self._debug:
                info[self._targets[t]] = spt.run_info()
        return (self._estimator.modes, rslt, info)

    def analyse_noisy(self, data, noise, shift, threshold):
        # get noise PS mean and rms
        noise_mean = np.mean(noise,axis=0)
        noise_std = np.std(noise,axis=0)
        noise_std_diag = np.zeros((self._ntarget,self._estimator.nmode,self._nfreq),dtype=np.float32)
        for t in range(self._ntarget):
            for l in range(self._estimator.nmode):
                noise_std_diag[t,l] = np.diag(noise_std[t,l])
        # shift for each angular mode independently
        safe_shift = shift*np.mean(noise_std_diag,axis=2)  # safe_shift in shape (nmode,ntarget)
        rslt = np.empty((self._nsamp,self._ntarget,self._estimator.nmode),dtype=np.float32)
        info = dict()
        for s in range(self._nsamp):
            for t in range(self._ntarget):
                # send PS to ABS method
                spt = abssep(data[s,t]-noise_mean[t],noise_mean[t],noise_std_diag[t],shift=safe_shift[t],threshold=threshold)
                rslt[s,t] = spt.run()
                if self._debug:
                    info[self._targets[t]] = spt.run_info()
        return (self._estimator.modes, rslt, info)

    def postprocess(self, cmb_bp, fiducial_bp, kwargs=dict()):
        # prepare model, parameter list generated during init models
        if self._background is not None:
            self._background_obj = self._background(list(self._fiducials.keys()),self._estimator)
        # estimate M
        x_hat = cmb_bp.reshape(self._nsamp,self._ntarget,self._estimator.nmode,1,1)
        x_fid = fiducial_bp.reshape(self._nsamp,self._ntarget,self._estimator.nmode,1,1)
        n_hat = np.zeros((self._ntarget,self._estimator.nmode,1,1),dtype=np.float32)
        x_mat = oas_cov(vec_gauss(x_hat+x_fid))
        if (self._likelihood == 'gauss'):
            self._engine = tpfit_gauss(np.mean(x_hat,axis=0),np.mean(x_fid,axis=0),n_hat,x_mat,self._background_obj,None)
        elif (self._likelihood == 'hl'):
            o_hat = n_hat.copy()  # offset defined in lollipop likelihood (1503.01347)
            for l in range(o_hat.shape[1]):
                o_hat[:,l,:,:] *= np.sqrt(2.*self._estimator.modes[l]+1./2.)
            self._engine = tpfit_hl(np.mean(x_hat,axis=0),np.mean(x_fid,axis=0),n_hat,x_mat,self._background_obj,None,o_hat)
        if (len(self._paramrange)):
            self._engine.rerange(self._paramrange)
        rslt = self._engine.run(kwargs)
        self._paramlist = sorted(self._engine.activelist)
        return rslt
