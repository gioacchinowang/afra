import numpy as np
import healpy as hp
from afra.tools.fg_models import * 
from afra.tools.bg_models import * 
from afra.tools.ps_estimator import pstimator
from afra.tools.aux import vec_gauss, oas_cov, unity_mapper
from afra.tools.icy_decorator import icy
from afra.methods.tpfit import tpfit_gauss, tpfit_hl


@icy
class tpfpipe(object):
    """
    The template fitting pipeline class.
    
    without template (template=None):
        D_cmb,l at effective ells
        D_sync,l at effective ells
        D_dust,l at effective ells
        foreground frequency scaling parameters
        foregorund cross-corr parameters
    
    with low+high frequency templates:
        D_cmb,l at effective ells
        foreground frequency scaling parameters
        foregorund cross-corr parameters
    """
    def __init__(self, signals, noises, fiducials, mask=None, fwhms=None, templates=None, template_noises=None, template_fwhms=None, likelihood='gauss', targets='T', foreground=None, background=None, filt=None):
        """
        Parameters
        ----------
        
        signals : dict
            Measured signal maps,
            should be arranged in type {frequency (GHz): (map #, pixel #)}.
        
        noises : dict
            Simulated noise map samples,
            should be arranged in type: {frequency (GHz): (sample #, map #, pixel #)}.

        fiducials : numpy.array
            Simulated fiducial map samples,
            should be arranged in type {frequency: (sample #, map #, pixel #)}.
        
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (1, pixel #).
        
        fwhms : dict
            FWHM of gaussian beams for each frequency.
        
        templates : dict
            Template map dict,
            should be arranged in form: {frequency: map #, pixel #}.

        template_noises : dict
            Template noise map dict,
            should be arranged in form: {frequency: map #, pixel #}.
        
        template_fwhms : dict
            Template map fwhm dict,
            should be arranged in form: {frequency: fwhm}.
        
        targets : str
            Choosing among 'T', 'E' and 'B', 'EB', 'TEB'.
        
        likelihood : str
            likelihood type, can be either 'gauss' or 'hl'.

        filt : dict()
            filtering effect for CMB
        """
        # measurements
        self.signals = signals
        self.noises = noises
        self.fiducials = fiducials
        # adding template maps (for estimating template PS band power)
        self.templates = templates
        self.template_noises = template_noises
        self.template_fwhms = template_fwhms
        self._template_ps = None
        self.mask = mask
        self.fwhms = fwhms
        self.targets = targets
        # choose likelihood method
        self.likelihood = likelihood
        # init parameter list
        self.paramlist = list()
        self.paramrange = dict()
        self.debug = False
        # choose fore-/back-ground models
        self.foreground = foreground
        self.background = background
        # analyse select dict
        self._anadict = {'gauss': self.analyse_gauss,
                        'hl': self.analyse_hl}
        # ps estimator, to be assigned
        self._estimator = None
        # fiducial PS, to be assigned
        self._fiducial = None
        # Bayesian engine, to be assigned
        self._engine = None
        # filtering matrix dict
        self.filt = filt

    @property
    def freqlist(self):
        return self._freqlist

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def template_freqlist(self):
        return self._template_freqlist

    @property
    def template_nfreq(self):
        return self._template_nfreq

    @property
    def signals(self):
        return self._signals

    @property
    def noises(self):
        return self._noises

    @property
    def fiducials(self):
        return self._fiducials

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def mask(self):
        return self._mask

    @property
    def fwhms(self):
        return self._fwhms

    @property
    def templates(self):
        return self._templates

    @property
    def template_noises(self):
        return self._template_noises

    @property
    def template_fwhms(self):
        return self._template_fwhms

    @property
    def template_nsamp(self):
        return self._template_nsamp

    @property
    def template_ps(self):
        return self._template_ps

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def targets(self):
        return self._targets

    @property
    def ntarget(self):
        return self._ntarget

    @property
    def debug(self):
        return self._debug

    @property
    def paramrange(self):
        return self._paramrange

    @property
    def paramlist(self):
        return self._paramlist

    @property
    def foreground(self):
        return self._foreground

    @property
    def background(self):
        return self._background

    @property
    def filt(self):
        return self._filt

    @paramlist.setter
    def paramlist(self, paramlist):
        assert isinstance(paramlist, list)
        self._paramlist = paramlist

    @paramrange.setter
    def paramrange(self, paramrange):
        assert isinstance(paramrange, dict)
        self._paramrange = paramrange

    @signals.setter
    def signals(self, signals):
        assert isinstance(signals, dict)
        self._nfreq = len(signals)
        self._freqlist = sorted(signals.keys())
        assert (len(signals[next(iter(signals))].shape) == 2)
        assert (signals[next(iter(signals))].shape[0] == 3)
        self._npix = signals[next(iter(signals))].shape[1]
        self._nside = int(np.sqrt(self._npix//12))
        self._signals = signals.copy()

    @noises.setter
    def noises(self, noises):
        assert isinstance(noises, dict)
        assert (len(noises) == self._nfreq)
        assert (len(noises[next(iter(noises))].shape) == 3)
        self._nsamp = noises[next(iter(noises))].shape[0]
        assert (noises[next(iter(noises))].shape[1] == 3)
        assert (noises[next(iter(noises))].shape[2] == self._npix)
        self._noises = noises.copy()

    @fiducials.setter
    def fiducials(self, fiducials):
        assert isinstance(fiducials, dict)
        assert (len(fiducials) == self._nfreq)
        assert (fiducials[next(iter(fiducials))].shape == (self._nsamp,3,self._npix))
        self._fiducials = fiducials.copy()

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

    @templates.setter
    def templates(self, templates):
        if templates is not None:
            assert isinstance(templates, dict)
            self._template_freqlist = sorted(templates.keys())
            self._template_nfreq = len(templates)
            assert (self._template_nfreq < 3)
            assert (templates[next(iter(templates))].shape == (3,self._npix))
            self._template_flag = True
            self._templates = templates.copy()
        else:
            self._template_flag = False
            self._templates = None

    @template_noises.setter
    def template_noises(self, template_noises):
        if template_noises is not None:
            assert isinstance(template_noises, dict)
            assert (template_noises.keys() == self._templates.keys())
            assert (len(template_noises) == self._template_nfreq)
            self._template_noises = template_noises.copy()
            self._template_nsamp = len(template_noises[next(iter(template_noises))])
        else:
            assert (not self._template_flag)
            self._template_noises = None

    @template_fwhms.setter
    def template_fwhms(self, template_fwhms):
        if template_fwhms is not None:
            assert isinstance(template_fwhms, dict)
            assert (template_fwhms.keys() == self._templates.keys())
            self._template_fwhms = template_fwhms.copy()
        else:
            self._template_fwhms = dict()
            if self._template_flag:
                for name in self._template_freqlist:
                    self._template_fwhms[name] = None

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
        	self._signals[f][:,self._mask==0.] = 0.
        	if self._noises is not None:
                    self._noises[f][:,:,self._mask==0.] = 0.

    @targets.setter
    def targets(self, targets):
        assert isinstance(targets, str)
        self._targets = targets
        self._ntarget = len(targets)

    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug

    @likelihood.setter
    def likelihood(self, likelihood):
        assert isinstance(likelihood, str)
        self._likelihood = likelihood

    @background.setter
    def background(self, background):
        if background is None:
            self._background = None
        else:
            self._background = background

    @foreground.setter
    def foreground(self, foreground):
        if foreground is None:
            self._foreground = None
        else:
            self._foreground = foreground

    @filt.setter
    def filt(self, filt):
        if filt is not None:
            assert isinstance(filt, dict)
            assert (self._targets in filt)
        self._filt = filt

    def run(self,aposcale=6.,psbin=20,lmin=None,lmax=None,kwargs=dict()):
        """
        # preprocess
        # x_hat: mreasured PS bandpower, with measurement noise included naturally
        # x_fid: fiducial PS bandpower, without measurement noise
        # n_hat: mean of noise PS bandpower
        # x_mat: covariance of vectorized x_hat
        """
        if self._debug:
            print ('\n template fitting pipeline check list \n')
            print ('measurement frequency band')
            print (self._freqlist)
            print ('# of simulated samples')
            print (self._nsamp)
            print ('map HEALPix Nside')
            print (self._nside)
            print ('with template?')
            print (self._template_flag)
            if self._template_flag:
                print ('template reference frequency bands')
                print (self._template_freqlist)
                print ('# of template frequency bands')
                print (self._template_nfreq)
                print ('template beams')
                print (self._template_fwhms)
            print ('FWHMs')
            print (self._fwhms)
            print ('PS estimation apodization scale')
            print (aposcale)
            print ('PS estimation angular modes bin size')
            print (psbin)
            print ('PS minimal multipole')
            print (lmin)
            print ('PS maximal multipole')
            print (lmax)
            print ('foreground model')
            print (self._foreground)
            print ('background model')
            print (self._background)
            print ('\n')
        x_hat, x_fid, n_hat, x_mat = self.preprocess(aposcale,psbin,lmin,lmax)
        return self.analyse(x_hat,x_fid,n_hat,x_mat,kwargs)

    def preprocess(self,aposcale,psbin,lmin,lmax):
        # prepare ps estimator
        self._estimator = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmin=lmin,lmax=lmax,targets=self._targets,filt=self._filt)
        nmode = len(self._estimator.modes)
        if self._template_flag:
            self._template_ps = dict()
            for i in range(self._template_nfreq):
                signal_ps = np.zeros((self._ntarget,nmode),dtype=np.float32)
                noise_ps = np.zeros((self._template_nsamp,self._ntarget,nmode),dtype=np.float32)
                _fi = self._template_freqlist[i]
                stmp = self._estimator.autoBP(self._templates[_fi],fwhms=self._template_fwhms[_fi])
                wsp = self._estimator.autoWSP(self._templates[_fi],fwhms=self._template_fwhms[_fi])
                for t in range(self._ntarget):
                    signal_ps[t] = stmp[1+t]
                for s in range(self._template_nsamp):
                    # auto correlation
                    ntmp = self._estimator.autoBP(self._template_noises[_fi][s],wsp=wsp,fwhms=self._template_fwhms[_fi])
                    # assign results
                    for t in range(self._ntarget):
                        noise_ps[s,t] = ntmp[1+t]
                self._template_ps[_fi] = signal_ps - np.mean(noise_ps,axis=0)
        # prepare model, parameter list generated during init models
        if self._background is not None:
            self._background = self._background(self._freqlist,self._estimator)
        else:
            self._background = None
        if self._foreground is not None:
            self._foreground = self._foreground(self._freqlist,self._estimator,self._template_ps)
        else:
            self._foreground = None
        # estimate X_hat and M
        wsp_dict = dict()  # wsp pool
        signal_ps = np.zeros((self._nsamp+1,self._ntarget,nmode,self._nfreq,self._nfreq),dtype=np.float32)
        noise_ps = np.zeros((self._nsamp+1,self._ntarget,nmode,self._nfreq,self._nfreq),dtype=np.float32)
        # filling wsp pool and estimated measured bandpowers
        for i in range(self._nfreq):
            _fi = self._freqlist[i]
            wsp_dict[(i,i)] = self._estimator.autoWSP(self._signals[_fi],fwhms=self._fwhms[_fi])
            stmp = self._estimator.autoBP(self._signals[_fi],fwhms=self._fwhms[_fi])
            for t in range(self._ntarget):
                for k in range(nmode):
                    signal_ps[0,t,k,i,i] = stmp[1+t][k]
            for j in range(i+1,self._nfreq):
                _fj = self._freqlist[j]
                wsp_dict[(i,j)] = self._estimator.crosWSP(np.r_[self._signals[_fi],self._signals[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                stmp = self._estimator.crosBP(np.r_[self._signals[_fi],self._signals[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                for t in range(self._ntarget):
                    for k in range(nmode):
                        signal_ps[0,t,k,i,j] = stmp[1+t][k]
                        signal_ps[0,t,k,j,i] = stmp[1+t][k]
        # work out estimations
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                _fi = self._freqlist[i]
                # auto correlation
                ntmp = self._estimator.autoBP(self._noises[_fi][s],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[_fi])
                stmp = self._estimator.autoBP(self._fiducials[_fi][s]+self._noises[_fi][s],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[_fi])
                # assign results
                for t in range(self._ntarget):
                    for k in range(nmode):
                        noise_ps[s+1,t,k,i,i] = ntmp[1+t][k]
                        signal_ps[s+1,t,k,i,i] = stmp[1+t][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    _fj = self._freqlist[j]
                    # cross correlation
                    ntmp = self._estimator.crosBP(np.r_[self._noises[_fi][s],self._noises[_fj][s]],wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                    stmp = self._estimator.crosBP(np.r_[self._fiducials[_fi][s]+self._noises[_fi][s],self._fiducials[_fj][s]+self._noises[_fj][s]],wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                    for t in range(self._ntarget):
                        for k in range(nmode):
                            noise_ps[s+1,t,k,i,j] = ntmp[1+t][k]
                            noise_ps[s+1,t,k,j,i] = ntmp[1+t][k]
                            signal_ps[s+1,t,k,i,j] = stmp[1+t][k]
                            signal_ps[s+1,t,k,j,i] = stmp[1+t][k]
        if self._debug:
            return ( signal_ps, noise_ps )
        return ( signal_ps[0], np.mean(signal_ps[1:],axis=0), np.mean(noise_ps[1:],axis=0) ,oas_cov(vec_gauss(signal_ps[1:])) )

    def analyse(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        return self._anadict[self._likelihood](x_hat,x_fid,n_hat,x_mat,kwargs)

    def analyse_gauss(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        # gauss likelihood simplifies the usage of noise and fiducial model
        self._engine = tpfit_gauss(x_hat,x_fid,n_hat,x_mat,self._background,self._foreground)
        if (len(self._paramrange)):
            self._engine.rerange(self._paramrange)
        result = self._engine.run(kwargs)
        self._paramlist = sorted(self._engine.activelist)
        return result

    def analyse_hl(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        self._engine = tpfit_hl(x_hat,x_fid,n_hat,x_mat,self._background,self._foreground)
        if (len(self._paramrange)):
            self._engine.rerange(self._paramrange)
        result = self._engine.run(kwargs)
        self._paramlist = sorted(self._engine.activelist)
        return result

    def reprocess(self,signals):
        """
        use new signal dict and pre-calculated noise level
        to produce new dl_hat
        """
        # read new signals dict
        assert isinstance(signals, dict)
        assert (self._freqlist == list(signals.keys()))
        assert (signals[next(iter(signals))].shape == (3,self._npix))
        nmode = len(self._estimator.modes)  # angular modes
        signal_ps = np.zeros((self._ntarget,nmode,self._nfreq,self._nfreq),dtype=np.float32)
        for i in range(self._nfreq):
            _fi = self._freqlist[i]
            stmp = self._estimator.autoBP(signals[_fi],fwhms=self._fwhms[_fi])
            for t in range(self._ntarget):
                for k in range(nmode):
                    signal_ps[t,k,i,i] = stmp[1+t][k]
            for j in range(i+1,self._nfreq):
                _fj = self._freqlist[j]
                stmp = self._estimator.crosBP(np.r_[signals[_fi],signals[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                for t in range(self._ntarget):
                    for k in range(nmode):
                        signal_ps[t,k,i,j] = stmp[1+t][k]
                        signal_ps[t,k,j,i] = stmp[1+t][k]
        return signal_ps
