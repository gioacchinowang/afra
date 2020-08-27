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
    def __init__(self, signals, variances, mask=None, fwhms=None, templates=None, template_fwhms=None, likelihood='gauss', target='T', foreground=None, background=None):
        """
        Parameters
        ----------
        
        signals : dict
            Measured signal maps,
            should be arranged in type {frequency: (map #, pixel #)}.
        
        variances : dict
            Measured noise variance maps,
            should be arranged in type: {frequency: (map #, pixel #)}.
            By default, no variance maps required.
        
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (1, pixel #).
        
        fwhms : dict
            FWHM of gaussian beams for each frequency.
        
        templates : dict
            Template map dict,
            should be arranged in form: {frequency: map #, pixel #}.
        
        template_fwhms : dict
            Template map fwhm dict,
            should be arranged in form: {frequency: fwhm}.
        
        target : str
            Choosing among 'T', 'E' and 'B'.
        
        likelihood : str
            likelihood type, can be either 'gauss' or 'hl'.
        """
        # measurements
        self.signals = signals
        self.variances = variances
        # adding template maps (for estimating template PS band power)
        self.templates = templates
        self.template_fwhms = template_fwhms
        self.mask = mask
        self.fwhms = fwhms
        # choose likelihood method
        self.likelihood = likelihood
        # init parameter list
        self.param_list = list()
        self.param_range = dict()
        self.target = target
        self.debug = False
        # choose fore-/back-ground models
        self.foreground = foreground
        self.background = background
        # analyse select dict
        self._anadict = {'gauss': self.analyse_gauss,
                        'hl': self.analyse_hl}
        # ps estimator, to be assigned
        self._est = None
        # fiducial PS, to be assigned
        self._fiducial = None
        # Bayesian engine, to be assigned
        self._engine = None

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
    def variances(self):
        return self._variances

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
    def template_fwhms(self):
        return self._template_fwhms

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def target(self):
        return self._target

    @property
    def debug(self):
        return self._debug

    @property
    def param_range(self):
        return self._param_range

    @property
    def nparam(self):
        return len(self._param_list)

    @property
    def foreground(self):
        return self._foreground

    @property
    def background(self):
        return self._background

    @param_range.setter
    def param_range(self, param_range):
        assert isinstance(param_range, dict)
        self._param_range = param_range

    @signals.setter
    def signals(self, signals):
        assert isinstance(signals, dict)
        self._nfreq = len(signals)
        self._freqlist = sorted(signals.keys())
        assert (len(signals[next(iter(signals))].shape) == 2)
        assert (signals[next(iter(signals))].shape[0] == 3)
        self._npix = signals[next(iter(signals))].shape[1]
        self._nside = int(np.sqrt(self._npix//12))
        self._signals = signals

    @variances.setter
    def variances(self, variances):
        if variances is not None:
            assert isinstance(variances, dict)
            assert (variances[next(iter(variances))].shape == (3,self._npix))
            self._noise_flag = True
            self._variances = variances
        else:
            self._noise_flag = False
            self._variances = None

    @fwhms.setter
    def fwhms(self, fwhms):
        if fwhms is not None:
            assert isinstance(fwhms, dict)
            assert (len(fwhms) == self._nfreq)
            self._fwhms = fwhms
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
            self._templates = templates
        else:
            self._template_flag = False
            self._templates = None

    @template_fwhms.setter
    def template_fwhms(self, template_fwhms):
        if template_fwhms is not None:
            assert isinstance(template_fwhms, dict)
            assert (template_fwhms.keys() == self._templates.keys())
            self._template_fwhms = template_fwhms
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
        	if self._variances is not None:
        	    self._variances[f][:,self._mask==0.] = 0.

    @target.setter
    def target(self, target):
        assert isinstance(target, str)
        self._target = target

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

    def run(self,aposcale=6.,psbin=20,lmin=None,lmax=None,nsamp=500,kwargs=dict()):
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
            print ('# of frequency bands')
            print (self._nfreq)
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
            print ('PS esitmation noise resampling size')
            print (nsamp)
            print ('foreground model')
            print (self._foreground)
            print ('background model')
            print (self._background)
            print ('\n')
        x_hat, x_fid, n_hat, x_mat = self.preprocess(aposcale,psbin,lmin,lmax,nsamp)
        return self.analyse(x_hat,x_fid,n_hat,x_mat,kwargs)

    def preprocess(self,aposcale,psbin,lmin,lmax,nsamp):
        # prepare model, parameter list generated during init models
        if self._foreground is not None:
            self._foreground = self._foreground(self._freqlist,self._target,self._mask,aposcale,psbin,lmin,lmax,self._templates,self._template_fwhms)
        else:
            self._foreground = None
        if self._background is not None:
            self._background = self._background(self._freqlist,self._target,self._mask,aposcale,psbin,lmin,lmax)
        else:
            self._background = None
        # prepare fiducial cambmodel
        fiducial_model = cambmodel(self._freqlist,self._target,self._mask,aposcale,psbin,lmin,lmax)
        self._fiducial = np.transpose(fiducial_model._templates['total'])
        # prepare ps estimator
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmin=lmin,lmax=lmax,target=self._target)
        # estimate X_hat and M
        modes = self._est.modes
        wsp_dict = dict()  # wsp pool
        noise_map = np.zeros((6,self._npix),dtype=np.float32)  # Ti, Tj
        signal_map = np.zeros((6,self._npix),dtype=np.float32)  # Ti, Tj
        signal_ps = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        noise_ps = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        # filling wsp pool and estimated measured bandpowers
        for i in range(self._nfreq):
            wsp_dict[(i,i)] = self._est.autoWSP(self._signals[self._freqlist[i]],fwhms=self._fwhms[self._freqlist[i]])
            tmp = self._est.autoBP(self._signals[self._freqlist[i]],fwhms=self._fwhms[self._freqlist[i]])
            for k in range(len(modes)):
                signal_ps[0,k,i,i] = tmp[1][k]
            for j in range(i+1,self._nfreq):
                wsp_dict[(i,j)] = self._est.crosWSP(np.r_[self._signals[self._freqlist[i]],self._signals[self._freqlist[j]]],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                tmp = self._est.crosBP(np.r_[self._signals[self._freqlist[i]],self._signals[self._freqlist[j]]],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                for k in range(len(modes)):
                    signal_ps[0,k,i,j] = tmp[1][k]
                    signal_ps[0,k,j,i] = tmp[1][k]
        # work out estimations
        for s in range(1,nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                noise_map[:3] = np.random.normal(size=(3,self._npix))*np.sqrt(self._variances[self._freqlist[i]])
                fiducial_map = hp.smoothing(hp.synfast(self._fiducial,nside=self._nside,new=True,verbose=False),fwhm=self._fwhms[self._freqlist[i]],verbose=0)
                # append noise to signal
                signal_map[:3] = fiducial_map + noise_map[:3]
                # auto correlation
                ntmp = self._est.autoBP(noise_map[:3],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[self._freqlist[i]])
                stmp = self._est.autoBP(signal_map[:3],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[self._freqlist[i]])
                # assign results
                for k in range(len(modes)):
                    noise_ps[s,k,i,i] = ntmp[1][k]
                    signal_ps[s,k,i,i] = stmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    noise_map[3:] = np.random.normal(size=(3,self._npix))*np.sqrt(self._variances[self._freqlist[j]])
                    fiducial_map = hp.smoothing(hp.synfast(self._fiducial,nside=self._nside,new=True,verbose=False),fwhm=self._fwhms[self._freqlist[j]],verbose=0)
                    signal_map[3:] = fiducial_map + noise_map[3:]
                    # cross correlation
                    ntmp = self._est.crosBP(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                    stmp = self._est.crosBP(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                    for k in range(len(modes)):
                        noise_ps[s,k,i,j] = ntmp[1][k]
                        noise_ps[s,k,j,i] = ntmp[1][k]
                        signal_ps[s,k,i,j] = stmp[1][k]
                        signal_ps[s,k,j,i] = stmp[1][k]
        if self._debug:
            return ( signal_ps, noise_ps )
        return ( signal_ps[0], np.mean(signal_ps[1:],axis=0), np.mean(noise_ps[1:],axis=0) ,oas_cov(vec_gauss(signal_ps[1:])) )

    def analyse(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        return self._anadict[self._likelihood](x_hat,x_fid,n_hat,x_mat,kwargs)

    def analyse_gauss(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        # gauss likelihood simplifies the usage of noise and fiducial model
        self._engine = tpfit_gauss(x_hat-n_hat,x_mat,self._background,self._foreground)
        if (len(self._param_range)):
            self._engine.rerange(self._param_range)
        result = self._engine.run(kwargs)
        # rescale variables to parameters
        names = list(self._engine.param_range.keys())
        for i in range(len(names)):
            low, high = self._engine.param_range[names[i]]
            for j in range(result.samples.shape[0]):
                result.samples[j, i] = unity_mapper(result.samples[j, i], [low, high])
        return result

    def analyse_hl(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        self._engine = tpfit_hl(x_hat,x_fid,n_hat,x_mat,self._background,self._foreground)
        if (len(self._param_range)):
            self._engine.rerange(self._param_range)
        result = self._engine.run(kwargs)
        # rescale variables to parameters
        names = list(self._engine.param_range.keys())
        for i in range(len(names)):
            low, high = self._engine.param_range[names[i]]
            for j in range(result.samples.shape[0]):
                result.samples[j, i] = unity_mapper(result.samples[j, i], [low, high])
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
        modes = self._est.modes  # angular modes
        signal_ps = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        for i in range(self._nfreq):
            tmp = self._est.autoBP(signals[self._freqlist[i]],fwhms=self._fwhms[self._freqlist[i]])
            for k in range(len(modes)):
                signal_ps[k,i,i] = tmp[1][k]
            for j in range(i+1,self._nfreq):
                tmp = self._est.crosBP(np.r_[signals[self._freqlist[i]],signals[self._freqlist[j]]],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                for k in range(len(modes)):
                    signal_ps[k,i,j] = tmp[1][k]
                    signal_ps[k,j,i] = tmp[1][k]
        return signal_ps
