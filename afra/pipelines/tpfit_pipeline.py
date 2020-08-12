import logging as log
import numpy as np
import healpy as hp
from afra.tools.fg_models import * 
from afra.tools.bg_models import * 
from afra.tools.ps_estimator import pstimator
from afra.tools.aux import vec_simple, oas_cov, unity_mapper
from afra.tools.icy_decorator import icy
from afra.methods.tpfit import tpfit_simple, tpfit_hl


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
    def __init__(self, signals, variances, mask=None, fwhms=None, templates=None, template_fwhms=None, likelihood='simple', foreground=None, background=None):
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
            
        freqs : list, tuple
            List of frequencies of measurements
            
        fwhms : dict
            FWHM of gaussian beams for each frequency
            
        templates : dict
            Template map dict,
            should be arranged in form: {frequency: map #, pixel #}

        template_fwhms : dict
            Template map fwhm dict
            should be arranged in form: {frequency: fwhm}

        likelihood : string
            likelihood type, can be either 'simple' or 'hl'.
        """
        log.debug('@ tpfpipe::__init__')
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
        #
        self.debug = False
        # choose fore-/back-ground models
        self.foreground = foreground
        self.background = background
        # preprocess select dict with keys defined by (self._likelihood, self._nmap)
        self._preprodict = {1: self.preprocess_T,
                            2: self.preprocess_B}
        # reprocess select dict with keys defined by (self._likelihood, self._nmap)
        self._reprodict = {1: self.reprocess_T,
                            2: self.reprocess_B}
        # analyse select dict
        self._anadict = {'simple': self.analyse_simple,
                        'hl': self.analyse_hl}
        # ps estimator, to be assigned
        self._est = None
        # fiducial PS, to be assigned
        self._fiducial = None
       
    @property
    def nmap(self):
        return self._nmap

    @property
    def freqs(self):
        return self._freqs

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def template_freqs(self):
        return self._template_freqs

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
    def debug(self):
        return self._debug
        
    @property
    def param_range(self):
        return self._param_range
        
    @property
    def nparam(self):
        """number of parameters"""
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
        """detect and register nfreq, nmap, npix and nside automatically
        """
        assert isinstance(signals, dict)
        self._freqs = list(signals.keys())
        log.debug('frequencies %s' % str(self._freqs))
        self._nfreq = len(self._freqs)
        log.debug('number of frequencies: %s' % str(self._nfreq))
        assert (len(signals[next(iter(signals))].shape) == 2)
        self._nmap = signals[next(iter(signals))].shape[0]
        log.debug('number of maps: %s' % str(self._nmap))
        self._npix = signals[next(iter(signals))].shape[1]
        self._nside = int(np.sqrt(self._npix//12))
        log.debug('HEALPix Nside: %s' % str(self._nside))
        self._signals = np.r_[[signals[x] for x in sorted(signals.keys())]]
        log.debug('signal maps loaded')
        
    @variances.setter
    def variances(self, variances):
        if variances is not None:
            assert isinstance(variances, dict)
            assert (variances[next(iter(variances))].shape[0] == self._nmap)
            assert (variances[next(iter(variances))].shape[1] == self._npix)
            self._noise_flag = True
            self._variances = np.r_[[variances[x] for x in sorted(variances.keys())]]
        else:
            self._noise_flag = False
            self._variances = None
        log.debug('variance maps loaded')

    @fwhms.setter
    def fwhms(self, fwhms):
        """signal maps' fwhms"""
        if fwhms is not None:
            assert isinstance(fwhms, dict)
            assert (list(fwhms.keys()) == self._freqs)
            self._fwhms = [fwhms[x] for x in sorted(fwhms.keys())]
        else:
            self._fwhms = [None]*self._nfreq
        log.debug('fwhms loaded')
        
    @templates.setter
    def templates(self, templates):
        """template maps at 1 or 2 frequency bands"""
        if templates is not None:
            assert isinstance(templates, dict)
            self._template_freqs = sorted(templates.keys())
            self._template_nfreq = len(self._template_freqs)
            assert (self._template_nfreq < 3)
            assert (templates[next(iter(templates))].shape[0] == self._nmap)
            assert (templates[next(iter(templates))].shape[1] == self._npix)
            self._template_flag = True
            self._templates = templates
        else:
            self._template_flag = False
            self._templates = None
        log.debug('template maps loaded')

    @template_fwhms.setter
    def template_fwhms(self, template_fwhms):
        """template maps' fwhms"""
        if template_fwhms is not None:
            assert isinstance(template_fwhms, dict)
            assert (template_fwhms.keys() == self._templates.keys())
            self._template_fwhms = template_fwhms
        else:
            self._template_fwhms = dict()
            if self._template_flag:
                for name in self._template_freqs:
                    self._template_fwhms[name] = None
        log.debug('template fwhms loaded')
    
    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones((1,self._npix),dtype=np.float32)
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = mask.copy()
        # clean up input maps with mask
        self._mask[:,self._mask[0]==0.] = 0.
        self._signals[:,:,self._mask[0]==0.] = 0.
        if self._variances is not None:
            self._variances[:,:,self._mask[0]==0.] = 0.
        if self._templates is not None:
            for name in self._templates.keys():
                self._templates[name][:,self._mask[0]==0.] = 0.
        log.debug('mask map loaded')
        
    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug
        log.debug('debug mode: %s' % str(self._debug))
        
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

    def __call__(self,aposcale,psbin,lmin,lmax,nsamp,kwargs=dict()):
        """
        Parameters
        ----------
        kwargs : dict
            extra input argument controlling sampling process

        Returns
        -------
        Dynesty sampling results
        """
        log.debug('@ tpfit_pipeline::__call__')
        if self._debug:
            print ('\n template fitting pipeline check list \n')
            print ('measurement frequency band')
            print (self._freqs)
            print ('# of frequency bands')
            print (self._nfreq)
            print ('# of maps per frequency')
            print (self._nmap)
            print ('map HEALPix Nside')
            print (self._nside)
            print ('with template?')
            print (self._template_flag)
            if self._template_flag:
                print ('template reference frequency bands')
                print (self._template_freqs)
                print ('# of template frequency bands')
                print (self._template_nfreq)
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
        return self.run(aposcale,psbin,lmin,lmax,nsamp,kwargs)
        
    def run(self,aposcale=6.,psbin=20,lmin=None,lmax=None,nsamp=500,kwargs=dict()):
        """
        # preprocess
        # x_hat: mreasured PS bandpower, with measurement noise included naturally
        # x_fid: fiducial PS bandpower, without measurement noise
        # n_hat: mean of noise PS bandpower
        # x_mat: covariance of vectorized x_hat
        """
        x_hat, x_fid, n_hat, x_mat = self.preprocess(aposcale,psbin,lmin,lmax,nsamp)
        return self.analyse(x_hat,x_fid,n_hat,x_mat,kwargs)

    def analyse(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        return self._anadict[self._likelihood](x_hat,x_fid,n_hat,x_mat,kwargs)

    def analyse_simple(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        # simple likelihood simplifies the usage of noise and fiducial model
        engine = tpfit_simple(x_hat-n_hat,x_mat,self._background,self._foreground)
        if (len(self._param_range)):
            engine.rerange(self._param_range)
        result = engine(kwargs)
        # rescale variables to parameters
        names = list(engine.param_range.keys())
        for i in range(len(names)):
            low, high = engine.param_range[names[i]]
            for j in range(result.samples.shape[0]):
                result.samples[j, i] = unity_mapper(result.samples[j, i], [low, high])
        return result

    def analyse_hl(self,x_hat,x_fid,n_hat,x_mat,kwargs):
        engine = tpfit_hl(x_hat,x_fid,n_hat,x_mat,self._background,self._foreground)
        if (len(self._param_range)):
            engine.rerange(self._param_range)
        result = engine(kwargs)
        # rescale variables to parameters
        names = list(engine.param_range.keys())
        for i in range(len(names)):
            low, high = engine.param_range[names[i]]
            for j in range(result.samples.shape[0]):
                result.samples[j, i] = unity_mapper(result.samples[j, i], [low, high])
        return result

    def preprocess(self,aposcale,psbin,lmin,lmax,nsamp):
        # prepare model, parameter list generated during init models
        if self._foreground is not None:
            self._foreground = self._foreground(self._freqs,self._nmap,self._mask,aposcale,psbin,lmin,lmax,self._templates,self._template_fwhms)
        else:
            self._foreground = None
        if self._background is not None:
            self._background = self._background(self._freqs,self._nmap,self._mask,aposcale,psbin,lmin,lmax)
        else:
            self._background = None
        # prepare fiducial cambmodel
        fiducial_model = cambmodel(self._freqs,self._nmap,self._mask,aposcale,psbin,lmin,lmax)
        self._fiducial = np.transpose(fiducial_model._templates['total'])
        # prepare ps estimator
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmin=lmin,lmax=lmax)
        # estimate X_hat and M
        return self._preprodict[self._nmap](nsamp)

    def preprocess_T(self,nsamp):
        """estimate measurements band power (vectorized) and its corresponding covariance matrix.
        
        Note that the 1st multipole bin is ignored.
        """
        modes = self._est.modes
        wsp_dict = dict()  # wsp pool
        noise_map = np.zeros((2,self._npix),dtype=np.float32)  # Ti, Tj
        signal_map = np.zeros((2,self._npix),dtype=np.float32)  # Ti, Tj
        signal_ps_t = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        noise_ps_t = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        # filling wsp pool and estimated measured bandpowers
        for i in range(self._nfreq):
            tmp = self._est.auto_t(self._signals[i],fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]
            for k in range(len(modes)):
                signal_ps_t[0,k,i,i] = tmp[1][k]
            for j in range(i+1,self._nfreq):
                tmp = self._est.cross_t(np.r_[self._signals[i],self._signals[j]],fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]
                for k in range(len(modes)):
                    signal_ps_t[0,k,i,j] = tmp[1][k]
                    signal_ps_t[0,k,j,i] = tmp[1][k]
        # work out estimations
        for s in range(1,nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                noise_map[0] = np.random.normal(size=self._npix)*np.sqrt(self._variances[i,0])
                fiducial_map = hp.smoothing(hp.synfast(self._fiducial,nside=self._nside,new=True,verbose=False),fwhm=self._fwhms[i],verbose=0)
                # append noise to signal
                signal_map[0] = fiducial_map[0] + noise_map[0]
                #signal_map[0] = self._signals[i,0] + noise_map[0]
                # auto correlation
                ntmp = self._est.auto_t(noise_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = self._est.auto_t(signal_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(len(modes)):
                    noise_ps_t[s,k,i,i] = ntmp[1][k]
                    signal_ps_t[s,k,i,i] = stmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    noise_map[1] = np.random.normal(size=self._npix)*np.sqrt(self._variances[j,0])
                    fiducial_map = hp.smoothing(hp.synfast(self._fiducial,nside=self._nside,new=True,verbose=False),fwhm=self._fwhms[j],verbose=0)
                    signal_map[1] = fiducial_map[0] + noise_map[1]
                    #signal_map[1] = self._signals[j,0] + noise_map[1]
                    # cross correlation
                    ntmp = self._est.cross_t(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = self._est.cross_t(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        noise_ps_t[s,k,i,j] = ntmp[1][k]
                        noise_ps_t[s,k,j,i] = ntmp[1][k]
                        signal_ps_t[s,k,i,j] = stmp[1][k]
                        signal_ps_t[s,k,j,i] = stmp[1][k]
        if self._debug:
            return ( signal_ps_t, noise_ps_t )
        return ( signal_ps_t[0], np.mean(signal_ps_t[1:],axis=0), np.mean(noise_ps_t[1:],axis=0) ,oas_cov(vec_simple(signal_ps_t[1:])) )

    def preprocess_B(self,nsamp):
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        modes = self._est.modes  # angular modes
        noise_ps_b = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        signal_ps_b = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        noise_map = np.zeros((4,self._npix),dtype=np.float32)  # Qi Ui Qj Uj
        signal_map = np.zeros((4,self._npix),dtype=np.float32)  # Qi Ui Qj Uj
        # filling wsp pool and estimated measured bandpowers
        for i in range(self._nfreq):
            tmp = self._est.auto_eb(self._signals[i],fwhms=self._fwhms[i])
            wsp_dict[(i,i)] = tmp[-1]  # register workspace
            for k in range(len(modes)):
                signal_ps_b[0,k,i,i] = tmp[2][k]
            for j in range(i+1,self._nfreq):
                tmp = self._est.cross_eb(np.r_[self._signals[i],self._signals[j]],fwhms=[self._fwhms[i],self._fwhms[j]])
                wsp_dict[(i,j)] = tmp[-1]  # register workspace
                for k in range(len(modes)):
                    signal_ps_b[0,k,i,j] = tmp[2][k]
                    signal_ps_b[0,k,j,i] = tmp[2][k]
        # work out estimations
        for s in range(1,nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                noise_map[0] = np.random.normal(size=self._npix)*np.sqrt(self._variances[i,0])
                noise_map[1] = np.random.normal(size=self._npix)*np.sqrt(self._variances[i,1])
                # append noise to signal
                fiducial_map = hp.smoothing(hp.synfast(self._fiducial,nside=self._nside,new=True,verbose=False),fwhm=self._fwhms[i],verbose=0)
                signal_map[0] = fiducial_map[1] + noise_map[0]
                signal_map[1] = fiducial_map[2] + noise_map[1]
                #signal_map[0] = self._signals[i,0] + noise_map[0]
                #signal_map[1] = self._signals[i,1] + noise_map[1]
                # auto correlation
                ntmp = self._est.auto_eb(noise_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                stmp = self._est.auto_eb(signal_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                # assign results
                for k in range(len(modes)):
                    noise_ps_b[s,k,i,i] = ntmp[2][k]
                    signal_ps_b[s,k,i,i] = stmp[2][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    noise_map[2] = np.random.normal(size=self._npix)*np.sqrt(self._variances[j,0])
                    noise_map[3] = np.random.normal(size=self._npix)*np.sqrt(self._variances[j,1])
                    fiducial_map = hp.smoothing(hp.synfast(self._fiducial,nside=self._nside,new=True,verbose=False),fwhm=self._fwhms[j],verbose=0)
                    signal_map[2] = fiducial_map[1] + noise_map[2]
                    signal_map[3] = fiducial_map[2] + noise_map[3]
                    #signal_map[2] = self._signals[j,0] + noise_map[2]
                    #signal_map[3] = self._signals[j,1] + noise_map[3]
                    # cross correlation
                    ntmp = self._est.cross_eb(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    stmp = self._est.cross_eb(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        noise_ps_b[s,k,i,j] = ntmp[2][k]
                        noise_ps_b[s,k,j,i] = ntmp[2][k]
                        signal_ps_b[s,k,i,j] = stmp[2][k]
                        signal_ps_b[s,k,j,i] = stmp[2][k]
        if self._debug:
            return ( signal_ps_b, noise_ps_b )
        return ( signal_ps_b[0], np.mean(signal_ps_b[1:],axis=0), np.mean(noise_ps_b[1:],axis=0), oas_cov(vec_simple(signal_ps_b[1:])) )

    def reprocess(self,signals):
        return self._reprodict[self._nmap](signals)
        
    def reprocess_T(self,signals):
        pass
        
    def reprocess_B(self,signals):
        """
        use new signal dict and pre-calculated noise level
        to produce new dl_hat
        """
        # read new signals dict
        assert isinstance(signals, dict)
        assert (self._freqs == list(signals.keys()))
        assert (len(signals[next(iter(signals))].shape) == 2)
        assert (self._nmap == signals[next(iter(signals))].shape[0])
        assert (self._npix == signals[next(iter(signals))].shape[1])
        new_signals = np.r_[[signals[x] for x in sorted(signals.keys())]]
        #
        modes = self._est.modes  # angular modes
        signal_ps = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        if (self._nmap == 1):
            for i in range(self._nfreq):
                tmp = self._est.auto_t(new_signals[i],fwhms=self._fwhms[i])
                for k in range(len(modes)):
                    signal_ps[k,i,i] = tmp[1][k]
                for j in range(i+1,self._nfreq):
                    tmp = self._est.cross_t(np.r_[new_signals[i],new_signals[j]],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        signal_ps[k,i,j] = tmp[1][k]
                        signal_ps[k,j,i] = tmp[1][k]
        elif (self._nmap == 2):
            for i in range(self._nfreq):
                tmp = self._est.auto_eb(new_signals[i],fwhms=self._fwhms[i])
                for k in range(len(modes)):
                    signal_ps[k,i,i] = tmp[2][k]
                for j in range(i+1,self._nfreq):
                    tmp = self._est.cross_eb(np.r_[new_signals[i],new_signals[j]],fwhms=[self._fwhms[i],self._fwhms[j]])
                    for k in range(len(modes)):
                        signal_ps[k,i,j] = tmp[2][k]
                        signal_ps[k,j,i] = tmp[2][k]
        return signal_ps
