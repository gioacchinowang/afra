import logging as log
import numpy as np
from abspy.tools.fg_models import syncdustmodel
from abspy.tools.bg_models import cmbmodel
from abspy.tools.ps_estimator import pstimator
from abspy.tools.aux import vec_simple, oas_cov, g_simple, bp_window
from abspy.tools.icy_decorator import icy
from abspy.methods.tpfit import tpfit_simple


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
    def __init__(self, signals, variances, mask=None, fwhms=None, templates=None, template_fwhms=None, likelihood='simple'):
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
    def param_list(self):
        """parameter name list"""
        return self._param_list

    @property
    def param_range(self):
        return self._param_range
        
    @property
    def nparam(self):
        """number of parameters"""
        return len(self._param_list)
        
    @param_list.setter
    def param_list(self, param_list):
        assert isinstance(param_list, (list,tuple))
        self._param_list = param_list

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
        """template maps at two frequency bands"""
        if templates is not None:
            assert isinstance(templates, dict)
            self._template_freqs = sorted(templates.keys())
            self._template_nfreq = len(self._template_freqs)
            assert (self._template_nfreq == 2)
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
        self._mask[:,self._mask[0]<1.] = 0.
        self._signals[:,:,self._mask[0]<1.] = 0.
        if self._variances is not None:
            self._variances[:,:,self._mask[0]<1.] = 0.
        if self._templates is not None:
            for name in self._templates.keys():
                self._templates[name][:,self._mask[0]<1.] = 0.
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

    def bp_window(self,aposcale,psbin,lmax=None):
        """window function matrix for converting global PS into band-powers"""
        if lmax is None:
            lmax = 3*self._nside
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin)
        return bp_window(est,lmax)
        
    def __call__(self,aposcale,psbin,nsamp,kwargs=dict()):
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
            print ('PS esitmation noise resampling size')
            print (nsamp)
            print ('\n')
        return self.run(aposcale,psbin,nsamp,kwargs)
        
    def run(self,aposcale=6.,psbin=20,nsamp=500,kwargs=dict()):
        """
        # preprocess 
        # simple likelihood
        # -> estimate Dl_hat from measurements
        # -> estimate M from noise
        
        # HL likelihood
        # -> estimate Dl_hat from measurements
        # -> generate Dl_fid from fiducial CMB model
        # -> estimate M from noise
        """
        if (self._likelihood == 'simple'):
            # estimate X_hat with measured noise
            x_hat, x_mat = self.preprocess_simple(aposcale,psbin,nsamp)
            # prepare model, parameter list generated during init models
            foreground = syncdustmodel(self._freqs,self._nmap,self._mask,aposcale,psbin,self._templates,self._template_fwhms)
            background = cmbmodel(self._freqs,self._nmap,self._mask,aposcale,psbin)
            self._param_list = foreground.param_list + background.param_list  # update parameter list from models
            engine = tpfit_simple(x_hat,x_mat,background,foreground)
            if (len(self._param_range)):
                engine.rerange(self._param_range)
            return engine(kwargs)
        elif (self._likelihood == 'hl'):
            #x_hat, x_mat = self.preprocess_hl()
            raise ValueError('unsupported likelihood type')
        else:
            raise ValueError('unsupported likelihood type')

    def preprocess_hl(self,aposcale,psbin,nsamp):
        pass 

    def preprocess_simple(self,aposcale,psbin,nsamp):
        """estimate measurements band power (vectorized) and its corresponding covariance matrix.
        
        Note that the 1st multipole bin is ignored.
        """
        est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin)  # init PS estimator
        modes = est.modes[1:]  # angular modes
        if (self._nmap == 1):
            ps_t = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
            for s in range(nsamp):
                # prepare noise samples on-fly
                for i in range(self._nfreq):
                    # auto correlation
                    ntmp = np.random.normal(size=(self._nmap,self._npix))*np.sqrt(self._variances[i])
                    stmp = est.auto_t(self._signals[i]+ntmp,fwhms=self._fwhms[i])
                    # assign results
                    for k in range(len(modes)):
                        ps_t[s,k,i,i] = stmp[1][k+1]
                    # cross correlation
                    for j in range(i+1,self._nfreq):
                        # cross correlation
                        ntmp = np.random.normal(size=(self._nmap*2,self._npix))*np.sqrt(np.r_[self._variances[i],self._variances[j]])
                        stmp = est.cross_t(np.r_[self._signals[i],self._signals[j]]+ntmp,fwhms=[self._fwhms[i],self._fwhms[j]])
                        for k in range(len(modes)):
                            ps_t[s,k,i,j] = stmp[1][k+1]
            xl_set = vec_simple(ps_t)
            return ( np.mean(xl_set,axis=0), oas_cov(xl_set) )
        elif (self._nmap == 2):
            # allocate
            ps_b = np.zeros((nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
            for s in range(nsamp):
                # prepare noise samples on-fly
                for i in range(self._nfreq):
                    # auto correlation
                    ntmp = np.random.normal(size=(self._nmap,self._npix))*np.sqrt(self._variances[i])
                    stmp = est.auto_eb(self._signals[i]+ntmp,fwhms=self._fwhms[i])
                    # assign results
                    for k in range(len(modes)):
                        ps_b[s,k,i,i] = stmp[2][k+1]
                    # cross correlation
                    for j in range(i+1,self._nfreq):
                        # cross correlation
                        ntmp = np.random.normal(size=(self._nmap*2,self._npix))*np.sqrt(np.r_[self._variances[i],self._variances[j]])
                        stmp = est.cross_eb(np.r_[self._signals[i],self._signals[j]]+ntmp,fwhms=[self._fwhms[i],self._fwhms[j]])
                        for k in range(len(modes)):
                            ps_b[s,k,i,j] = stmp[2][k+1]
            xl_set = vec_simple(ps_b)
            return ( np.mean(xl_set,axis=0), oas_cov(xl_set) )
        else:
            raise ValueError('unsupported nmap')
