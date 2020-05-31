import logging as log
import numpy as np
import dynesty
from abspy.tools.fg_models import syncdustmodel
from abspy.tools.bg_models import cmbmodel
from abspy.tools.ps_estimator import pstimator
from abspy.tools.icy_decorator import icy


@icy
class tpfpipe(object):
    """
    The template fitting pipeline class.
    
    without template (template=None):
        D_b,l at effective ells
        D_s,l at effective ells
        D_d,l at effective ells
        foreground frequency scaling parameters
        foregorund cross-corr parameters
        
    with low+high frequency templates:
        D_b,l at effective ells
        foreground frequency scaling parameters
        foregorund cross-corr parameters
    """
    def __init__(self, singals, freqs, nmap, nside, variances, mask=None, fwhms=None, templates=None, refs=[30.,353.], sampling_opt=dict()):
        """
        Parameters
        ----------
        
        signals : numpy.ndarray
            Measured signal maps,
            should be arranged in shape: (frequency #, map #, pixel #).
            
        variances : numpy.ndarray
            Measured noise variance maps,
            should be arranged in shape: (frequency #, map #, pixel #).
            By default, no variance maps required.
            
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (1, pixel #).
            
        freqs : list, tuple
            List of frequencies of measurements
            
        nmap : int
            Number of maps,
            if 1, taken as T maps only,
            if 2, taken as Q,U maps only,
            
        nside : int
            HEALPix Nside
            
        fwhms : list, tuple
            FWHM of gaussian beams for each frequency
            
        templates : numpy.ndarray
            Template map,
            should be arranged in shape: (ref frequency #, map #, pixel #)
            
        refs : list, tuple
            reference frequency of template maps,
            two element list, first for synchrotron, second for dust
            
        sampling_opt : dict
            Dynesty smapling options
        """
        log.debug('@ tpfpipe::__init__')
        # basic settings
        self.freqs = freqs
        self.nmap = nmap
        self.nside = nside
        self.fwhms = fwhms
        # measurements
        self.signals = signals
        self.variances = variances
        self.mask = mask
        # reference frequencies
        self.refs = refs
        # adding template maps (for estimating template PS band power)
        self.templates = templates
        # sampling optinos
        self.sampling_opt = sampling_opt
        # init active parameter list
        self.active_param_list = list()
        # PS estimation setting
        self._psbin = 20
        self._nsamp = 500
        
    @property
    def freqs(self):
        return self._freqs
        
    @property
    def nmap(self):
        return self._nmap
        
    @property
    def nside(self):
        return self._nside
        
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
    def refs(self):
        return self._refs
        
    @property
    def templates(self):
        return self._templates
        
    @property
    def sampling_opt(self):
        return self._sampling_opt
        
    @property
    def active_param_list(self):
        """active parameter name list"""
        return self._active_param_list
        
    @property
    def nparam(self):
        """number of active parameters"""
        return len(self._active_param_list)
        
    @refs.setter
    def refs(self, refs):
        assert (len(refs) == 2)  # need two refernece frequencies
        assert (refs[0] < refs[1])  # 1st for sync, 2nd for dust
        self._refs = refs
        log.debug('reference frequencies %s' % str(self._refs))
        
    @sampling_opt.setter
    def sampling_opt(self, opt):
        assert isinstance(opt, dict)
        self._sampling_opt = opt
        
    @freqs.setter
    def freqs(self, freqs):
        assert isinstance(freqs, (list,tuple))
        self._freqs = freqs
        self._nfreq = len(freqs)
        log.debug('number of frequencies %s' % str(self._nfreq))
        
    @nmap.setter
    def nmap(self, nmap):
        assert isinstance(nmap, int)
        assert (nmap > 0)
        self._nmap = nmap
        log.debug('number of maps %s' % str(self._nmap))
        
    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        self._npix = 12*nside**2
        log.debug('HEALPix Nside %s' % str(self._nside))
        
    @fwhms.setter
    def fwhms(self, fwhms):
        """FWHM for each frequency"""
        assert isinstance(fwhms, (list,tuple))
        assert (len(fwhms) == self._nfreq)
        self._fwhms = fwhms
        
    @active_param_list.setter
    def active_param_list(self, active_param_list):
        assert isinstance(active_param_list, (list,tuple))
        self._active_param_list = active_param_list
        
    @signals.setter
    def signals(self, signals):
        assert isinstance(signals, np.ndarray)
        assert (signals.shape == (self._nfreq,self._nmap,self._npix))
        self._signals = signals
        log.debug('signal maps loaded')
        
    @variances.setter
    def variances(self, variances):
        assert isinstance(variances, np.ndarray)
        assert (variances.shape == (self._nfreq,self._nmap,self._npix))
        self._variances = variances
        log.debug('variance maps loaded')
        
    @templates.setter
    def templates(self, templates):
        """template maps at two frequency bands"""
        if templates is not None:
            assert isinstance(templates, np.ndarray)
            assert (templates.shape == (2,self._nmap,self._npix))
            self._template_flag = True
        else:
            self._template_flag = False
        self._templates = templates
        log.debug('template maps loaded')
    
    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones((1,self._npix),dtype=bool)
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = mask
        log.debug('mask map loaded')
        
    def __call__(self, kwargs=dict()):
        """
        Parameters
        ----------
        kwargs : dict
            extra input argument controlling sampling process
            i.e., 'dlogz' for stopping criteria

        Returns
        -------
        Dynesty sampling results
        """
        log.debug('@ tpfit_pipeline::__call__')
        return self.run(kwargs)
        
    def run(self, kwargs=dict()):
        # conduct PS estimation
        bprslt = self.bpestimator()
        # models
        foreground = syncdustmodel(bprslt[0])
        background = cmbmodel(bprslt[0])
        #@property
        #def param_list(self):
        #"""combined parameter list of foreground and background mdoels"""
        #return self._fgmodel.param_list + self._bgmodel.param_list
        
        '''
        assert isinstance(fgmodel, fgmodel)
        self._fgmodel = fgmodel
        log.debug('foreground model set')
        
        assert isinstance(bgmodel, bgmodel)
        self._bgmodel = bgmodel
        log.debug('background model set')
        '''
        
        '''
        # init dynesty
        sampler = dynesty.NestedSampler(self._core_likelihood,
                                        self.prior,
                                        len(self.active_parameters),
                                        **self.sampling_opt)
        sampler.run_nested(**kwargs)
        return sampler.results
        '''
        return bprslt
        
    def info(self):
        print ('sampling check list')
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
        print ('template reference frequency bands')
        print (self._refs)
        print ('FWHMs')
        print (self._fwhms)
        print ('active parameter list')
        print (self._active_param_list)
        print ('PS estimation angular modes bin size, self._psbin')
        print (self._psbin)
        print ('PS esitmation noise resampling size, self._nsamp')
        print (self._nsamp)
        
    def bpestimator(self):
        """measurements band power estimator,
        apodization scale bydefault set as 5.0
        
        Returns
        -------
        
        angular mode list
        
        band power mean in shape (# modes, # freq, # freq)
        
        band power std in shape (# modes, # freq, # freq)
        """
        if (self._nmap == 1):
            # estimate noise PS and noise RMS
            est = pstimator(nside=self._nside,mask=self._mask,aposcale=5.0,psbin=self._psbin)  # init PS estimator
            # run trial PS estimations for workspace template
            wsp_dict = dict()
            modes = list()
            for i in range(self._nfreq):
                tmp = est.auto_t(self._signals[0,0].reshape(1,-1),fwhms=self._fwhms[i])
                wsp_dict[(i,i)] = tmp[-1]  # register workspace
                modes = list(tmp[0])  # register angular modes
                for j in range(i+1,self._nfreq):
                    tmp = est.cross_t(self._signals[:2,0],fwhms=[self._fwhms[i],self._fwhms[j]])
                    wsp_dict[(i,j)] = tmp[-1]  # register workspace
            nell = len(modes)  # know the number of angular modes
            # allocate
            ps_t_mean = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            ps_t_std = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            signal_map = np.zeros((2,self._npix),dtype=np.float64)
            for s in range(self._nsamp):
                # prepare noise samples on-fly
                for i in range(self._nfreq):
                    # noise realization
                    signal_map[0] = self._signals[i,0] + self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variances[i,0])
                    # auto correlation
                    stmp = est.auto_t(signal_map[0].reshape(1,-1),wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                    # assign results
                    for k in range(nell):
                        ps_t_mean[k,i,i] += stmp[1][k]
                        ps_t_std[k,i,i] += stmp[1][k]**2
                    # cross correlation
                    for j in range(i+1,self._nfreq):
                        # noise realization
                        signal_map[1] = self._signals[j,0] + self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variances[j,0])
                        # cross correlation
                        stmp = est.cross_t(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                        for k in range(nell):
                            ps_t_mean[k,i,j] += stmp[1][k]
                            ps_t_mean[k,j,i] += stmp[1][k]
                            ps_t_std[k,i,j] += stmp[1][k]**2
                            ps_t_std[k,j,i] += stmp[1][k]**2
            return (modes, ps_t_mean/self._nsamp, np.sqrt(ps_t_std/self._nsamp - (ps_t_mean/self._nsamp)**2))
        elif (self._nmap == 2):
            # estimate noise PS and noise RMS
            est = pstimator(nside=self._nside,mask=self._mask,aposcale=5.0,psbin=self._psbin)  # init PS estimator
            # run trial PS estimations for workspace template
            wsp_dict = dict()
            modes = list()
            for i in range(self._nfreq):
                tmp = est.auto_eb(self._signals[0],fwhms=self._fwhms[i])
                wsp_dict[(i,i)] = tmp[-1]  # register workspace
                modes = list(tmp[0])  # register angular modes
                for j in range(i+1,self._nfreq):
                    tmp = est.cross_eb(self._signals[:2].reshape(4,-1),fwhms=[self._fwhms[i],self._fwhms[j]])
                    wsp_dict[(i,j)] = tmp[-1]  # register workspace
            nell = len(modes)  # know the number of angular modes
            # allocate
            ps_e_mean = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            ps_b_mean = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            ps_e_std = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            ps_b_std = np.zeros((nell,self._nfreq,self._nfreq),dtype=np.float64)
            noise_map = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
            signal_map = np.zeros((4,self._npix),dtype=np.float64)  # Qi Ui Qj Uj
            for s in range(self._nsamp):
                # prepare noise samples on-fly
                for i in range(self._nfreq):
                    # noise realization
                    signal_map[0] = self._signals[i,0] + self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variances[i,0])
                    signal_map[1] = self._signals[i,1] + self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variances[i,1])
                    # auto correlation
                    stmp = est.auto_eb(signal_map[:2],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[i])
                    # assign results
                    for k in range(nell):
                        ps_e_mean[k,i,i] += stmp[1][k]
                        ps_e_std[k,i,i] += stmp[1][k]**2
                        ps_b_mean[k,i,i] += stmp[2][k]
                        ps_b_std[k,i,i] += stmp[2][k]**2
                    # cross correlation
                    for j in range(i+1,self._nfreq):
                        # noise realization
                        signal_map[2] = self._signals[j,0] + self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variances[j,0])
                        signal_map[3] = self._signals[j,1] + self._mask[0]*np.random.normal(size=self._npix)*np.sqrt(self._variances[j,1])
                        # cross correlation
                        stmp = est.cross_eb(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[i],self._fwhms[j]])
                        for k in range(nell):
                            ps_e_mean[k,i,j] += stmp[1][k]
                            ps_e_mean[k,j,i] += stmp[1][k]
                            ps_e_std[k,i,j] += stmp[1][k]**2
                            ps_e_std[k,j,i] += stmp[1][k]**2
                            ps_b_mean[k,i,j] += stmp[2][k]
                            ps_b_mean[k,j,i] += stmp[2][k]
                            ps_b_std[k,i,j] += stmp[2][k]**2
                            ps_b_std[k,j,i] += stmp[2][k]**2
            return (modes, ps_e_mean/self._nsamp, np.sqrt(ps_e_std/self._nsamp - (ps_e_mean/self._nsamp)**2), ps_b_mean/self._nsamp, np.sqrt(ps_b_std/self._nsamp - (ps_b_mean/self._nsamp)**2))
        else:
            raise ValueError('unsupported number of maps')
        
    def activate(self, pname):
        """set a parameter as active parameter"""
        log.debug('@ tpfit_pipeline::set_active')
        assert isinstance(pname, str)
        if pname in self.active_param_list:
            print ('%s already activated' % pname)
        elif pname in self._param_list:
            self._active_param_list.append(pname)
            print ('%s activated' % pname)
        else:
            raise ValueError('unknown parameter name')
        
    def prior(self, cube):
        """flat prior"""
        return cube
        
    def _core_likelihood(self, cube):
        """
        core log-likelihood calculator
        cube remains the same on each node
        now self._simulator will work on each node and provide multiple ensemble size

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
        # return active variables from pymultinest cube to factories
        # and then generate new field objects
        head_idx = int(0)
        tail_idx = int(0)
        field_list = tuple()
        # random seeds manipulation
        self._randomness()
        # the ordering in factory list and variable list is vital
        for factory in self._factory_list:
            variable_dict = dict()
            tail_idx = head_idx + len(factory.active_parameters)
            factory_cube = cube[head_idx:tail_idx]
            for i, av in enumerate(factory.active_parameters):
                variable_dict[av] = factory_cube[i]
            field_list += (factory.generate(variables=variable_dict,
                                            ensemble_size=self._ensemble_size,
                                            ensemble_seeds=self._ensemble_seeds),)
            log.debug('create '+factory.name+' field')
            head_idx = tail_idx
        assert(head_idx == len(self._active_parameters))
        observables = self._simulator(field_list)
        # apply mask
        observables.apply_mask(self.likelihood.mask_dict)
        # add up individual log-likelihood terms
        current_likelihood = self.likelihood(observables)
        # check likelihood value until negative (or no larger than given threshold)
        if self._check_threshold and current_likelihood > self._likelihood_threshold:
            raise ValueError('log-likelihood beyond threashould')
        return current_likelihood * self.likelihood_rescaler
