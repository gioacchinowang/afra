"""
Galactic foreground models

- syncmodel
    synchrotron cross-correlation band power
    
- dustmodel
    thermal dust cross-correlation band power
    
- syncdustmodel
    synchrotron-dust correlation contribution band power
"""
import numpy as np
from afra.tools.icy_decorator import icy
from afra.tools.ps_estimator import pstimator


@icy
class fgmodel(object):

    def __init__(self, freqlist, target, mask, aposcale, psbin, lmin=None, lmax=None, templates=None, template_fwhms=None):
        self.freqlist = freqlist
        self.target = target
        self.mask = mask
        self.aposcale = aposcale
        self.psbin = psbin
        self.templates = templates
        self.template_fwhms = template_fwhms
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=self._aposcale,psbin=self._psbin,lmin=lmin,lmax=lmax,target=self._target)  # init PS estimator
        self._modes = self._est.modes  # discard 1st multipole bin
        self._params = dict()  # base class holds empty dict
        self._params_dft = dict()
        self._template_ps = dict()

    @property
    def freqlist(self):
        return self._freqlist

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def target(self):
        return self._target

    @property
    def modes(self):
        return self._modes

    @property
    def npix(self):
        return self._npix

    @property
    def nside(self):
        return self._nside

    @property
    def mask(self):
        return self._mask

    @property
    def aposcale(self):
        return self._aposcale

    @property
    def psbin(self):
        return self._psbin

    @property
    def templates(self):
        return self._templates

    @property
    def template_freqlist(self):
        return self._template_freqlist

    @property
    def template_nfreq(self):
        return self._template_nfreq

    @property
    def template_fwhms(self):
        return self._template_fwhms

    @property
    def params(self):
        return self._params

    @property
    def param_dft(self):
        return self._param_dft

    @property
    def param_list(self):
        return self._param_list

    @property
    def est(self):
        return self._est

    @property
    def template_ps(self):
        return self._template_ps

    @template_ps.setter
    def template_ps(self, template_ps):
        assert isinstance(template_ps, dict)

    @freqlist.setter
    def freqlist(self, freqlist):
        assert isinstance(freqlist, (list,tuple))
        self._freqlist = freqlist
        self._nfreq = len(self._freqlist)

    @target.setter
    def target(self, target):
        assert isinstance(target, str)
        self._target = target

    @aposcale.setter
    def aposcale(self, aposcale):
        self._aposcale = aposcale

    @psbin.setter
    def psbin(self, psbin):
        self._psbin = psbin

    @mask.setter
    def mask(self, mask):
        assert isinstance(mask, np.ndarray)
        self._mask = mask.copy()
        self._npix = len(mask)
        self._nside = int(np.sqrt(self._npix//12))

    @templates.setter
    def templates(self, templates):
        """template maps at two frequency bands"""
        if templates is not None:
            assert isinstance(templates, dict)
            self._template_freqlist = sorted(templates.keys())
            self._template_nfreq = len(templates)
            assert (self._template_nfreq < 3)
            assert (templates[next(iter(templates))].shape == (3,self._npix))
            self._template_flag = True
            self._templates = templates
            #apply mask
            for key in templates.keys():
                self._templates[key][:,self._mask<1.] = 0.
        else:
            self._template_flag = False
            self._templates = None

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
                for name in self._template_freqlist:
                    self._template_fwhms[name] = None

    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params.update({name: pdict[name]})

    def i2cmb(self,freq,ref):
            """Brightness flux to CMB temperature unit converting ratio"""
            hGk_t0 = 0.04799340833334541/2.73  # h*GHz/k_B/T0
            p = hGk_t0*freq
            p0 = hGk_t0*ref
            return (ref/freq)**4*np.exp(p0-p)*(np.exp(p)-1.)**2/(np.exp(p0)-1.)**2


@icy
class syncmodel(fgmodel):
    
    def __init__(self, freqlist, target, mask, aposcale, psbin, lmin=None, lmax=None, templates=None, template_fwhms=None):
        super(syncmodel, self).__init__(freqlist,target,mask,aposcale,psbin,lmin,lmax,templates,template_fwhms)
        assert (self._template_nfreq == 1)
        # # setup self.params' keys by param_list and content by param_dft
        self.reset(self.default)
        # calculate template PS
        if self._template_flag:
            for ref in self._template_freqlist:
                self.template_ps[ref] = self._est.autoBP(self._templates[ref],fwhms=self._template_fwhms[ref])[1]

    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        if not self._template_flag:
            prefix = ['bp_s_'+self._target+'_']
            for i in prefix:
            	for j in range(len(self._modes)):
                	plist.append(i+str(self._modes[j]))
        plist.append('beta_s')
        return plist

    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        if not self._template_flag:
            for i in _tmp[:-1]:
                prange[i] = [0.,1.e+4]
        prange['beta_s'] = [-5.,0.]
        return prange

    @property
    def default(self):
        """register default parameter values
        """
        prange = self.param_range
        pdft = dict()
        for key in prange.keys():
            pdft[key] = 0.5*(prange[key][0] + prange[key][1])
        return pdft

    def bandpower(self):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_s = self.params['beta_s']
        ref = self._template_freqlist[0]
        # estimate bandpower from templates
        ps_s = self._template_ps[ref]
        dl = np.zeros((len(self._modes),self._nfreq,self._nfreq))
        for l in range(len(self._modes)):
            bp_s = ps_s[l]
            for i in range(self._nfreq):
                f_ratio_i = (self._freqlist[i]/ref)**beta_val_s
                c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                dl[l,i,i] = bp_s * (f_ratio_i*c_ratio_i)**2
                for j in range(i+1,self._nfreq):
                    f_ratio_j = (self._freqlist[j]/ref)**beta_val_s
                    c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                    dl[l,i,j] = bp_s * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                    dl[l,j,i] = dl[l,i,j]
        return dl


@icy
class dustmodel(fgmodel):
   
    def __init__(self, freqlist, target, mask, aposcale, psbin, lmin=None, lmax=None, templates=None, template_fwhms=None):
        super(dustmodel, self).__init__(freqlist,target,mask,aposcale,psbin,lmin,lmax,templates,template_fwhms)
        assert (self._template_nfreq == 1)
        # setup self.params' keys by param_list and content by param_dft
        self.reset(self.default)
        # calculate template PS
        if self._template_flag:
            for ref in self._template_freqlist:
                self.template_ps[ref] = self._est.autoBP(self._templates[ref],fwhms=self._template_fwhms[ref])[1]

    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        if not self._template_flag:
            prefix = ['bp_d_'+self._target+'_']
            for i in prefix:
            	for j in range(len(self._modes)):
                	plist.append(i+str(self._modes[j]))
        plist.append('beta_d')
        return plist

    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        if not self._template_flag:
            for i in _tmp[:-1]:
                prange[i] = [0.,1.e+4]
        prange['beta_d'] = [0.,5.]
        return prange

    @property
    def default(self):
        """register default parameter values
        """
        prange = self.param_range
        pdft = dict()
        for key in prange.keys():
            pdft[key] = 0.5*(prange[key][0] + prange[key][1])
        return pdft

    def bratio(self,freq,ref):
        hGk_td = 0.04799340833334541/19.6  # h*GHz/k_B/Td
        return (freq/ref)**3*(np.exp(hGk_td*ref)-1.)/(np.exp(hGk_td*freq)-1.)

    def bandpower(self):
        """dust model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_d = self.params['beta_d']
        ref = self._template_freqlist[0]
        ps_d = self._template_ps[ref]
        # estimate bandpower from templates
        dl = np.zeros((len(self._modes),self._nfreq,self._nfreq))
        for l in range(len(self._modes)):
            bp_d = ps_d[l]
            for i in range(self._nfreq):
                f_ratio_i = (self._freqlist[i]/ref)**beta_val_d*self.bratio(self._freqlist[i],ref)
                c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                dl[l,i,i] = bp_d * (f_ratio_i*c_ratio_i)**2
                for j in range(i+1,self._nfreq):
                    f_ratio_j = (self._freqlist[j]/ref)**beta_val_d*self.bratio(self._freqlist[j],ref)
                    c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                    dl[l,i,j] = bp_d * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                    dl[l,j,i] = dl[l,i,j]
        return dl


@icy
class syncdustmodel(fgmodel):

    def __init__(self, freqlist, target, mask, aposcale, psbin, lmin=None, lmax=None, templates=None, template_fwhms=None):
        super(syncdustmodel, self).__init__(freqlist,target,mask,aposcale,psbin,lmin,lmax,templates,template_fwhms)
        assert (self._template_nfreq == 2)
        # setup self.params' keys by param_list and content by param_dft
        self.reset(self.default)
        # calculate template PS
        if self._template_flag:
            for ref in self._template_freqlist:
                self.template_ps[ref] = self._est.autoBP(self._templates[ref],fwhms=self._template_fwhms[ref])[1]

    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        if not self._template_flag:
            prefix = ['bp_s_'+self._target+'_','bp_d_'+self._target+'_']
            for i in prefix:
            	for j in range(len(self._modes)):
                	plist.append(i+str(self._modes[j]))
        plist.append('beta_s')
        plist.append('beta_d')
        plist.append('rho')
        return plist

    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        if not self._template_flag:
            for i in _tmp[:-3]:
                prange[i] = [0.,1.e+4]
        prange['beta_s'] = [-5.,0.]
        prange['beta_d'] = [0.,5.]
        prange['rho'] = [-1.,1.]
        return prange

    @property
    def default(self):
        """register default parameter values
        """
        prange = self.param_range
        pdft = dict()
        for key in prange.keys():
            pdft[key] = 0.5*(prange[key][0] + prange[key][1])
        return pdft

    def bratio(self,freq,ref):
        hGk_td = 0.04799340833334541/19.6  # h*GHz/k_B/Td
        return (freq/ref)**3*(np.exp(hGk_td*ref)-1.)/(np.exp(hGk_td*freq)-1.)

    def bandpower(self):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_s = self.params['beta_s']
        beta_val_d = self.params['beta_d']
        rho = self.params['rho']
        ref = self._template_freqlist
        ps_s = self._template_ps[ref[0]]
        ps_d = self._template_ps[ref[1]]
        # estimate bandpower from templates
        dl = np.zeros((len(self._modes),self._nfreq,self._nfreq))
        for l in range(len(self._modes)):
            bp_s = ps_s[l]
            bp_d = ps_d[l]
            for i in range(self._nfreq):
                f_ratio_is = (self._freqlist[i]/ref[0])**beta_val_s
                f_ratio_id = (self._freqlist[i]/ref[1])**beta_val_d*self.bratio(self._freqlist[i],ref[1])
                c_ratio_is = self.i2cmb(self._freqlist[i],ref[0])
                c_ratio_id = self.i2cmb(self._freqlist[i],ref[1])
                dl[l,i,i] = bp_s * (f_ratio_is*c_ratio_is)**2
                dl[l,i,i] += bp_d * (f_ratio_id*c_ratio_id)**2
                dl[l,i,i] += rho * np.sqrt(bp_s*bp_d) * (f_ratio_is*f_ratio_id*c_ratio_is*c_ratio_id)
                for j in range(i+1,self._nfreq):
                    f_ratio_js = (self._freqlist[j]/ref[0])**beta_val_s
                    f_ratio_jd = (self._freqlist[j]/ref[1])**beta_val_d*self.bratio(self._freqlist[j],ref[1])
                    c_ratio_js = self.i2cmb(self._freqlist[j],ref[0])
                    c_ratio_jd = self.i2cmb(self._freqlist[j],ref[1])
                    dl[l,i,j] = bp_s * (f_ratio_is*c_ratio_is*f_ratio_js*c_ratio_js)
                    dl[l,i,j] += bp_d * (f_ratio_id*c_ratio_id*f_ratio_jd*c_ratio_jd)
                    dl[l,i,j] += rho * np.sqrt(bp_s*bp_d) * ( f_ratio_is*c_ratio_is*f_ratio_jd*c_ratio_jd + f_ratio_id*c_ratio_id*f_ratio_js*c_ratio_js  )
                    dl[l,j,i] = dl[l,i,j]
        return dl
