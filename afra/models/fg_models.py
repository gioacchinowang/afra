import numpy as np
from afra.tools.ps_estimator import pstimator
from afra.tools.icy_decorator import icy


class fgmodel(object):

    def __init__(self, freqlist, estimator, template_bp=None):
        self.freqlist = freqlist
        self.estimator = estimator
        self.template_bp = template_bp
        self.params = dict()  # base class holds empty dict
        self.paramrange = dict()
        self.paramdft = dict()
        self.paramlist = list()
        self.blacklist = list()

    @property
    def freqlist(self):
        return self._freqlist

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def estimator(self):
        return self._estimator

    @property
    def params(self):
        return self._params

    @property
    def paramrange(self):
        return self._paramrange

    @property
    def paramdft(self):
        return self._paramdft

    @property
    def paramlist(self):
        return self._paramlist

    @property
    def blacklist(self):
        return self._blacklist

    @property
    def template_bp(self):
        return self._template_bp

    @property
    def template_flag(self):
        return self._template_flag

    @property
    def template_freqlist(self):
        return self._template_freqlist

    @property
    def template_nfreq(self):
        return self._template_nfreq

    @freqlist.setter
    def freqlist(self, freqlist):
        assert isinstance(freqlist, (list,tuple))
        self._freqlist = freqlist
        self._nfreq = len(self._freqlist)

    @estimator.setter
    def estimator(self, estimator):
        assert isinstance(estimator, pstimator)
        self._estimator = estimator

    @params.setter
    def params(self, params):
        assert isinstance(params, dict)
        self._params = params

    @paramdft.setter
    def paramdft(self, paramdft):
        assert isinstance(paramdft, dict)
        self._paramdft = paramdft

    @paramrange.setter
    def paramrange(self, paramrange):
        assert isinstance(paramrange, dict)
        self._paramrange = paramrange

    @paramlist.setter
    def paramlist(self, paramlist):
        assert isinstance(paramlist, (list,tuple))
        self._paramlist = paramlist

    @blacklist.setter
    def blacklist(self, blacklist):
        assert isinstance(blacklist, (list,tuple))
        for p in blacklist:
            assert (p in self._paramlist)
        self._blacklist = blacklist

    @template_bp.setter
    def template_bp(self, template_bp):
        """template maps at two frequency bands"""
        if template_bp is not None:
            assert isinstance(template_bp, dict)
            self._template_freqlist = sorted(template_bp.keys())
            self._template_nfreq = len(self._template_freqlist)
            assert (self._template_nfreq < 3)
            assert (template_bp[next(iter(template_bp))].shape == (self._estimator._ntarget,self._estimator._nmode))
            self._template_flag = True
            self._template_bp = template_bp
        else:
            self._template_flag = False
            self._template_freqlist = list()
            self._template_nfreq = 0
            self._template_bp = None

    def initdft(self):
        """register default parameter values
        """
        pdft = dict()
        for key in self._paramrange.keys():
            pdft[key] = 0.5*(self._paramrange[key][0] + self._paramrange[key][1])
        self.reset(pdft)
        return pdft

    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.paramlist:
                self._params.update({name: pdict[name]})

    def i2cmb(self,freq,ref):
        """Brightness flux to CMB temperature unit converting ratio"""
        hGk_t0 = 0.04799340833334541/2.73  # h*GHz/k_B/T0
        p = hGk_t0*freq
        p0 = hGk_t0*ref
        return (ref/freq)**4*np.exp(p0-p)*(np.exp(p)-1.)**2/(np.exp(p0)-1.)**2


@icy
class asyncmodel(fgmodel):
    """analytic synchrotron model,
    with parameters:
    A_s(E/B): synchrotron amplitude at ell=80
    alpha_s: multipole scaling index
    beta_s: frequency scaling index
    """

    def __init__(self, freqlist, estimator, template_bp=None):
        super(asyncmodel, self).__init__(freqlist,estimator,template_bp)
        self.paramlist = self.initlist()
        self.paramrange = self.initrange()
        self.paramdft = self.initdft()
        assert (not self._template_flag)
        # reference frequency
        self._template_freqlist = [23]

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for t in self._estimator._targets:
            plist.append('A_s'+t)
        plist.append('alpha_s')
        plist.append('beta_s')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        for t in self._estimator._targets:
            prange['A_s'+t] = [0.,1.0e+4]
        prange['alpha_s'] = [-5.,5.]
        prange['beta_s'] = [-5.,0]
        return prange

    def bandpower(self):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_s = self.params['beta_s']
        ref = self._template_freqlist[0]
        dl_tmp = self._estimator.bpconvert(np.insert((np.arange(1,self._estimator._lmax+1).astype(np.float64)/80.)**self._params['alpha_s'],[0],0.))
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float64)
        # estimate bandpower from templates
        for t in range(self._estimator._ntarget):
            bp_s = self._params['A_s'+self._estimator._targets[t]]*dl_tmp
            for i in range(self._nfreq):
                f_ratio_i = (self._freqlist[i]/ref)**beta_val_s
                c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                dl[t,:,i,i] = bp_s * (f_ratio_i*c_ratio_i)**2
                for j in range(i+1,self._nfreq):
                    f_ratio_j = (self._freqlist[j]/ref)**beta_val_s
                    c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                    dl[t,:,i,j] = bp_s * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                    dl[t,:,j,i] = dl[t,:,i,j]
        return dl


@icy
class tsyncmodel(fgmodel):
    """templated synchrotron model,
    with: 
    template given at reference frequency
    beta_s: frequency scaling index
    """
    
    def __init__(self, freqlist, estimator, template_bp=None):
        super(tsyncmodel, self).__init__(freqlist,estimator,template_bp)
        self.paramlist = self.initlist()
        self.paramrange = self.initrange()
        self.paramdft = self.initdft()
        # calculate template PS
        assert (self._template_flag)
        assert (self._template_nfreq == 1)

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        plist.append('beta_s')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        prange['beta_s'] = [-5.,0]
        return prange

    def bandpower(self):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_s = self.params['beta_s']
        ref = self._template_freqlist[0]
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float64)
        # estimate bandpower from templates
        ps_s = self._template_bp[ref]
        for t in range(self._estimator._ntarget):
            bp_s = ps_s[t]
            for i in range(self._nfreq):
                f_ratio_i = (self._freqlist[i]/ref)**beta_val_s
                c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                dl[t,:,i,i] = bp_s * (f_ratio_i*c_ratio_i)**2
                for j in range(i+1,self._nfreq):
                    f_ratio_j = (self._freqlist[j]/ref)**beta_val_s
                    c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                    dl[t,:,i,j] = bp_s * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                    dl[t,:,j,i] = dl[t,:,i,j]
        return dl


@icy
class adustmodel(fgmodel):

    def __init__(self, freqlist, estimator, template_bp=None):
        super(adustmodel, self).__init__(freqlist,estimator,template_bp)
        self.paramlist = self.initlist()
        self.paramrange = self.initrange()
        self.paramdft = self.initdft()
        assert (not self._template_flag)
        self._template_freqlist = [353.]

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for t in self._estimator._targets:
            plist.append('A_d'+t)
        plist.append('alpha_d')
        plist.append('beta_d')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        for t in self._estimator._targets:
            prange['A_d'+t] = [0.,1.0e+4]
        prange['alpha_d'] = [-5.,5.]
        prange['beta_d'] = [0.,5.]
        return prange

    def bratio(self,freq,ref):
        hGk_td = 0.04799340833334541/19.6  # h*GHz/k_B/Td
        return (freq/ref)**3*(np.exp(hGk_td*ref)-1.)/(np.exp(hGk_td*freq)-1.)

    def bandpower(self):
        """dust model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_d = self.params['beta_d']
        ref = self._template_freqlist[0]
        dl_tmp = self._estimator.bpconvert(np.insert((np.arange(1,self._estimator._lmax+1).astype(np.float64)/80.)**self._params['alpha_d'],[0],0.))
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float64)
        for t in range(self._estimator._ntarget):
            bp_d = self._params['A_d'+self._estimator._targets[t]]*dl_tmp
            for i in range(self._nfreq):
                f_ratio_i = (self._freqlist[i]/ref)**beta_val_d*self.bratio(self._freqlist[i],ref)
                c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                dl[t,:,i,i] = bp_d * (f_ratio_i*c_ratio_i)**2
                for j in range(i+1,self._nfreq):
                    f_ratio_j = (self._freqlist[j]/ref)**beta_val_d*self.bratio(self._freqlist[j],ref)
                    c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                    dl[t,:,i,j] = bp_d * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                    dl[t,:,j,i] = dl[t,:,i,j]
        return dl


@icy
class tdustmodel(fgmodel):
   
    def __init__(self, freqlist, estimator, template_bp=None):
        super(tdustmodel, self).__init__(freqlist,estimator,template_bp)
        self.paramlist = self.initlist()
        self.paramrange = self.initrange()
        self.paramdft = self.initdft()
        assert (self._template_flag)
        assert (self._template_nfreq == 1)

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        plist.append('beta_d')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        prange['beta_d'] = [0.,5.]
        return prange

    def bratio(self,freq,ref):
        hGk_td = 0.04799340833334541/19.6  # h*GHz/k_B/Td
        return (freq/ref)**3*(np.exp(hGk_td*ref)-1.)/(np.exp(hGk_td*freq)-1.)

    def bandpower(self):
        """dust model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_d = self.params['beta_d']
        ref = self._template_freqlist[0]
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float64)
        # estimate bandpower from templates
        ps_d = self._template_bp[ref]
        for t in range(self._estimator._ntarget):
            bp_d = ps_d[t]
            for i in range(self._nfreq):
                f_ratio_i = (self._freqlist[i]/ref)**beta_val_d*self.bratio(self._freqlist[i],ref)
                c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                dl[t,:,i,i] = bp_d * (f_ratio_i*c_ratio_i)**2
                for j in range(i+1,self._nfreq):
                    f_ratio_j = (self._freqlist[j]/ref)**beta_val_d*self.bratio(self._freqlist[j],ref)
                    c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                    dl[t,:,i,j] = bp_d * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                    dl[t,:,j,i] = dl[t,:,i,j]
        return dl


@icy
class asyncadustmodel(fgmodel):

    def __init__(self, freqlist, estimator, template_bp=None):
        super(asyncadustmodel, self).__init__(freqlist,estimator,template_bp)
        self.paramlist = self.initlist()
        self.paramrange = self.initrange()
        self.paramdft = self.initdft()
        assert (not self._template_flag)
        self._template_freqlist = [23.,353.]

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for t in self._estimator._targets:
            plist.append('A_d'+t)
            plist.append('A_s'+t)
        plist.append('alpha_d')
        plist.append('alpha_s')
        plist.append('beta_d')
        plist.append('beta_s')
        plist.append('rho')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        for t in self._estimator._targets:
            prange['A_d'+t] = [0.,1.0e+4]
            prange['A_s'+t] = [0.,1.0e+4]
        prange['alpha_d'] = [-5.,5.]
        prange['alpha_s'] = [-5.,5.]
        prange['beta_d'] = [0.,5.]
        prange['beta_s'] = [-5.,0.]
        prange['rho'] = [-0.5,0.5]
        return prange

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
        ref_s = self._template_freqlist[0]
        ref_d = self._template_freqlist[1]
        dl_stmp = self._estimator.bpconvert(np.insert((np.arange(1,self._estimator._lmax+1).astype(np.float64)/80.)**self._params['alpha_s'],[0],0.))
        dl_dtmp = self._estimator.bpconvert(np.insert((np.arange(1,self._estimator._lmax+1).astype(np.float64)/80.)**self._params['alpha_d'],[0],0.))
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float64)
        for t in range(self._estimator._ntarget):
            bp_s = self._params['A_s'+self._estimator._targets[t]]*dl_stmp
            bp_d = self._params['A_d'+self._estimator._targets[t]]*dl_dtmp
            for i in range(self._nfreq):
                f_ratio_is = (self._freqlist[i]/ref_s)**beta_val_s
                f_ratio_id = (self._freqlist[i]/ref_d)**beta_val_d*self.bratio(self._freqlist[i],ref_d)
                c_ratio_is = self.i2cmb(self._freqlist[i],ref_s)
                c_ratio_id = self.i2cmb(self._freqlist[i],ref_d)
                dl[t,:,i,i] = bp_s * (f_ratio_is*c_ratio_is)**2
                dl[t,:,i,i] += bp_d * (f_ratio_id*c_ratio_id)**2
                dl[t,:,i,i] += rho * np.sqrt(bp_s*bp_d) * (f_ratio_is*f_ratio_id*c_ratio_is*c_ratio_id)
                for j in range(i+1,self._nfreq):
                    f_ratio_js = (self._freqlist[j]/ref_s)**beta_val_s
                    f_ratio_jd = (self._freqlist[j]/ref_d)**beta_val_d*self.bratio(self._freqlist[j],ref_d)
                    c_ratio_js = self.i2cmb(self._freqlist[j],ref_s)
                    c_ratio_jd = self.i2cmb(self._freqlist[j],ref_d)
                    dl[t,:,i,j] = bp_s * (f_ratio_is*c_ratio_is*f_ratio_js*c_ratio_js)
                    dl[t,:,i,j] += bp_d * (f_ratio_id*c_ratio_id*f_ratio_jd*c_ratio_jd)
                    dl[t,:,i,j] += rho * np.sqrt(bp_s*bp_d) * ( f_ratio_is*c_ratio_is*f_ratio_jd*c_ratio_jd + f_ratio_id*c_ratio_id*f_ratio_js*c_ratio_js  )
                    dl[t,:,j,i] = dl[t,:,i,j]
        return dl


@icy
class tsynctdustmodel(fgmodel):

    def __init__(self, freqlist, estimator, template_bp=None):
        super(tsynctdustmodel, self).__init__(freqlist,estimator,template_bp)
        self.paramlist = self.initlist()
        self.paramrange = self.initrange()
        self.paramdft = self.initdft()
        assert (self._template_flag)
        assert (self._template_nfreq == 2)

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        plist.append('beta_d')
        plist.append('beta_s')
        plist.append('rho')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        prange['beta_d'] = [0.,5.]
        prange['beta_s'] = [-5.,0.]
        prange['rho'] = [-0.5,0.5]
        return prange

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
        ref_s = self._template_freqlist[0]
        ref_d = self._template_freqlist[1]
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float64)
        # estimate bandpower from templates
        ps_s = self._template_bp[ref_s]
        ps_d = self._template_bp[ref_d]
        for t in range(self._estimator._ntarget):
            bp_s = ps_s[t]
            bp_d = ps_d[t]
            bp_c = bp_s*bp_d
            for i in range(self._nfreq):
                f_ratio_is = (self._freqlist[i]/ref_s)**beta_val_s
                f_ratio_id = (self._freqlist[i]/ref_d)**beta_val_d*self.bratio(self._freqlist[i],ref_d)
                c_ratio_is = self.i2cmb(self._freqlist[i],ref_s)
                c_ratio_id = self.i2cmb(self._freqlist[i],ref_d)
                dl[t,:,i,i] = bp_s * (f_ratio_is*c_ratio_is)**2
                dl[t,:,i,i] += bp_d * (f_ratio_id*c_ratio_id)**2
                dl[t,:,i,i] += rho * np.sign(bp_c) * np.sqrt(np.fabs(bp_c)) * (f_ratio_is*f_ratio_id*c_ratio_is*c_ratio_id)
                for j in range(i+1,self._nfreq):
                    f_ratio_js = (self._freqlist[j]/ref_s)**beta_val_s
                    f_ratio_jd = (self._freqlist[j]/ref_d)**beta_val_d*self.bratio(self._freqlist[j],ref_d)
                    c_ratio_js = self.i2cmb(self._freqlist[j],ref_s)
                    c_ratio_jd = self.i2cmb(self._freqlist[j],ref_d)
                    dl[t,:,i,j] = bp_s * (f_ratio_is*c_ratio_is*f_ratio_js*c_ratio_js)
                    dl[t,:,i,j] += bp_d * (f_ratio_id*c_ratio_id*f_ratio_jd*c_ratio_jd)
                    dl[t,:,i,j] += rho * np.sign(bp_c) * np.sqrt(np.fabs(bp_c)) * ( f_ratio_is*c_ratio_is*f_ratio_jd*c_ratio_jd + f_ratio_id*c_ratio_id*f_ratio_js*c_ratio_js  )
                    dl[t,:,j,i] = dl[t,:,i,j]
        return dl
