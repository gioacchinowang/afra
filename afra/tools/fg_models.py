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

    def __init__(self, freqlist, estimator, template_ps=None):
        self.freqlist = freqlist
        self.estimator = estimator
        self.template_ps = template_ps
        self._params = dict()  # base class holds empty dict
        self._paramlist = list()
        self._blacklist = list()
        self._paramrange = dict()
        self._paramdft = dict()

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
    def template_ps(self):
        return self._template_ps

    @property
    def template_freqlist(self):
        return self._template_freqlist

    @property
    def template_nfreq(self):
        return self._template_nfreq

    @property
    def params(self):
        return self._params

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
    def paramrange(self):
        return self._paramrange

    @freqlist.setter
    def freqlist(self, freqlist):
        assert isinstance(freqlist, (list,tuple))
        self._freqlist = freqlist
        self._nfreq = len(self._freqlist)

    @estimator.setter
    def estimator(self, estimator):
        assert isinstance(estimator, pstimator)
        self._estimator = estimator

    @template_ps.setter
    def template_ps(self, template_ps):
        """template maps at two frequency bands"""
        if template_ps is not None:
            assert isinstance(template_ps, dict)
            self._template_freqlist = sorted(template_ps.keys())
            self._template_nfreq = len(template_ps)
            assert (self._template_nfreq < 3)
            assert (template_ps[next(iter(template_ps))].shape == (self._estimator._ntarget,self._estimator._nmode))
            self._template_flag = True
            self._template_ps = template_ps
        else:
            self._template_flag = False
            self._template_ps = None
            self._template_freqlist = list()

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
class syncmodel(fgmodel):
    
    def __init__(self, freqlist, estimator, template_ps=None):
        super(syncmodel, self).__init__(freqlist,estimator,template_ps)
        self._paramlist = self.initlist()
        self._paramrange = self.initrange()
        self._paramdft = self.initdft()
        # calculate template PS
        if self._template_flag:
            assert (self._template_nfreq == 1)
        else:
            self._template_freqlist = [23]

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        if not self._template_flag:
            for t in self._estimator._targets:
                plist.append('A_s_'+t)
                plist.append('alpha_s_'+t)
        plist.append('beta_s')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        if not self._template_flag:
            for t in self._estimator._targets:
                prange['A_s_'+t] = [0.,1.0e+4]
                prange['alpha_s_'+t] = [-5.,5.]
        prange['beta_s'] = [-5.,0]
        return prange

    def initdft(self):
        """register default parameter values
        """
        pdft = dict()
        for key in self._paramrange.keys():
            pdft[key] = 0.5*(self._paramrange[key][0] + self._paramrange[key][1])
        self.reset(pdft)
        return pdft

    def bandpower(self):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        beta_val_s = self.params['beta_s']
        ref = self._template_freqlist[0]
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float32)
        # estimate bandpower from templates
        if self._template_flag:
            ps_s = self._template_ps[ref]
            for t in range(self._estimator._ntarget):
                for l in range(self._estimator._nmode):
                    bp_s = ps_s[t,l]
                    for i in range(self._nfreq):
                        f_ratio_i = (self._freqlist[i]/ref)**beta_val_s
                        c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                        dl[t,l,i,i] = bp_s * (f_ratio_i*c_ratio_i)**2
                        for j in range(i+1,self._nfreq):
                            f_ratio_j = (self._freqlist[j]/ref)**beta_val_s
                            c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                            dl[t,l,i,j] = bp_s * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                            dl[t,l,j,i] = dl[t,l,i,j]
        else:
            for t in range(self._estimator._ntarget):
                for l in range(self._estimator._nmode):
                    bp_s = self._params['A_s_'+self._estimator._targets[t]]*(np.array(self._estimator._modes[l])/80.)**self._params['alpha_s_'+self._estimator._targets[t]]
                    for i in range(self._nfreq):
                        f_ratio_i = (self._freqlist[i]/ref)**beta_val_s
                        c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                        dl[t,l,i,i] = bp_s * (f_ratio_i*c_ratio_i)**2
                        for j in range(i+1,self._nfreq):
                            f_ratio_j = (self._freqlist[j]/ref)**beta_val_s
                            c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                            dl[t,l,i,j] = bp_s * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                            dl[t,l,j,i] = dl[t,l,i,j]
        return dl


@icy
class dustmodel(fgmodel):
   
    def __init__(self, freqlist, estimator, template_ps=None):
        super(dustmodel, self).__init__(freqlist,estimator,template_ps)
        self._paramlist = self.initlist()
        self._paramrange = self.initrange()
        self._paramdft = self.initdft()
        # calculate template PS
        if self._template_flag:
            assert (self._template_nfreq == 1)
        else:
            self._template_freqlist = [353.]

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        if not self._template_flag:
            for t in self._estimator._targets:
                plist.append('A_d_'+t)
                plist.append('alpha_d_'+t)
        plist.append('beta_d')
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        if not self._template_flag:
            for t in self._estimator._targets:
                prange['A_d_'+t] = [0.,1.0e+4]
                prange['alpha_d_'+t] = [-5.,5.]
        prange['beta_d'] = [0.,5.]
        return prange

    def initdft(self):
        """register default parameter values
        """
        pdft = dict()
        for key in self._paramrange.keys():
            pdft[key] = 0.5*(self._paramrange[key][0] + self._paramrange[key][1])
        self.reset(pdft)
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
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float32)
        # estimate bandpower from templates
        if self._template_flag:
            ps_d = self._template_ps[ref]
            for t in range(self._estimator._ntarget):
                for l in range(self._estimator._nmode):
                    bp_d = ps_d[t,l]
                    for i in range(self._nfreq):
                        f_ratio_i = (self._freqlist[i]/ref)**beta_val_d*self.bratio(self._freqlist[i],ref)
                        c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                        dl[t,l,i,i] = bp_d * (f_ratio_i*c_ratio_i)**2
                        for j in range(i+1,self._nfreq):
                            f_ratio_j = (self._freqlist[j]/ref)**beta_val_d*self.bratio(self._freqlist[j],ref)
                            c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                            dl[t,l,i,j] = bp_d * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                            dl[t,l,j,i] = dl[t,l,i,j]
        else:
            for t in range(self._estimator._ntarget):
                for l in range(self._estimator._nmode):
                    bp_d = self._params['A_d_'+self._estimator._targets[t]]*(np.array(self._estimator._modes[l])/80.)**self._params['alpha_d_'+self._estimator._targets[t]]
                    for i in range(self._nfreq):
                        f_ratio_i = (self._freqlist[i]/ref)**beta_val_d*self.bratio(self._freqlist[i],ref)
                        c_ratio_i = self.i2cmb(self._freqlist[i],ref)
                        dl[t,l,i,i] = bp_d * (f_ratio_i*c_ratio_i)**2
                        for j in range(i+1,self._nfreq):
                            f_ratio_j = (self._freqlist[j]/ref)**beta_val_d*self.bratio(self._freqlist[j],ref)
                            c_ratio_j = self.i2cmb(self._freqlist[j],ref)
                            dl[t,l,i,j] = bp_d * (f_ratio_i*f_ratio_j*c_ratio_i*c_ratio_j)
                            dl[t,l,j,i] = dl[t,l,i,j]
        return dl


@icy
class syncdustmodel(fgmodel):

    def __init__(self, freqlist, estimator, template_ps=None):
        super(syncdustmodel, self).__init__(freqlist,estimator,template_ps)
        self._paramlist = self.initlist()
        self._paramrange = self.initrange()
        self._paramdft = self.initdft()
        # calculate template PS
        if self._template_flag:
            assert (self._template_nfreq == 2)
        else:
            self._template_freqlist = [23.,353.]

    def initlist(self):
        """parameters are set as
        - bandpower "bp_s_x", exponential index of amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        if not self._template_flag:
            for t in self._estimator._targets:
                plist.append('A_d_'+t)
                plist.append('A_s_'+t)
                plist.append('alpha_d_'+t)
                plist.append('alpha_s_'+t)
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
        if not self._template_flag:
            for t in self._estimator._targets:
                prange['A_d_'+t] = [0.,1.0e+4]
                prange['A_s_'+t] = [0.,1.0e+4]
                prange['alpha_d_'+t] = [-5.,5.]
                prange['alpha_s_'+t] = [-5.,5.]
        prange['beta_d'] = [0.,5.]
        prange['beta_s'] = [-5.,0.]
        prange['rho'] = [-0.5,0.5]
        return prange

    def initdft(self):
        """register default parameter values
        """
        pdft = dict()
        for key in self._paramrange.keys():
            pdft[key] = 0.5*(self._paramrange[key][0] + self._paramrange[key][1])
        self.reset(pdft)
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
        ref_s = self._template_freqlist[0]
        ref_d = self._template_freqlist[1]
        dl = np.zeros((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float32)
        # estimate bandpower from templates
        if self._template_flag:
            ps_s = self._template_ps[ref_s]
            ps_d = self._template_ps[ref_d]
            for t in range(self._estimator._ntarget):
                for l in range(self._estimator._nmode):
                    bp_s = ps_s[t,l]
                    bp_d = ps_d[t,l]
                    bp_c = bp_s*bp_d
                    for i in range(self._nfreq):
                        f_ratio_is = (self._freqlist[i]/ref_s)**beta_val_s
                        f_ratio_id = (self._freqlist[i]/ref_d)**beta_val_d*self.bratio(self._freqlist[i],ref_d)
                        c_ratio_is = self.i2cmb(self._freqlist[i],ref_s)
                        c_ratio_id = self.i2cmb(self._freqlist[i],ref_d)
                        dl[t,l,i,i] = bp_s * (f_ratio_is*c_ratio_is)**2
                        dl[t,l,i,i] += bp_d * (f_ratio_id*c_ratio_id)**2
                        dl[t,l,i,i] += rho * np.sign(bp_c) * np.sqrt(np.fabs(bp_c)) * (f_ratio_is*f_ratio_id*c_ratio_is*c_ratio_id)
                        for j in range(i+1,self._nfreq):
                            f_ratio_js = (self._freqlist[j]/ref_s)**beta_val_s
                            f_ratio_jd = (self._freqlist[j]/ref_d)**beta_val_d*self.bratio(self._freqlist[j],ref_d)
                            c_ratio_js = self.i2cmb(self._freqlist[j],ref_s)
                            c_ratio_jd = self.i2cmb(self._freqlist[j],ref_d)
                            dl[t,l,i,j] = bp_s * (f_ratio_is*c_ratio_is*f_ratio_js*c_ratio_js)
                            dl[t,l,i,j] += bp_d * (f_ratio_id*c_ratio_id*f_ratio_jd*c_ratio_jd)
                            dl[t,l,i,j] += rho * np.sign(bp_c) * np.sqrt(np.fabs(bp_c)) * ( f_ratio_is*c_ratio_is*f_ratio_jd*c_ratio_jd + f_ratio_id*c_ratio_id*f_ratio_js*c_ratio_js  )
                            dl[t,l,j,i] = dl[t,l,i,j]
        else:
            for t in range(self._estimator._ntarget):
                for l in range(self._estimator._nmode):
                    bp_s = self._params['A_s_'+self._estimator._targets[t]]*(np.array(self._estimator._modes[l])/80.)**self._params['alpha_s_'+self._estimator._targets[t]]
                    bp_d = self._params['A_d_'+self._estimator._targets[t]]*(np.array(self._estimator._modes[l])/80.)**self._params['alpha_d_'+self._estimator._targets[t]]
                    for i in range(self._nfreq):
                        f_ratio_is = (self._freqlist[i]/ref_s)**beta_val_s
                        f_ratio_id = (self._freqlist[i]/ref_d)**beta_val_d*self.bratio(self._freqlist[i],ref_d)
                        c_ratio_is = self.i2cmb(self._freqlist[i],ref_s)
                        c_ratio_id = self.i2cmb(self._freqlist[i],ref_d)
                        dl[t,l,i,i] = bp_s * (f_ratio_is*c_ratio_is)**2
                        dl[t,l,i,i] += bp_d * (f_ratio_id*c_ratio_id)**2
                        dl[t,l,i,i] += rho * np.sqrt(bp_s*bp_d) * (f_ratio_is*f_ratio_id*c_ratio_is*c_ratio_id)
                        for j in range(i+1,self._nfreq):
                            f_ratio_js = (self._freqlist[j]/ref_s)**beta_val_s
                            f_ratio_jd = (self._freqlist[j]/ref_d)**beta_val_d*self.bratio(self._freqlist[j],ref_d)
                            c_ratio_js = self.i2cmb(self._freqlist[j],ref_s)
                            c_ratio_jd = self.i2cmb(self._freqlist[j],ref_d)
                            dl[t,l,i,j] = bp_s * (f_ratio_is*c_ratio_is*f_ratio_js*c_ratio_js)
                            dl[t,l,i,j] += bp_d * (f_ratio_id*c_ratio_id*f_ratio_jd*c_ratio_jd)
                            dl[t,l,i,j] += rho * np.sqrt(bp_s*bp_d) * ( f_ratio_is*c_ratio_is*f_ratio_jd*c_ratio_jd + f_ratio_id*c_ratio_id*f_ratio_js*c_ratio_js  )
                            dl[t,l,j,i] = dl[t,l,i,j]
        return dl
