"""
CMB models

- ncmbmodel
    cmb band-power model
    
- acmbmodel
    cmb band-power model with camb template
"""
import numpy as np
from afra.tools.icy_decorator import icy
from afra.tools.ps_estimator import pstimator


@icy
class bgmodel(object):

    def __init__(self, freqlist, estimator):
        self.freqlist = freqlist
        self.estimator = estimator
        self._params = dict()  # base class holds empty dict
        self._paramlist = list()
        self._blacklist = list()  # fixed parameter list
        self._paramdft = dict()
        self._paramrange = dict()
        self._template_ps = None  # template PS from camb

    @property
    def freqlist(self):
        return self._freqlist

    @property
    def estimator(self):
        return self._estimator

    @property
    def nfreq(self):
        return self._nfreq

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

    @property
    def template_ps(self):
        return self._template_ps

    @property
    def est(self):
        return self._est

    @freqlist.setter
    def freqlist(self, freqlist):
        assert isinstance(freqlist, (list,tuple))
        self._freqlist = freqlist
        self._nfreq = len(self._freqlist)

    @blacklist.setter
    def blacklist(self, blacklist):
        assert isinstance(blacklist, list)
        for p in blacklist:
            assert (p in self._paramlist)
        self._blacklist = blacklist

    @estimator.setter
    def estimator(self, estimator):
        assert isinstance(estimator, pstimator)
        self._estimator = estimator

    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self._paramlist:
                self._params.update({name: pdict[name]})


@icy
class ncmbmodel(bgmodel):

    def __init__(self, freqlist, estimator): 
        super(ncmbmodel, self).__init__(freqlist,estimator)
        self._paramlist = self.initlist()
        self._paramrange = self.initrange()
        self._paramdft = self.initdft()

    def initlist(self):
        """parameters are set as
        - bandpower "bp_c_x", exponential index of amplitude
        """
        plist = list()
        for t in self._estimator._targets:
            for j in range(len(self._estimator._modes)):
                plist.append('bp_c_'+t+'_'+'{:.2f}'.format(self._estimator._modes[j]))
        return plist

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        for i in self._paramlist:
            prange[i] = [0.,1.0e+4]
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
        """cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
         
        freq_list : float
            list of frequency in GHz
            
        freq_ref : float
            synchrotron template reference frequency
        """
        fiducial_bp = np.zeros((self._estimator._ntarget,self._estimator._nmode),dtype=np.float32)
        for t in range(self._estimator._ntarget):
            for l in range(self._estimator._nmode):
                    fiducial_bp[t,l] = self._params['bp_c_'+self._estimator._targets[t]+'_'+'{:.2f}'.format(self._estimator._modes[l])]
        fiducial_bp = self._estimator.filtrans(fiducial_bp)
        bp_out = np.ones((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float32)
        for t in range(self._estimator._ntarget):
            for l in range(self._estimator._nmode):
                bp_out[t,l] *= fiducial_bp[t,l]
        return bp_out


@icy
class acmbmodel(bgmodel):
    """cmb model by camb"""

    def __init__(self, freqlist, estimator):
        super(acmbmodel, self).__init__(freqlist,estimator)
        self._paramlist = self.initlist()
        self._paramrange = self.initrange()
        self._paramdft = self.initdft()
        if not ('E' in self._estimator.targets):
            self._blacklist.append('Ae')
        if not ('B' in self._estimator.targets):
            self._blacklist.append('r')
            self._blacklist.append('Al')
        # calculate camb template CMB PS with default parameters
        import camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=0.06)
        pars.InitPower.set_params(As=2e-9,ns=0.965,r=0.05)
        pars.set_for_lmax(max(4000,self._estimator._lmax),lens_potential_accuracy=1)
        pars.WantTensors = True
        results = camb.get_results(pars)
        self._template_ps = results.get_cmb_power_spectra(pars,CMB_unit='muK',raw_cl=True)

    def initlist(self):
        return ['r','Al','Ae']

    def initrange(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        prange['r'] = [0.,1.]
        prange['Al'] = [0.,2.]
        prange['Ae'] = [0.,2.]
        return prange

    def initdft(self):
        """register default parameter values
        """
        pdft = {'r': 0.05, 'Al': 1., 'Ae': 1.}
        self.reset(pdft)
        return pdft

    def bandpower(self):
        """cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        enum = {'T':[0],'E':[1],'B':[2],'EB':[1,2],'TEB':[0,1,2]}
        fiducial_cl = np.transpose(self._template_ps['lensed_scalar']*[1.,self._params['Ae'],self._params['Al'],1.])[enum[self._estimator._targets],self._estimator._lmin:self._estimator._lmax+1] + np.transpose(self._template_ps['tensor']*[1.,self._params['Ae'],self._params['r']/0.05,1.])[enum[self._estimator._targets],self._estimator._lmin:self._estimator._lmax+1]
        # fiducial_cl in shape (ntarget,lmax-lmin)
        fiducial_bp = np.zeros((self._estimator._ntarget,self._estimator._nmode),dtype=np.float32)
        for t in range(self._estimator._ntarget):
            fiducial_bp[t] = self._estimator.bpconvert(fiducial_cl[t])
        fiducial_bp = self._estimator.filtrans(fiducial_bp)
        bp_out = np.ones((self._estimator._ntarget,self._estimator._nmode,self._nfreq,self._nfreq),dtype=np.float32)
        for t in range(self._estimator._ntarget):
            for l in range(self._estimator._nmode):
                bp_out[t,l] *= fiducial_bp[t,l]
        return bp_out
