"""
CMB models

- cmbmodel
    cmb band-power model
    
- cambmodel
    cmb band-power model with camb template
"""
import numpy as np
from afra.tools.icy_decorator import icy
from afra.tools.ps_estimator import pstimator
from afra.tools.aux import bp_window


@icy
class bgmodel(object):

    def __init__(self, freqlist, targets, mask, aposcale, psbin, lmin=None, lmax=None):
        self.freqlist = freqlist
        self.targets = targets
        self.mask = mask
        self.aposcale = aposcale
        self.psbin = psbin
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=self._aposcale,psbin=self._psbin,lmin=lmin,lmax=lmax,targets=self._targets)  # init PS estimator
        self._modes = self._est.modes
        self._params = dict()  # base class holds empty dict
        self._params_dft = dict()
        self._templates = None  # template PS from camb

    @property
    def freqlist(self):
        return self._freqlist

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def targets(self):
        return self._targets

    @property
    def ntarget(self):
        return self._ntarget

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

    @freqlist.setter
    def freqlist(self, freqlist):
        assert isinstance(freqlist, (list,tuple))
        self._freqlist = freqlist
        self._nfreq = len(self._freqlist)

    @targets.setter
    def targets(self, targets):
        assert isinstance(targets, str)
        self._targets = targets
        self._ntarget = len(targets)

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

    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params.update({name: pdict[name]})


@icy
class cmbmodel(bgmodel):
    
    def __init__(self, freqlist, targets, mask, aposcale, psbin, lmin=None, lmax=None):
        super(cmbmodel, self).__init__(freqlist,targets,mask,aposcale,psbin,lmin,lmax)
        # setup self.params' keys by param_list and content by param_dft
        self.reset(self.default)

    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_c_x", exponential index of amplitude
        """
        plist = list()
        prefix = list()
        for t in self._targets:
            prefix.append('bp_c_'+t+'_')
        for i in prefix:
            for j in range(len(self._modes)):
                plist.append(i+str(self._modes[j]))
        return plist
        
    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        for i in _tmp:
            prange[i] = [0.,1.e+4]
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
        """cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freq_list : float
            list of frequency in GHz
            
        freq_ref : float
            synchrotron template reference frequency
        """
        bp = np.ones((self._ntarget,len(self._modes),self._nfreq,self._nfreq),dtype=np.float32)
        for t in range(self._ntarget):
            for l in range(len(self._modes)):
                bp[t,l] *= self._params['bp_c_'+self._targets[t]+'_'+str(self._modes[l])]
        return bp


@icy
class cambmodel(bgmodel):
    """cmb model by camb"""

    def __init__(self, freqlist, targets, mask, aposcale, psbin, lmin=None, lmax=None):
        super(cambmodel, self).__init__(freqlist,targets,mask,aposcale,psbin,lmin,lmax)
        # setup self.params' keys by param_list and content by param_dft
        self.reset(self.default)
        # calculate camb template CMB PS with default parameters
        import camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=0.06)
        pars.InitPower.set_params(As=2e-9,ns=0.965,r=0.05)
        pars.set_for_lmax(4000,lens_potential_accuracy=1)
        pars.WantTensors = True
        results = camb.get_results(pars)
        self._templates = results.get_cmb_power_spectra(pars,CMB_unit='muK',raw_cl=True)

    @property
    def param_list(self):
        """parameters are set as
        - "r", tensor-to-scalar ratio
        """
        return ['r','Lb']

    @property
    def default(self):
        """register default parameter values
        """
        return {'r': 0.05, 'Lb': 1.}

    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        prange['r'] = [0.,1.]
        prange['Lb'] = [0.,2.]
        return prange

    def bandpower(self):
        """cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        """
        enum = {'T':[0],'E':[1],'B':[2],'EB':[1,2],'TEB':[0,1,2]}
        fiducial_cl = np.transpose(self._templates['lensed_scalar']*[1.,1.,self._params['Lb'],1.])[enum[self._targets],:3*self._nside] + np.transpose(self._templates['tensor']*[1.,1.,self._params['r']/0.05,1.])[enum[self._targets],:3*self._nside]
        # fiducial_cl in shape (ntarget,3*nside)
        fiducial_bp = np.empty((self._ntarget,len(self._est.modes)),dtype=np.float32)
        for t in range(self._ntarget):
            fiducial_bp[t] = bp_window(self._est).dot(fiducial_cl[t])
        bp_out = np.ones((self._ntarget,len(self._modes),self._nfreq,self._nfreq),dtype=np.float32)
        for t in range(self._ntarget):
            for l in range(len(self._modes)):
                bp_out[t,l] *= fiducial_bp[t,l]
        return bp_out
