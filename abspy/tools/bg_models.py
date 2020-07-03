"""
CMB models

- cmbmodel
    cmb band-power model
"""

import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy
from abspy.tools.ps_estimator import pstimator


@icy
class bgmodel(object):
    
    def __init__(self, freqs, nmap, mask, aposcale, psbin):
        self.freqs = freqs
        self.nmap = nmap
        self.mask = mask
        self.aposcale = aposcale
        self.psbin = psbin
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=self._aposcale,psbin=self._psbin)  # init PS estimator
        self._modes = self._est.modes[int(self._nmap==2):]  # adjust for B mode
        self._params = dict()  # base class holds empty dict
        self._params_dft = dict()

    @property
    def freqs(self):
        return self._freqs

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def nmap(self):
        return self._nmap

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

    @freqs.setter
    def freqs(self, freqs):
        assert isinstance(freqs, (list,tuple))
        self._freqs = freqs
        self._nfreq = len(self._freqs)

    @nmap.setter
    def nmap(self, nmap):
        self._nmap = nmap

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
        self._npix = mask.shape[1]
        self._nside = int(np.sqrt(self._npix//12))

    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params.update({name: pdict[name]})


@icy
class cmbmodel(bgmodel):
    
    def __init__(self, freqs, nmap, mask, aposcale, psbin):
        super(cmbmodel, self).__init__(freqs,nmap,mask,aposcale,psbin)
        # setup self.params' keys by param_list and content by param_dft
        self.reset(self.default)

    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_c_x", exponential index of amplitude
        """
        plist = list()
        if (self._nmap == 1):
            name = ['bp_c_T_']
        elif (self._nmap == 2):
            name = ['bp_c_B_']
        else:
            raise ValueError('unsupported nmap')
        for i in range(len(name)):
            for j in range(len(self._modes)):
                plist.append(name[i]+str(self._modes[j]))
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
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freq_list : float
            list of frequency in GHz
            
        freq_ref : float
            synchrotron template reference frequency
        """
        if self._nmap == 1:
            bp_t = np.ones((len(self._modes),self._nfreq,self._nfreq))
            for l in range(len(self._modes)):
                bp_t[l] *= self._params['bp_c_T_'+str(self._modes[l])]
            return bp_t
        if self._nmap == 2:
            bp_b = np.ones((len(self._modes),self._nfreq,self._nfreq))
            for l in range(len(self._modes)):
                bp_b[l] *= self._params['bp_c_B_'+str(self._modes[l])]
            return bp_b
