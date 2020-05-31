"""
CMB models

- cmbmodel
    cmb band-power model
"""

import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy


@icy
class bgmodel(object):
    
    def __init__(self, modes=list()):
        self.modes = modes
        self._params = dict()
        
    @property
    def modes(self):
        """angular mode list"""
        return self._modes
        
    @property
    def params(self):
        return self._params
        
    @modes.setter
    def modes(self, modes):
        assert isinstance(modes, list)
        self._modes = modes

    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params[name] = pdict[name]


@icy
class cmbmodel(bgmodel):
    
    def __init__(self, modes=list()):
        super(cmbmodel, self).__init__(modes)
    
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_c_x" amplitude at reference frequency x
        """
        plist = list()
        for i in range(len(self._modes)):
            plist.append('bp_c_'+str(self._modes[i]))
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
        
    def bandpower(self, freq_list):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freq_list : float
            list of frequency in GHz
            
        freq_ref : float
            synchrotron template reference frequency
        """
        assert isinstance(freq_list, (list,tuple))
        rslt = np.ones((len(self._modes),len(freq_list),len(freq_list)))
        for l in range(len(self._modes)):
            bp_c = self.params['bp_c_'+str(self._modes[l])]
            rslt[l] *= bp_c
        return rslt
