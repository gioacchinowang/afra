"""
The ABS noise resampling class.

Author:
- Jian Yao (SJTU)
- Jiaxin Wang (SJTU) jiaixn.wang@sjtu.eud.cn
"""

import logging as log
import numpy as np
from copy import deepcopy
from abspy.tools.icy_decorator import icy

@icy
class nresamp(object):
    
    def __init__(self, nmap, nside, variance, mask=None):
        log.debug('@ nresamp::__init__')
        self.nmap = nmap
        self.nside = nside
        self.variance = variance
        self.mask = mask
        self.sampsize = 100
        
    @property
    def nmap(self):
        return self._nmap
        
    @property
    def nside(self):
        return self._nside
        
    @property
    def variance(self):
        return self._variance
        
    @property
    def mask(self):
        return self._mask
        
    @nmap.setter
    def nmap(self, nmap):
        assert isinstance(nmap, int)
        assert (nmap > 0)
        self._nmap = nmap
        log.debug('number of maps'+str(self._nmap))
        
    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        self._npix = 12*nside**2
        log.debug('HEALPix Nside'+str(self._nside))
        
    @variance.setter
    def variance(self, variance):
        if variance is not None:
            assert isinstance(variance, np.ndarray)
            assert (variance.shape == (self._nfreq,self._nmap,self._npix))
            self._noise_flag = True
        else:
            self._noise_flag = False
        self._variance = variance
        log.debug('variance maps loaded')
        
    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones((1,self._npix),dtype=bool)
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = mask
        log.debug('mask map loaded')
        
    
        
    
