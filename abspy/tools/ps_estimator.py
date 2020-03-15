"""
The pseudo-PS estimation module,
by default it requires the NaMaster package.
"""
import pymaster as nmt
"""
For using other PS estimators,
please do your own estimation pipeline.
"""
import healpy as hp
import numpy as np
import logging as log
from abspy.tools.icy_decorator import icy

@icy
class pstimator(object):

    def __init__(self, nside, mask=None, aposcale=None, psbin=None):
        """
        Parameters
        ----------
        
        nside : integer
            HEALPix Nside.
            
        mask : numpy.ndarray
            A single-row array of mask map.
        
        aposcale : float
            Apodization size in deg.
            
        psbin : (positive) integer
            Number of angular modes for each PS bin.
        """
        self.nside = nside
        self.mask = mask
        self.aposcale = aposcale
        self.psbin = psbin
        
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
        
    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        self._npix = 12*self._nside**2
        
    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones((1,self._npix))
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = mask
        
    @aposcale.setter
    def aposcale(self, aposcale):
        if aposcale is None:
            self._aposcale = 1.0
        else:
            assert (aposcale > 0)
            self._aposcale = aposcale
            
    @psbin.setter
    def psbin(self, psbin):
        if psbin is None:
            self._psbin = 10
        else:
            assert isinstance(psbin, int)
            assert (psbin > 0)
            self._psbin = psbin
        
    def auto_t(self, maps):
        """
        Auto PS,
        apply NaMaster estimator to T (scalar) map with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of single T map.
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (1,self._npix))
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        _mapT = maps[0]
        # assemble NaMaster fields
        _f0 = nmt.NmtField(_apd_mask, [_mapT])
        _b = nmt.NmtBin(self._nside, nlb=self._psbin)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f0, _f0, _b)  # scalar - scalar
        return (_b.get_effective_ells(), _cl00[0])
        
    def cross_t(self, maps):
        """
        Cross PS,
        apply NaMaster estimator to T (scalar) map with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A two-row array array of two T maps.
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (2,self._npix))
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        _mapT01 = maps[0]
        _mapT02 = maps[1]
        # assemble NaMaster fields
        _f01 = nmt.NmtField(_apd_mask, [_mapT01])
        _f02 = nmt.NmtField(_apd_mask, [_mapT02])
        _b = nmt.NmtBin(self._nside, nlb=self._psbin)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f01, _f02, _b)  # scalar - scalar
        return (_b.get_effective_ells(), _cl00[0])
    
    def auto_eb(self, maps):
        """
        Auto PS,
        apply NaMaster estimator to QU (spin-2) maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A two-row array of Q, U maps,
            with polarization in CMB convention.
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (2,self._npix))
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        _mapQ = maps[0]
        _mapU = maps[1]
        # assemble NaMaster fields
        _f2 = nmt.NmtField(_apd_mask, [_mapQ, _mapU])
        _b = nmt.NmtBin(self._nside, nlb=self._psbin)
        # MASTER estimator
        _cl22 = nmt.compute_full_master(_f2, _f2, _b)  # tensor - tensor
        return (_b.get_effective_ells(), _cl22[0], _cl22[3])
        
    def cross_eb(self, maps):
        """
        Cross PS,
        apply NaMaster estimator to QU (spin-2) maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A four-row array of Q, U maps, arranged as {Q1, U1, Q2, U2},
            with polarization in CMB convention.
          
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (4,self._npix))
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        _mapQ01 = maps[0]
        _mapU01 = maps[1]
        _mapQ02 = maps[2]
        _mapU02 = maps[3]
        # assemble NaMaster fields
        _f21 = nmt.NmtField(_apd_mask, [_mapQ01, _mapU01])
        _f22 = nmt.NmtField(_apd_mask, [_mapQ02, _mapU02])
        _b = nmt.NmtBin(self._nside, nlb=self._psbin)
        # MASTER estimator
        _cl22 = nmt.compute_full_master(_f21, _f22, _b)  # tensor - tensor
        return (_b.get_effective_ells(), _cl22[0], _cl22[3])
    
    def auto_teb(self, maps):
        """
        Auto PS,
        apply NaMaster estimator to TQU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A three-row array of T, Q, U maps,
            with polarization in CMB convention.
           
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (3,self._npix))
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        _mapT = maps[0]
        _mapQ = maps[1]
        _mapU = maps[2]
        # assemble NaMaster fields
        _f0 = nmt.NmtField(_apd_mask, [_mapT])
        _f2 = nmt.NmtField(_apd_mask, [_mapQ, _mapU])
        _b = nmt.NmtBin(self._nside, nlb=self._psbin)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f0, _f0, _b)  # scalar - scalar
        _cl22 = nmt.compute_full_master(_f2, _f2, _b)  # tensor - tensor
        return (_b.get_effective_ells(), _cl00[0], _cl22[0], _cl22[3])
        
    def cross_teb(self, maps):
        """
        Cross PS,
        apply NaMaster estimator to TQU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A six-row array of T, Q, U maps, arranged as {T,Q,U,T,Q,U},
            with polarization in CMB convention.
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (6,self._npix))
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        _mapT01 = maps[0]
        _mapQ01 = maps[1]
        _mapU01 = maps[2]
        _mapT02 = maps[3]
        _mapQ02 = maps[4]
        _mapU02 = maps[5]
        # assemble NaMaster fields
        _f01 = nmt.NmtField(_apd_mask, [_mapT01])
        _f21 = nmt.NmtField(_apd_mask, [_mapQ01, _mapU01])
        _f02 = nmt.NmtField(_apd_mask, [_mapT02])
        _f22 = nmt.NmtField(_apd_mask, [_mapQ02, _mapU02])
        _b = nmt.NmtBin(self._nside, nlb=self._psbin)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f01, _f02, _b)  # scalar - scalar
        _cl22 = nmt.compute_full_master(_f21, _f22, _b)  # tensor - tensor
        return (_b.get_effective_ells(), _cl00[0], _cl22[0], _cl22[3])
