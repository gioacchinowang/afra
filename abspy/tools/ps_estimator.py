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

    def __init__(self):
        pass
        
    def auto_t(self, maps, mask=None, aposcale=None, binning=None):
        """
        Auto PS,
        apply NaMaster estimator to T (scalar) map with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of single T map.
            
        mask : numpy.ndarray
            A single-row array of mask map
            
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape[0] == 1)
        assert (mask.shape[0] == 1)
        assert (maps.shape[1] == mask.shape[1])
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask[0], aposcale, apotype='Smooth')
        _mapT = maps[0]
        # assemble NaMaster fields
        _f0 = nmt.NmtField(_apd_mask, [_mapT])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f0, _f0, _b)  # scalar - scalar
        return _b.get_effective_ells(), _cl00[0]
        
    def cross_t(self, maps, mask=None, aposcale=None, binning=None):
        """
        Cross PS,
        apply NaMaster estimator to T (scalar) map with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A two-row array array of two T maps.
            
        mask : numpy.ndarray
            A single-row array pf mask map
            
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape[0] == 2)
        assert (mask.shape[0] == 1)
        assert (maps.shape[1] == mask.shape[1])
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask[0], aposcale, apotype='Smooth')
        _mapT01 = maps[0]
        _mapT02 = maps[1]
        # assemble NaMaster fields
        _f01 = nmt.NmtField(_apd_mask, [_mapT01])
        _f02 = nmt.NmtField(_apd_mask, [_mapT02])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f01, _f02, _b)  # scalar - scalar
        return _b.get_effective_ells(), _cl00[0]
    
    def auto_eb(self, maps, mask=None, aposcale=None, binning=None):
        """
        Auto PS,
        apply NaMaster estimator to QU (spin-2) maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A two-row array of Q, U maps,
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            A single-row array of mask map
            
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape[0] == 2)
        assert (mask.shape[0] == 1)
        assert (maps.shape[1] == mask.shape[1])
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask[0], aposcale, apotype='Smooth')
        _mapQ = maps[0]
        _mapU = maps[1]
        # assemble NaMaster fields
        _f2 = nmt.NmtField(_apd_mask, [_mapQ, _mapU])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl22 = nmt.compute_full_master(_f2, _f2, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl22[0], _cl22[3]
        
    def cross_eb(self, maps, mask=None, aposcale=None, binning=None):
        """
        Cross PS,
        apply NaMaster estimator to QU (spin-2) maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A four-row array of Q, U maps, arranged as {Q1, U1, Q2, U2},
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            A single-row array of mask map
            
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape[0] == 4)
        assert (mask.shape[0] == 1)
        assert (maps.shape[1] == mask.shape[1])
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask[0], aposcale, apotype='Smooth')
        _mapQ01 = maps[0]
        _mapU01 = maps[1]
        _mapQ02 = maps[2]
        _mapU02 = maps[3]
        # assemble NaMaster fields
        _f21 = nmt.NmtField(_apd_mask, [_mapQ01, _mapU01])
        _f22 = nmt.NmtField(_apd_mask, [_mapQ02, _mapU02])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl22 = nmt.compute_full_master(_f21, _f22, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl22[0], _cl22[3]
    
    def auto_teb(self, maps, mask=None, aposcale=None, binning=None):
        """
        Auto PS,
        apply NaMaster estimator to TQU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A three-row array of T, Q, U maps,
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            A single-row array of mask map
            
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape[0] == 3)
        assert (mask.shape[0] == 1)
        assert (maps.shape[1] == mask.shape[1])
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask[0], aposcale, apotype='Smooth')
        _mapT = maps[0]
        _mapQ = maps[1]
        _mapU = maps[2]
        # assemble NaMaster fields
        _f0 = nmt.NmtField(_apd_mask, [_mapT])
        _f2 = nmt.NmtField(_apd_mask, [_mapQ, _mapU])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f0, _f0, _b)  # scalar - scalar
        _cl22 = nmt.compute_full_master(_f2, _f2, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl00[0], _cl22[0], _cl22[3]
        
    def cross_teb(self, maps, mask=None, aposcale=None, binning=None):
        """
        Cross PS,
        apply NaMaster estimator to TQU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A six-row array of T, Q, U maps, arranged as {T,Q,U,T,Q,U},
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            A single-row array of mask map
            
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT, EE, BB)
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape[0] == 6)
        assert (mask.shape[0] == 1)
        assert (maps.shape[1] == mask.shape[1])
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask[0], aposcale, apotype='Smooth')
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
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f01, _f02, _b)  # scalar - scalar
        _cl22 = nmt.compute_full_master(_f21, _f22, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl00[0], _cl22[0], _cl22[3]
