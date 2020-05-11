"""
The pseudo-PS estimation module,
by default it requires the NaMaster package.
"""
import pymaster as nmt
"""
For using other PS estimators,
please modify the implementations below,
without touching the API.
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
        
    def auto_t(self, maps, wsp=None, fwhms=None):
        """
        Auto PS,
        apply NaMaster estimator to T (scalar) map with(out) masks.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of single T map.
            
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
            
        fwhms : float
            FWHM of gaussian beams
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (1,self._npix))
        # mask apodization
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        # assemble NaMaster fields
        if fwhms is None:
            _f0 = nmt.NmtField(_apd_mask, [maps[0]])
        else:
            _f0 = nmt.NmtField(_apd_mask, [maps[0]], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        _b = nmt.NmtBin(self._nside, nlb=self._psbin, is_Dell=True)
        # estimate PS
        if wsp is None:
            _w = nmt.NmtWorkspace()
            _w.compute_coupling_matrix(_f0, _f0, _b)
            _cl00c = nmt.compute_coupled_cell(_f0, _f0)
            _cl00 = _w.decouple_cell(_cl00c)
            return (_b.get_effective_ells(), _cl00[0], _w)
        else:
            _cl00c = nmt.compute_coupled_cell(_f0, _f0)
            _cl00 = wsp.decouple_cell(_cl00c)
            return (_b.get_effective_ells(), _cl00[0])
        
        
    def cross_t(self, maps, wsp=None, fwhms=[None,None]):
        """
        Cross PS,
        apply NaMaster estimator to T (scalar) map with(out) masks.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A two-row array array of two T maps.
            
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
            
        fwhms : list, tuple
            FWHM of gaussian beams
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, TT, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (2,self._npix))
        assert (len(fwhms) == 2)
        # mask apodization
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        # assemble NaMaster fields
        if fwhms[0] is None:
            _f01 = nmt.NmtField(_apd_mask, [maps[0]])
        else:
            _f01 = nmt.NmtField(_apd_mask, [maps[0]], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            _f02 = nmt.NmtField(_apd_mask, [maps[1]])
        else:
            _f02 = nmt.NmtField(_apd_mask, [maps[1]], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        _b = nmt.NmtBin.from_nside_linear(self._nside, nlb=self._psbin, is_Dell=True)
        # estimate PS
        if wsp is None:
            _w = nmt.NmtWorkspace()
            _w.compute_coupling_matrix(_f01, _f02, _b)
            _cl00c = nmt.compute_coupled_cell(_f01, _f02)
            _cl00 = _w.decouple_cell(_cl00c)
            return (_b.get_effective_ells(), _cl00[0], _w)
        else:
            _cl00c = nmt.compute_coupled_cell(_f01, _f02)
            _cl00 = wsp.decouple_cell(_cl00c)
            return (_b.get_effective_ells(), _cl00[0])
    
    def auto_eb(self, maps, wsp=None, fwhms=None):
        """
        Auto PS,
        apply NaMaster estimator to QU (spin-2) maps with(out) masks.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A two-row array of Q, U maps,
            with polarization in CMB convention.
            
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
            
        fwhms : float
            FWHM of gaussian beams
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, EE, BB, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (2,self._npix))
        # mask apodization
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        # assemble NaMaster fields
        if fwhms is None:
            _f2 = nmt.NmtField(_apd_mask, [maps[0], maps[1]], purify_e=False, purify_b=True)
        else:
            _f2 = nmt.NmtField(_apd_mask, [maps[0], maps[1]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        _b = nmt.NmtBin.from_nside_linear(self._nside, nlb=self._psbin, is_Dell=True)
        # estimate PS
        if wsp is None:
            _w = nmt.NmtWorkspace()
            _w.compute_coupling_matrix(_f2, _f2, _b)
            _cl22c = nmt.compute_coupled_cell(_f2, _f2)
            _cl22 = _w.decouple_cell(_cl22c)
            return (_b.get_effective_ells(), _cl22[0], _cl22[3], _w)
        else:
            _cl22c = nmt.compute_coupled_cell(_f2, _f2)
            _cl22 = wsp.decouple_cell(_cl22c)
            return (_b.get_effective_ells(), _cl22[0], _cl22[3])
        
    def cross_eb(self, maps, wsp=None, fwhms=[None,None]):
        """
        Cross PS,
        apply NaMaster estimator to QU (spin-2) maps with(out) masks.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A four-row array of Q, U maps, arranged as {Q1, U1, Q2, U2},
            with polarization in CMB convention.
            
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
            
        fwhms : list, tuple
            FWHM of gaussian beams
          
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, EE, BB, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (4,self._npix))
        assert (len(fwhms) == 2)
        # mask apodization
        _apd_mask = nmt.mask_apodization(self._mask[0], self._aposcale, apotype='Smooth')
        # assemble NaMaster fields
        if fwhms[0] is None:
            _f21 = nmt.NmtField(_apd_mask, [maps[0], maps[1]], purify_e=False, purify_b=True)
        else:
            _f21 = nmt.NmtField(_apd_mask, [maps[0], maps[1]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            _f22 = nmt.NmtField(_apd_mask, [maps[2], maps[3]], purify_e=False, purify_b=True)
        else:
            _f22 = nmt.NmtField(_apd_mask, [maps[2], maps[3]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        _b = nmt.NmtBin.from_nside_linear(self._nside, nlb=self._psbin, is_Dell=True)
        # estimate PS
        if wsp is None:
            _w = nmt.NmtWorkspace()
            _w.compute_coupling_matrix(_f21, _f22, _b)
            _cl22c = nmt.compute_coupled_cell(_f21, _f22)
            _cl22 = _w.decouple_cell(_cl22c)
            return (_b.get_effective_ells(), _cl22[0], _cl22[3], _w)
        else:
            _cl22c = nmt.compute_coupled_cell(_f21, _f22)
            _cl22 = wsp.decouple_cell(_cl22c)
            return (_b.get_effective_ells(), _cl22[0], _cl22[3])
