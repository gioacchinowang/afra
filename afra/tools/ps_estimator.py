"""
The pseudo-PS estimation module,
by default it requires the NaMaster package.
"""
import pymaster as nmt
import healpy as hp
import numpy as np
import logging as log
from afra.tools.icy_decorator import icy


@icy
class pstimator(object):
    """
    power-spectrum estimator using NaMaster
    """

    def __init__(self, nside, mask=None, aposcale=None, psbin=None, lmin=None, lmax=None):
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
            
        lmax : (positive) integer
            Maximal angular mode.
        """
        self.nside = nside
        self.aposcale = aposcale
        self.psbin = psbin
        self.lmin = lmin
        self.lmax = lmax
        self.mask = mask
        self._b = self.bands() 
        self._modes = self._b.get_effective_ells()
        
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
    def lmin(self):
        return self._lmin
        
    @property
    def lmax(self):
        return self._lmax

    @property
    def modes(self):
        return self._modes
        
    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        self._npix = 12*self._nside**2
        
    @mask.setter
    def mask(self, mask):
        """apply apodization during initialization"""
        if mask is None:
            self._mask = np.ones((1,self._npix),dtype=np.float64)
        else:
            assert isinstance(mask, np.ndarray)
            assert (mask.shape == (1,self._npix))
            self._mask = nmt.mask_apodization(mask[0], self._aposcale, apotype='C2').reshape(1,-1)
        
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

    @lmin.setter
    def lmin(self, lmin):
        if lmin is None:
            self._lmin = 2
        else:
            assert isinstance(lmin, int)
            assert (lmin < 3*self._nside)
            self._lmin = lmin
            
    @lmax.setter
    def lmax(self, lmax):
        if lmax is None:
            self._lmax = 2*self._nside
        else:
            assert isinstance(lmax, int)
            assert (lmax < 3*self._nside)
            self._lmax = lmax

    def bands(self):
        ells = np.arange(3*self._nside, dtype='int32')  # Array of multipoles
        weights = np.ones_like(ells)/self._psbin  # Array of weights
        bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
        i = 0
        while self._psbin * (i + 1) + self._lmin < self._lmax:
            bpws[self._psbin * i + self._lmin: self._psbin * (i+1) + self._lmin] = i
            i += 1
        return nmt.NmtBin(nside=self._nside, bpws=bpws, ells=ells, weights=weights, is_Dell=True, lmax=self._lmax)
        
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
        maps = maps.copy()
        maps[:,self._mask[0]==0.] = 0.  # !!!
        # assemble NaMaster fields
        if fwhms is None:
            f0 = nmt.NmtField(self._mask[0], [maps[0]])
        else:
            f0 = nmt.NmtField(self._mask[0], [maps[0]], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f0, f0, self._b)
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = w.decouple_cell(cl00c)
            return (self._modes, cl00[0], w)
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])
        
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
        maps = maps.copy()
        maps[:,self._mask[0]==0.] = 0.  # !!!
        # assemble NaMaster fields
        if fwhms[0] is None:
            f01 = nmt.NmtField(self._mask[0], [maps[0]])
        else:
            f01 = nmt.NmtField(self._mask[0], [maps[0]], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f02 = nmt.NmtField(self._mask[0], [maps[1]])
        else:
            f02 = nmt.NmtField(self._mask[0], [maps[1]], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f01, f02, self._b)
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = w.decouple_cell(cl00c)
            return (self._modes, cl00[0], w)
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])
    
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
        maps = maps.copy()
        maps[:,self._mask[0]==0.] = 0.  # !!!
        # assemble NaMaster fields
        if fwhms is None:
            f2 = nmt.NmtField(self._mask[0], [maps[0], maps[1]], purify_e=False, purify_b=True)
        else:
            f2 = nmt.NmtField(self._mask[0], [maps[0], maps[1]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f2, f2, self._b)
            cl22c = nmt.compute_coupled_cell(f2, f2)
            cl22 = w.decouple_cell(cl22c)
            return (self._modes, cl22[0], cl22[3], cl22[1], w)
        else:
            cl22c = nmt.compute_coupled_cell(f2, f2)
            cl22 = wsp.decouple_cell(cl22c)
            return (self._modes, cl22[0], cl22[3], cl22[1])
        
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
        maps = maps.copy()
        maps[:,self._mask[0]==0.] = 0.  # !!!
        # assemble NaMaster fields
        if fwhms[0] is None:
            f21 = nmt.NmtField(self._mask[0], [maps[0], maps[1]], purify_e=False, purify_b=True)
        else:
            f21 = nmt.NmtField(self._mask[0], [maps[0], maps[1]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f22 = nmt.NmtField(self._mask[0], [maps[2], maps[3]], purify_e=False, purify_b=True)
        else:
            f22 = nmt.NmtField(self._mask[0], [maps[2], maps[3]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f21, f22, self._b)
            cl22c = nmt.compute_coupled_cell(f21, f22)
            cl22 = w.decouple_cell(cl22c)
            return (self._modes, cl22[0], cl22[3], cl22[1], w)
        else:
            cl22c = nmt.compute_coupled_cell(f21, f22)
            cl22 = wsp.decouple_cell(cl22c)
            return (self._modes, cl22[0], cl22[3], cl22[1])

    def auto_teb(self, maps, fwhms=None):
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (3,self._npix))
        maps = maps.copy()
        maps[:,self._mask[0]==0.] = 0.  # !!!
        # assemble NaMaster fields
        if fwhms is None:
            f0 = nmt.NmtField(self._mask[0], [maps[0]])
            f2 = nmt.NmtField(self._mask[0], [maps[1], maps[2]], purify_e=False, purify_b=True)
        else:
            f0 = nmt.NmtField(self._mask[0], [maps[0]], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
            f2 = nmt.NmtField(self._mask[0], [maps[1], maps[2]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # estimate PS
        cl00 = nmt.compute_full_master(f0, f0, self._b)
        cl02 = nmt.compute_full_master(f0, f2, self._b)
        cl22 = nmt.compute_full_master(f2, f2, self._b)
        return (self._modes, cl00[0], cl02[0], cl02[1], cl22[0], cl22[1], cl22[3])
    
    def cross_teb(self, maps, wsp=None, fwhms=[None,None]):
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (6,self._npix))
        maps = maps.copy()
        maps[:,self._mask[0]==0.] = 0.  # !!!
        # assemble NaMaster fields
        if fwhms[0] is None:
            f01 = nmt.NmtField(self._mask[0], [maps[0]])
            f21 = nmt.NmtField(self._mask[0], [maps[1], maps[2]], purify_e=False, purify_b=True)
        else:
            f01 = nmt.NmtField(self._mask[0], [maps[0]], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
            f21 = nmt.NmtField(self._mask[0], [maps[1], maps[2]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f02 = nmt.NmtField(self._mask[0], [maps[3]])
            f22 = nmt.NmtField(self._mask[0], [maps[4], maps[5]], purify_e=False, purify_b=True)
        else:
            f02 = nmt.NmtField(self._mask[0], [maps[3]], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
            f22 = nmt.NmtField(self._mask[0], [maps[4], maps[5]], purify_e=False, purify_b=True, beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # estimate PS
        cl00 = nmt.compute_full_master(f01, f02, self._b)
        cl02 = nmt.compute_full_master(f01, f22, self._b)
        cl22 = nmt.compute_full_master(f21, f22, self._b)
        return (self._modes, cl00[0], cl02[0], cl02[1], cl22[0], cl22[1], cl22[3])
