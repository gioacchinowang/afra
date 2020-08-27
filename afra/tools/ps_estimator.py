"""
The pseudo-PS estimation module,
by default it requires the NaMaster package.
"""
import pymaster as nmt
import healpy as hp
import numpy as np
from afra.tools.icy_decorator import icy


@icy
class pstimator(object):

    def __init__(self, nside, mask=None, aposcale=None, psbin=None, lmin=None, lmax=None, target='T'):
        """
        Parameters
        ----------
        
        nside : integer
            HEALPix Nside.
        
        mask : numpy.ndarray
            A single-vector array of mask map.
        
        aposcale : float
            Apodization size in deg.
        
        psbin : (positive) integer
            Number of angular modes for each PS bin.
        
        lmin : (positive) integer
            Minimal angular mode.
        
        lmax : (positive) integer
            Maximal angular mode.
        
        target : string
            Choosing among 'T', 'E' or 'B' mode.
        """
        self.nside = nside
        self.aposcale = aposcale
        self.psbin = psbin
        self.lmin = lmin
        self.lmax = lmax
        self.mask = mask
        self._b = self.bands()
        self._modes = self._b.get_effective_ells()
        self._target = target
        self._autodict = {'T':self.autoBP_T,'E':self.autoBP_E,'B':self.autoBP_B}
        self._crosdict = {'T':self.crosBP_T,'E':self.crosBP_E,'B':self.crosBP_B}

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return self._npix

    @property
    def mask(self):
        return self._mask

    @property
    def apomask(self):
        return self._apomask

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

    @property
    def target(self):
        return self._target

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
            self._mask = np.ones(self._npix,dtype=np.float32)
        else:
            assert isinstance(mask, np.ndarray)
            assert (len(mask) == self._npix)
            self._mask = mask.copy()
            self._apomask = nmt.mask_apodization(mask, self._aposcale, apotype='C2')

    @aposcale.setter
    def aposcale(self, aposcale):
        if aposcale is None:
            self._aposcale = 0.0
        else:
            assert (aposcale > 0)
            self._aposcale = aposcale

    @psbin.setter
    def psbin(self, psbin):
        if psbin is None:
            self._psbin = 1
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

    @target.setter
    def target(self, target):
        assert isinstance(target, str)
        self._target = target

    def purified_e(self, maps):
        """Get pure E mode scalar map with B2E leakage corrected.
        Has to be conducted before apodizing the mask.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            TQU maps at various frequencies,
            in shape (3, N_pix).
        
        Returns
        -------
        purified maps : numpy.ndarray
            B mode maps at various frequencies,
            in shape (N_pix).
        """
        assert (maps.shape == (3,self._npix))
        mask_sum = np.sum(self._mask)
        fill = np.arange(self._npix)[np.where(self._mask>0.)] # the pixel index of the available index
        rslt = np.zeros(self._npix,dtype=np.float32)
        # get the template of E to B leakage
        Alm0 = hp.map2alm(maps) #alms of the masked maps
        E0 = hp.alm2map(Alm0[1],nside=self._nside,verbose=0)  # corrupted E map
        Alm0[0] = 0.
        Alm0[1] = 0.
        B0 = hp.alm2map(Alm0,nside=self._nside,verbose=0)  # TQU of B mode only
        B0[:,self._mask==0.] = 0.  # re-mask
        Alm1 = hp.map2alm(B0)  # Alms of the TUQ from E-mode only
        E1 = hp.alm2map(Alm1[1],nside=self._nside,verbose=0)  # template of B2E leakage
        # compute the residual of linear fit
        x = E1[fill]
        y = E0[fill]
        mx  = np.sum(x)/mask_sum
        my  = np.sum(y)/mask_sum
        cxx = np.sum((x-mx)*(x-mx))
        cxy = np.sum((y-my)*(x-mx))
        a1  = cxy/cxx
        a0  = my - mx*a1
        rslt[fill] = y - a0 - a1*x
        return rslt

    def purified_b(self, maps):
        """Get pure E mode scalar map with E2B leakage corrected.
        Has to be conducted before apodizing the mask.
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            TQU maps at various frequencies,
            in shape (3, N_pix).
        
        Returns
        -------
        purified maps : numpy.ndarray
            B mode maps at various frequencies,
            in shape (N_pix).
        """
        assert (maps.shape == (3,self._npix))
        mask_sum = np.sum(self._mask)
        fill = np.arange(self._npix)[np.where(self._mask>0.)] # the pixel index of the available index
        rslt = np.zeros(self._npix,dtype=np.float32)
        # get the template of E to B leakage
        Alm0 = hp.map2alm(maps) #alms of the masked maps
        B0 = hp.alm2map(Alm0[2],nside=self._nside,verbose=0)  # corrupted B map
        Alm0[0] = 0.
        Alm0[2] = 0.
        E0 = hp.alm2map(Alm0,nside=self._nside,verbose=0)  # TQU of E mode only
        E0[:,self._mask==0.] = 0.  # re-mask
        Alm1 = hp.map2alm(E0)  # Alms of the TUQ from E-mode only
        B1 = hp.alm2map(Alm1[2],nside=self._nside,verbose=0)  # template of E2B leakage
        # compute the residual of linear fit
        x = B1[fill]
        y = B0[fill]
        mx  = np.sum(x)/mask_sum
        my  = np.sum(y)/mask_sum
        cxx = np.sum((x-mx)*(x-mx))
        cxy = np.sum((y-my)*(x-mx))
        a1  = cxy/cxx
        a0  = my - mx*a1
        rslt[fill] = y - a0 - a1*x
        return rslt

    def bands(self):
        """NaMaster multipole band object"""
        ells = np.arange(3*self._nside, dtype='int32')  # Array of multipoles
        weights = np.ones_like(ells)/self._psbin  # Array of weights
        bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
        i = 0
        while self._psbin * (i + 1) + self._lmin < self._lmax:
            bpws[self._psbin * i + self._lmin: self._psbin * (i+1) + self._lmin] = i
            i += 1
        return nmt.NmtBin(nside=self._nside, bpws=bpws, ells=ells, weights=weights, is_Dell=True, lmax=self._lmax)

    def autoWSP(self, maps, fwhms=None):
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (3,self._npix))
        dat = maps[0]
        # assemble NaMaster fields
        if fwhms is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # prepare workspace
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f0, f0, self._b)
        return w

    def crosWSP(self, maps, fwhms=[None,None]):
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (6,self._npix))
        assert (len(fwhms) == 2)
        dat1 = maps[0]
        dat2 = maps[3]
        # assemble NaMaster fields
        if fwhms[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # prepare workspace
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f01, f02, self._b)
        return w

    def autoBP(self, maps, wsp=None, fwhms=None):
        """
        Auto BP
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of a single map.
        
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
        
        fwhms : float
            FWHM of gaussian beams
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, XX, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (3,self._npix))
        maps[:,self._mask==0.] = 0.  # !!!
        # select among T, E and B
        return self._autodict[self._target](maps,wsp,fwhms)

    def crosBP(self, maps, wsp=None, fwhms=[None,None]):
        """
        Cross BP
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of a single map.
        
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
        
        fwhms : float
            FWHM of gaussian beams.
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, XX, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (6,self._npix))
        assert (len(fwhms) == 2)
        maps[:,self._mask==0.] = 0.  # !!!
        # select among T, E and B
        return self._crosdict[self._target](maps,wsp,fwhms)

    def autoBP_T(self, maps, wsp=None, fwhms=None):
        dat = maps[0]
        # assemble NaMaster fields
        if fwhms is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f0, f0, self._b)
            return (self._modes, cl00[0])
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])

    def crosBP_T(self, maps, wsp=None, fwhms=[None,None]):
        dat1 = maps[0]
        dat2 = maps[3]
        # assemble NaMaster fields
        if fwhms[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f01, f02, self._b)
            return (self._modes, cl00[0])
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])

    def autoBP_E(self, maps, wsp=None, fwhms=None):
        dat = self.purified_e(maps)
        # assemble NaMaster fields
        if fwhms is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f0, f0, self._b)
            return (self._modes, cl00[0])
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])

    def crosBP_E(self, maps, wsp=None, fwhms=[None,None]):
        dat1 = self.purified_e(maps[:3])
        dat2 = self.purified_e(maps[3:])
        # assemble NaMaster fields
        if fwhms[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f01, f02, self._b)
            return (self._modes, cl00[0])
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])

    def autoBP_B(self, maps, wsp=None, fwhms=None):
        dat = self.purified_b(maps)
        # assemble NaMaster fields
        if fwhms is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(fwhms, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f0, f0, self._b)
            return (self._modes, cl00[0])
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])

    def crosBP_B(self, maps, wsp=None, fwhms=[None,None]):
        dat1 = self.purified_b(maps[:3])
        dat2 = self.purified_b(maps[3:])
        # assemble NaMaster fields
        if fwhms[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(fwhms[0], 3*self._nside-1))
        if fwhms[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(fwhms[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f01, f02, self._b)
            return (self._modes, cl00[0])
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, cl00[0])
