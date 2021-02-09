import numpy as np
import healpy as hp
import pymaster as nmt
from afra.tools.icy_decorator import icy


@icy
class pstimator(object):

    def __init__(self, nside, mask=None, aposcale=None, psbin=None, lmin=None, lmax=None, lbin=None, lcut=None, targets='T', filt=None):
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

        lbin : (positive) integer
            Angular mode bin size for NaMaster calculation.

        lcut : (positive) integer
            Ignoring the first and last number of bins from NaMaster results.
        
        targets : string
            Choosing among 'T', 'E', 'B', 'EB', 'TEB' mode.

        filt : dict
            filtering effect in BP, recording extra mixing/rescaling of BP
        """
        self.nside = nside
        self.aposcale = aposcale
        self.lbin = None
        self.lcut = None
        self.lmin = lmin
        self.lmax = lmax
        self.mask = mask
        self.b = None
        self.psbin = psbin
        self.targets = targets
        self.filt = filt
        self._autodict = {'T':self.autoBP_T,'E':self.autoBP_E,'B':self.autoBP_B,'EB':self.autoBP_EB,'TEB':self.autoBP_TEB}
        self._crosdict = {'T':self.crosBP_T,'E':self.crosBP_E,'B':self.crosBP_B,'EB':self.crosBP_EB,'TEB':self.autoBP_TEB}

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
    def b(self):
        return self._b

    @property
    def aposcale(self):
        return self._aposcale

    @property
    def nmode(self):
        return self._nmode

    @property
    def lbin(self):
        return self._lbin

    @property
    def lcut(self):
        return self._lcut

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
    def targets(self):
        return self._targets

    @property
    def ntarget(self):
        return self._ntarget

    @property
    def filt(self):
        return self._filt

    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        self._npix = 12*self._nside**2

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones(self._npix,dtype=np.float64)
            self._apomask = np.ones(self._npix,dtype=np.float64)
        else:
            assert isinstance(mask, np.ndarray)
            assert (len(mask) == self._npix)
            self._mask = mask.copy()
            self._apomask = nmt.mask_apodization(self._mask, self._aposcale, apotype='C2')

    @aposcale.setter
    def aposcale(self, aposcale):
        if aposcale is None:
            self._aposcale = 0.0
        else:
            assert (aposcale > 0)
            self._aposcale = aposcale

    @lbin.setter
    def lbin(self, lbin):
        if lbin is None:
            self._lbin = 5
        else:
            assert isinstance(lbin, int)
            assert (lbin > 0)
            self._lbin = lbin

    @lcut.setter
    def lcut(self, lcut):
        if lcut is None:
            self._lcut = 5
        else:
            assert isinstance(lcut, int)
            assert (lcut > 0)
            self._lcut = lcut

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
            self._lmax = self._nside
        else:
            assert isinstance(lmax, (int,np.int64))
            assert (lmax < 3*self._nside)
            self._lmax = lmax

    @b.setter
    def b(self, b):
        if b is None:
            """customize NaMaster multipole band object"""
            ell_ini = np.arange((self._lmax-self._lmin)//self._lbin)*self._lbin + self._lmin
            ell_end = ell_ini + self._lbin
            self._b = nmt.NmtBin.from_edges(ell_ini, ell_end, is_Dell=True)
            self.lmax = self._b.lmax  # correct lmax
        else:
            self._b = b

    @psbin.setter
    def psbin(self, psbin):
        if psbin is None:
            self._psbin = 1
            self.modes = self.rebinning(self._b.get_effective_ells())
        else:
            assert isinstance(psbin, int)
            assert (psbin > 0 and psbin < (self._lmax-self._lmin)//self._lbin - 2*self._lcut)
            self._psbin = psbin
            self.modes = self.rebinning(self._b.get_effective_ells())

    @modes.setter
    def modes(self, modes):
        assert isinstance(modes, (list,tuple,np.ndarray))
        self._modes = modes
        self._nmode = len(modes)

    @targets.setter
    def targets(self, targets):
        assert isinstance(targets, str)
        self._targets = targets
        self._ntarget = len(targets)

    @filt.setter
    def filt(self, filt):
        if filt is not None:
            assert isinstance(filt, dict)
            assert (self._targets in filt)
        self._filt = filt

    def rebinning(self, bp):
        bp_trim = bp[self._lcut:-self._lcut]
        bbp = np.empty(self._psbin, dtype=np.float64)
        idx_ini = np.arange(self._psbin)*(len(bp_trim)//self._psbin)
        idx_end = idx_ini + len(bp_trim)//self._psbin
        for i in range(self._psbin):
            bbp[i] = np.mean(bp_trim[idx_ini[i]:idx_end[i]])
        return bbp

    def bpconvert(self, ps):
        """
        "top-hat" window function matrix
        for converting PS into band-powers

        Parameters
        ----------
            input power-spectrum in multipole range (lmin:lmax+1)
        
        Return
        ----------
            band-power converting matrix in shape (# eff-ell)
        """
        raw_conv = self._b.bin_cell(ps)
        if (raw_conv.ndim == 1):
            return self.rebinning(raw_conv)
        else:
            fine_conv = np.empty((raw_conv.shape[0],self._psbin),dtype=np.float64)
            for i in range(fine_conv.shape[0]):
                fine_conv[i] = self.rebinning(raw_conv[i])
            return fine_conv

    def filtrans(self, bp):
        """
        apply filtering effect on band-powers.

        Parameters
        ----------

        bp : numpy.ndarray
            band-power in shape (# targets, # modes).

        Returns
        -------
            filtered band-power in shape (# targets, # modes).
        """
        if self._filt is None:
            return bp
        assert (bp.shape == (self._ntarget,self._nmode))
        transmat = self._filt[self._targets]
        assert (transmat.shape[0] == (bp.shape[0]*bp.shape[1]))
        return (transmat.dot(bp.reshape(-1,1))).reshape(self._ntarget,-1)

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
        rslt = np.zeros(self._npix,dtype=np.float64)
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
        a1  = np.nan_to_num(cxy/cxx)
        a0  = my - mx*a1
        rslt[fill] = y - a0 - a1*x
        return rslt

    def purified_b(self, maps):
        """Get pure E mode scalar map with E2B leakage corrected.
        Has to be conducted before apodizing the mask.
        (https://arxiv.org/abs/1811.04691)
        
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
        rslt = np.zeros(self._npix,dtype=np.float64)
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
        a1  = np.nan_to_num(cxy/cxx)
        a0  = my - mx*a1
        rslt[fill] = y - a0 - a1*x
        return rslt

    def purified_eb(self, maps):
        """Get pure EB mode scalar map with B2E & E2B leakage corrected.
        Has to be conducted before apodizing the mask.
        Parameters
        ----------
        maps : numpy.ndarray
            TQU maps at various frequencies,
            in shape (3, N_pix).
        Returns
        -------
        purified maps : numpy.ndarray
            EB mode maps at various frequencies,
            in shape (2, N_pix).
        """
        assert (maps.shape == (3,self._npix))
        mask_sum = np.sum(self._mask)
        fill = np.arange(self._npix)[np.where(self._mask>0.)] # the pixel index of the available index
        rslt = np.zeros((2,self._npix),dtype=np.float64)
        # get the template of E to B leakage
        Alm0 = hp.map2alm(maps) #alms of the masked maps
        E0 = hp.alm2map(Alm0[1],nside=self._nside,verbose=0)  # corrupted E map
        B0 = hp.alm2map(Alm0[2],nside=self._nside,verbose=0)  # corrupted B map
        Alm0_B = Alm0.copy()
        Alm0_E = Alm0.copy()
        Alm0_B[0] = 0.
        Alm0_B[1] = 0.
        Alm0_E[0] = 0.
        Alm0_E[2] = 0.
        B0p = hp.alm2map(Alm0_B,nside=self._nside,verbose=0)  # TQU of B mode only
        E0p = hp.alm2map(Alm0_E,nside=self._nside,verbose=0)  # TQU of E mode only
        B0p[:,self._mask==0.] = 0.  # re-mask
        E0p[:,self._mask==0.] = 0.
        Alm1_E = hp.map2alm(E0p)  # Alms of the TUQ from E-mode only
        Alm1_B = hp.map2alm(B0p)  # Alms of the TQU from B-mode only
        E1 = hp.alm2map(Alm1_B[1],nside=self._nside,verbose=0)  # template of B2E leakage
        B1 = hp.alm2map(Alm1_E[2],nside=self._nside,verbose=0)  # template of E2B leakage
        # compute the residual of linear fit (E mode)
        x = E1[fill]
        y = E0[fill]
        mx  = np.sum(x)/mask_sum
        my  = np.sum(y)/mask_sum
        cxx = np.sum((x-mx)*(x-mx))
        cxy = np.sum((y-my)*(x-mx))
        a1  = np.nan_to_num(cxy/cxx)
        a0  = my - mx*a1
        rslt[0,fill] = y - a0 - a1*x
        # (B mode)
        x = B1[fill]
        y = B0[fill]
        mx  = np.sum(x)/mask_sum
        my  = np.sum(y)/mask_sum
        cxx = np.sum((x-mx)*(x-mx))
        cxy = np.sum((y-my)*(x-mx))
        a1  = np.nan_to_num(cxy/cxx)
        a0  = my - mx*a1
        rslt[1,fill] = y - a0 - a1*x
        return rslt

    def autoWSP(self, maps, beams=None):
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (3,self._npix))
        dat = maps[0]
        # assemble NaMaster fields
        if beams is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(beams, 3*self._nside-1))
        # prepare workspace
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f0, f0, self._b)
        return w

    def crosWSP(self, maps, beams=[None,None]):
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (6,self._npix))
        assert (len(beams) == 2)
        dat1 = maps[0]
        dat2 = maps[3]
        # assemble NaMaster fields
        if beams[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(beams[0], 3*self._nside-1))
        if beams[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(beams[1], 3*self._nside-1))
        # prepare workspace
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f01, f02, self._b)
        return w

    def autoBP(self, maps, wsp=None, beams=None):
        """
        Auto BP
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of a single map.
        
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
        
        beams : float
            FWHM of gaussian beams
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, XX, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (3,self._npix))
        _cleaned = maps.copy()
        _cleaned[:,self._mask==0.] = 0.  # !!!
        # select among T, E and B
        return self._autodict[self._targets](_cleaned,wsp,beams)

    def crosBP(self, maps, wsp=None, beams=[None,None]):
        """
        Cross BP
        
        Parameters
        ----------
        
        maps : numpy.ndarray
            A single-row array of a single map.
        
        wsp : (PS-estimator-defined) workspace
            A template of mask-induced mode coupling matrix.
        
        beams : float
            FWHM of gaussian beams.
        
        Returns
        -------
        
        pseudo-PS results : tuple of numpy.ndarray
            (ell, XX, wsp(if input wsp is None))
        """
        assert isinstance(maps, np.ndarray)
        assert (maps.shape == (6,self._npix))
        assert (len(beams) == 2)
        _cleaned = maps.copy()
        _cleaned[:,self._mask==0.] = 0.  # !!!
        # select among T, E and B
        return self._crosdict[self._targets](_cleaned,wsp,beams)

    def autoBP_T(self, maps, wsp=None, beams=None):
        dat = maps[0]
        # assemble NaMaster fields
        if beams is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(beams, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f0, f0, self._b)
            return (self._modes, self.rebinning(cl00[0]))
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, self.rebinning(cl00[0]))

    def crosBP_T(self, maps, wsp=None, beams=[None,None]):
        dat1 = maps[0]
        dat2 = maps[3]
        # assemble NaMaster fields
        if beams[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(beams[0], 3*self._nside-1))
        if beams[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(beams[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f01, f02, self._b)
            return (self._modes, self.rebinning(cl00[0]))
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, self.rebinning(cl00[0]))

    def autoBP_E(self, maps, wsp=None, beams=None):
        dat = self.purified_e(maps)
        # assemble NaMaster fields
        if beams is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(beams, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f0, f0, self._b)
            return (self._modes, self.rebinning(cl00[0]))
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, self.rebinning(cl00[0]))

    def crosBP_E(self, maps, wsp=None, beams=[None,None]):
        dat1 = self.purified_e(maps[:3])
        dat2 = self.purified_e(maps[3:])
        # assemble NaMaster fields
        if beams[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(beams[0], 3*self._nside-1))
        if beams[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(beams[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f01, f02, self._b)
            return (self._modes, self.rebinning(cl00[0]))
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, self.rebinning(cl00[0]))

    def autoBP_B(self, maps, wsp=None, beams=None):
        dat = self.purified_b(maps)
        # assemble NaMaster fields
        if beams is None:
            f0 = nmt.NmtField(self._apomask, [dat])
        else:
            f0 = nmt.NmtField(self._apomask, [dat], beam=hp.gauss_beam(beams, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f0, f0, self._b)
            return (self._modes, self.rebinning(cl00[0]))
        else:
            cl00c = nmt.compute_coupled_cell(f0, f0)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, self.rebinning(cl00[0]))

    def crosBP_B(self, maps, wsp=None, beams=[None,None]):
        dat1 = self.purified_b(maps[:3])
        dat2 = self.purified_b(maps[3:])
        # assemble NaMaster fields
        if beams[0] is None:
            f01 = nmt.NmtField(self._apomask, [dat1])
        else:
            f01 = nmt.NmtField(self._apomask, [dat1], beam=hp.gauss_beam(beams[0], 3*self._nside-1))
        if beams[1] is None:
            f02 = nmt.NmtField(self._apomask, [dat2])
        else:
            f02 = nmt.NmtField(self._apomask, [dat2], beam=hp.gauss_beam(beams[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00 = nmt.compute_full_master(f01, f02, self._b)
            return (self._modes, self.rebinning(cl00[0]))
        else:
            cl00c = nmt.compute_coupled_cell(f01, f02)
            cl00 = wsp.decouple_cell(cl00c)
            return (self._modes, self.rebinning(cl00[0]))

    def autoBP_EB(self, maps, wsp=None, beams=None):
        dat = self.purified_eb(maps)
        # assemble NaMaster fields
        if beams is None:
            f0_e = nmt.NmtField(self._apomask, [dat[0]])
            f0_b = nmt.NmtField(self._apomask, [dat[1]])
        else:
            f0_e = nmt.NmtField(self._apomask, [dat[0]], beam=hp.gauss_beam(beams, 3*self._nside-1))
            f0_b = nmt.NmtField(self._apomask, [dat[1]], beam=hp.gauss_beam(beams, 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00_e = nmt.compute_full_master(f0_e, f0_e, self._b)
            cl00_b = nmt.compute_full_master(f0_b, f0_b, self._b)
            return (self._modes, self.rebinning(cl00_e[0]), self.rebinning(cl00_b[0]))
        else:
            cl00c_e = nmt.compute_coupled_cell(f0_e, f0_e)
            cl00c_b = nmt.compute_coupled_cell(f0_b, f0_b)
            cl00_e = wsp.decouple_cell(cl00c_e)
            cl00_b = wsp.decouple_cell(cl00c_b)
            return (self._modes, self.rebinning(cl00_e[0]), self.rebinning(cl00_b[0]))

    def crosBP_EB(self, maps, wsp=None, beams=[None,None]):
        dat1 = self.purified_eb(maps[:3])
        dat2 = self.purified_eb(maps[3:])
        # assemble NaMaster fields
        if beams[0] is None:
            f01_e = nmt.NmtField(self._apomask, [dat1[0]])
            f01_b = nmt.NmtField(self._apomask, [dat1[1]])
        else:
            f01_e = nmt.NmtField(self._apomask, [dat1[0]], beam=hp.gauss_beam(beams[0], 3*self._nside-1))
            f01_b = nmt.NmtField(self._apomask, [dat1[1]], beam=hp.gauss_beam(beams[0], 3*self._nside-1))
        if beams[1] is None:
            f02_e = nmt.NmtField(self._apomask, [dat2[0]])
            f02_b = nmt.NmtField(self._apomask, [dat2[1]]) 
        else:
            f02_e = nmt.NmtField(self._apomask, [dat2[0]], beam=hp.gauss_beam(beams[1], 3*self._nside-1))
            f02_b = nmt.NmtField(self._apomask, [dat2[1]], beam=hp.gauss_beam(beams[1], 3*self._nside-1))
        # estimate PS
        if wsp is None:
            cl00_e = nmt.compute_full_master(f01_e, f02_e, self._b)
            cl00_b = nmt.compute_full_master(f01_b, f02_b, self._b)
            return (self._modes, self.rebinning(cl00_e[0]), self.rebinning(cl00_b[0]))
        else:
            cl00c_e = nmt.compute_coupled_cell(f01_e, f02_e)
            cl00c_b = nmt.compute_coupled_cell(f01_b, f02_b)
            cl00_e = wsp.decouple_cell(cl00c_e)
            cl00_b = wsp.decouple_cell(cl00c_b)
            return (self._modes, self.rebinning(cl00_e[0]), self.rebinning(cl00_b[0]))

    def autoBP_TEB(self, maps, wsp=None, beams=None):
        pass

    def crosBP_TEB(self, maps, wsp=None, beams=[None,None]):
        pass
