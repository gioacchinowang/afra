import numpy as np
import healpy as hp
from afra.methods.abs import abssep
from afra.tools.ps_estimator import pstimator
from afra.tools.icy_decorator import icy


@icy
class abspipe(object):
    """The ABS pipeline class."""

    def __init__(self, signals, noises=None, mask=None, fwhms=None, targets='T'):
        """
        The ABS pipeline for extracting CMB power-spectrum band power,
        according to given measured sky maps at various frequency bands.
        
        Parameters
        ----------
        
        signals : dict
            Measured signal maps,
            should be arranged in form: {frequency (GHz) : array(map #, pixel #)}.
        
        noises : dict
            Simulated noise map samples,
            should be arranged in type: {frequency (GHz): (sample #, map #, pixel #)}.
        
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (pixel #,).
        
        fwhms : dict
            FWHM (in rad) of gaussian beams for each frequency.
        
        targets : str
            Choose among 'T', 'E', 'B', 'EB', 'TEB'.
        """
        self.signals = signals
        self.noises = noises
        self.mask = mask
        self.fwhms = fwhms
        self.targets = targets
        # method select dict with keys defined by self._noise_flag
        self._methodict = {(True): self.method_noisy,
                           (False): self.method_quiet}
        # debug mode
        self.debug = False
        # ps estimator
        self._est = None

    @property
    def signals(self):
        return self._signals

    @property
    def noises(self):
        return self._noises

    @property
    def mask(self):
        return self._mask
        
    @property
    def freqlist(self):
        return self._freqlist

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def nside(self):
        return self._nside

    @property
    def targets(self):
        return self._targets

    @property
    def ntarget(self):
        return self._ntarget

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def debug(self):
        return self._debug

    @property
    def fwhms(self):
        return self._fwhms

    @signals.setter
    def signals(self, signals):
        """detect and register nfreq, nmap, npix and nside automatically
        """
        assert isinstance(signals, dict)
        self._nfreq = len(signals)
        self._freqlist = sorted(signals.keys())
        assert (len(signals[next(iter(signals))].shape) == 2)
        assert (signals[next(iter(signals))].shape[0] == 3)
        self._npix = signals[next(iter(signals))].shape[1]
        self._nside = int(np.sqrt(self._npix//12))
        self._signals = signals.copy()

    @noises.setter
    def noises(self, noises):
        if noises is not None:
            assert isinstance(noises, dict)
            assert (len(noises) == self._nfreq)
            assert (len(noises[next(iter(noises))].shape) == 3)
            self._nsamp = noises[next(iter(noises))].shape[0]
            assert (noises[next(iter(noises))].shape[1] == 3)
            assert (noises[next(iter(noises))].shape[2] == self._npix)
            self._noises = noises.copy()
            self._noise_flag = True
        else:
            self._noises = None
            self._noise_flag = False

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = np.ones(self._npix,dtype=np.float32)
        else:
            assert isinstance(mask, np.ndarray)
            assert (len(mask) == self._npix)
            self._mask = mask.copy()
        # clean up input maps with mask
        for f in self._freqlist:
        	self._signals[f][:,self._mask==0.] = 0.
        	if self._noises is not None:
                    self._noises[f][:,:,self._mask==0.] = 0.

    @debug.setter
    def debug(self, debug):
        assert isinstance(debug, bool)
        self._debug = debug

    @fwhms.setter
    def fwhms(self, fwhms):
        """signal maps' fwhms"""
        if fwhms is not None:
            assert isinstance(fwhms, dict)
            assert (len(fwhms) == self._nfreq)
            self._fwhms = fwhms.copy()
        else:
            self._fwhms = dict()
            for f in self._freqlist:
                self._fwhms[f] = None

    @targets.setter
    def targets(self, targets):
        assert isinstance(targets, str)
        self._targets = targets
        self._ntarget = len(targets)

    def run(self, aposcale, psbin, lmin=None, lmax=None, shift=None, threshold=None):
        """
        ABS pipeline class call function.
        
        Parameters
        ----------
        
        aposcale : float
            Apodization scale.
        
        psbin : integer
            Number of angular modes in each bin,
            for conducting pseudo-PS estimation.
        
        shift : float
            ABS method shift parameter.
        
        threshold : float or None
            ABS method threshold parameter.
            
        Returns
        -------
        
        angular modes, target angular power spectrum : tuple of lists
        """
        assert isinstance(aposcale, float)
        assert isinstance(psbin, int)
        assert (psbin > 0)
        assert (aposcale > 0)
        # init PS estimator
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmin=lmin,lmax=lmax,targets=self._targets)
        # method selection
        return self._methodict[self._noise_flag](shift, threshold)

    def method_quiet(self, shift, threshold):
        """
        CMB (band power) extraction without noise.
        
        Returns
        -------
        angular modes, requested PS, eigen info
        """
        modes = self._est.modes
        # prepare total signals PS in the shape required by ABS method
        signal_ps = np.zeros((self._ntarget,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        for i in range(self._nfreq):
            _fi = self._freqlist[i]
            # auto correlation
            stmp = self._est.autoBP(self._signals[_fi],fwhms=self._fwhms[_fi])
            # assign results
            for t in range(self._ntarget):
                for k in range(len(modes)):
                    signal_ps[t,k,i,i] = stmp[1+t][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                _fj = self._freqlist[j]
                stmp = self._est.crosBP(np.r_[self._signals[_fi],self._signals[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                for t in range(self._ntarget):
                    for k in range(len(modes)):
                        signal_ps[t,k,i,j] = stmp[1+t][k]
                        signal_ps[t,k,j,i] = stmp[1+t][k]
        # send PS to ABS method, noiseless case requires no shift nor threshold
        rslt = np.empty((self._ntarget,len(modes)),dtype=np.float32)
        info = dict()
        for t in range(self._ntarget):
            spt = abssep(signal_ps[t],shift=None,threshold=None)
            rslt[t] = spt.run()
            if self._debug:
                info[self._targets[t]] = spt.run_info()
        return (np.r_[modes.reshape(1,-1), rslt], info)

    def method_noisy_raw(self):
        """
        CMB T mode (band power) extraction with noise.
        
        Returns
        -------
        
        angular modes, T-mode PS, T-mode PS std
        """
        # run trial PS estimations for workspace template
        wsp_dict = dict()
        modes = self._est.modes
        for i in range(self._nfreq):
            _fi = self._freqlist[i]
            wsp_dict[(i,i)] = self._est.autoWSP(self._signals[_fi],fwhms=self._fwhms[_fi])
            for j in range(i+1,self._nfreq):
                _fj = self._freqlist[j]
                wsp_dict[(i,j)] = self._est.crosWSP(np.r_[self._signals[_fi],self._signals[_fj]],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
        # allocate
        noise_ps = np.zeros((self._nsamp,self._ntarget,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        signal_ps = np.zeros((self._nsamp,self._ntarget,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                _fi = self._freqlist[i]
                # auto correlation
                ntmp = self._est.autoBP(self._noises[_fi][s],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[_fi])
                stmp = self._est.autoBP(self._signals[_fi]+self._noises[_fi][s],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[_fi])
                # assign results
                for t in range(self._ntarget):
                    for k in range(len(modes)):
                        noise_ps[s,t,k,i,i] = ntmp[1+t][k]
                        signal_ps[s,t,k,i,i] = stmp[1+t][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    _fj = self._freqlist[j]
                    # cross correlation
                    ntmp = self._est.crosBP(np.r_[self._noises[_fi][s],self._noises[_fj][s]],wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                    stmp = self._est.crosBP(np.r_[self._signals[_fi]+self._noises[_fi][s],self._signals[_fj]+self._noises[_fj][s]],wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[_fi],self._fwhms[_fj]])
                    for t in range(self._ntarget):
                        for k in range(len(modes)):
                            noise_ps[s,t,k,i,j] = ntmp[1+t][k]
                            noise_ps[s,t,k,j,i] = ntmp[1+t][k]
                            signal_ps[s,t,k,i,j] = stmp[1+t][k]
                            signal_ps[s,t,k,j,i] = stmp[1+t][k]
        return (modes, noise_ps, signal_ps)

    def method_noisy(self, shift, threshold):
        modes, noise_ps, signal_ps = self.method_noisy_raw()
        # get noise PS mean and rms
        noise_ps_mean = np.mean(noise_ps,axis=0)
        noise_ps_std = np.std(noise_ps,axis=0)
        noise_ps_std_diag = np.zeros((self._ntarget,len(modes),self._nfreq),dtype=np.float32)
        for t in range(self._ntarget):
            for l in range(len(modes)):
                noise_ps_std_diag[t,l] = np.diag(noise_ps_std[t,l])
        # shift for each angular mode independently
        safe_shift = shift*np.mean(noise_ps_std_diag,axis=2)  # safe_shift in shape (nmode,ntarget)
        rslt = np.empty((self._nsamp,self._ntarget,len(modes)),dtype=np.float32)
        info = dict()
        for s in range(self._nsamp):
            for t in range(self._ntarget):
                # send PS to ABS method
                spt = abssep(signal_ps[s,t]-noise_ps_mean[t],noise_ps_mean[t],noise_ps_std_diag[t],shift=safe_shift[t],threshold=threshold)
                rslt[s,t] = spt.run()
                if self._debug:
                    info[self._targets[t]] = spt.run_info()
        return (np.r_[modes.reshape(1,-1), np.mean(rslt,axis=0), np.std(rslt,axis=0)], info)
