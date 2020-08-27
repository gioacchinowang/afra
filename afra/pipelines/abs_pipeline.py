import numpy as np
import healpy as hp
from afra.methods.abs import abssep
from afra.tools.ps_estimator import pstimator
from afra.tools.icy_decorator import icy


@icy
class abspipe(object):
    """The ABS pipeline class."""

    def __init__(self, signals, variances=None, mask=None, fwhms=None, target='T', nsamp=1000):
        """
        The ABS pipeline for extracting CMB power-spectrum band power,
        according to given measured sky maps at various frequency bands.
        
        Parameters
        ----------
        
        signals : dict
            Measured signal maps,
            should be arranged in form: {frequency (GHz) : array(map #, pixel #)}.
        
        variances : dict
            Measured noise variance maps,
            should be arranged in form: {frequency (GHz): array(map #, pixel #)}.
            By default, no variance maps required.
        
        mask : numpy.ndarray
            Single mask map,
            should be arranged in shape: (pixel #,).
        
        fwhms : dict
            FWHM (in rad) of gaussian beams for each frequency.
        
        target : str
            Choose among 'T', 'E' and 'B'.
        
        nsamp : int
            Size of re-sampling.
        """
        self.signals = signals
        self.variances = variances
        self.mask = mask
        self.fwhms = fwhms
        self._target = target
        # method select dict with keys defined by self._noise_flag
        self._methodict = {(True): self.method_noisy,
                           (False): self.method_quiet}
        # resampling size
        self.nsamp = nsamp
        # debug mode
        self.debug = False
        # ps estimator
        self._est = None

    @property
    def signals(self):
        return self._signals

    @property
    def variances(self):
        return self._variances

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
    def target(self):
        return self._target

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
        self._signals = signals

    @variances.setter
    def variances(self, variances):
        if variances is not None:
            assert isinstance(variances, dict)
            assert (variances[next(iter(variances))].shape == (3,self._npix))
            self._noise_flag = True
            self._variances = variances
        else:
            self._noise_flag = False
            self._variances = None

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
        	if self._variances is not None:
        	    self._variances[f][:,self._mask==0.] = 0.

    @nsamp.setter
    def nsamp(self, nsamp):
        assert isinstance(nsamp, int)
        self._nsamp = nsamp

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
            self._fwhms = fwhms
        else:
            self._fwhms = dict()
            for f in self._freqlist:
                self._fwhms[f] = None

    @target.setter
    def target(self, target):
        assert isinstance(target, str)
        self._target = target

    def run(self, aposcale, psbin, lmin=None, lmax=None, shift=None, threshold=None, verbose=False):
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
            
        verbose : boolean
            Return with extra information.
        
        Returns
        -------
        
        angular modes, target angular power spectrum : tuple of lists
        """
        assert isinstance(aposcale, float)
        assert isinstance(psbin, int)
        assert (psbin > 0)
        assert (aposcale > 0)
        # init PS estimator
        self._est = pstimator(nside=self._nside,mask=self._mask,aposcale=aposcale,psbin=psbin,lmin=lmin,lmax=lmax,target=self._target)
        # method selection
        return self._methodict[self._noise_flag](shift, threshold, verbose)

    def method_quiet(self, shift, threshold, verbose):
        """
        CMB T mode (band power) extraction without noise.
        
        Returns
        -------
        angular modes, T-mode PS
        """
        modes = self._est.modes
        # prepare total signals PS in the shape required by ABS method
        signal_ps = np.zeros((len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        for i in range(self._nfreq):
            # auto correlation
            stmp = self._est.autoBP(self._signals[self._freqlist[i]],fwhms=self._fwhms[self._freqlist[i]])
            # assign results
            for k in range(len(modes)):
                signal_ps[k,i,i] = stmp[1][k]
            # cross correlation
            for j in range(i+1,self._nfreq):
                stmp = self._est.crosBP(np.r_[self._signals[self._freqlist[i]],self._signals[self._freqlist[j]]],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                for k in range(len(modes)):
                    signal_ps[k,i,j] = stmp[1][k]
                    signal_ps[k,j,i] = stmp[1][k]
        # send PS to ABS method, noiseless case requires no shift nor threshold
        spt = abssep(signal_ps,shift=None,threshold=None)
        if verbose:
            return (modes, spt.run(), spt.run_info())
        return (modes, spt.run())

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
            wsp_dict[(i,i)] = self._est.autoWSP(self._signals[self._freqlist[i]],fwhms=self._fwhms[self._freqlist[i]])
            for j in range(i+1,self._nfreq):
                wsp_dict[(i,j)] = self._est.crosWSP(np.r_[self._signals[self._freqlist[i]],self._signals[self._freqlist[j]]],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
        # allocate
        noise_ps = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        signal_ps = np.zeros((self._nsamp,len(modes),self._nfreq,self._nfreq),dtype=np.float32)
        noise_map = np.zeros((6,self._npix),dtype=np.float32)
        signal_map = np.zeros((6,self._npix),dtype=np.float32)
        for s in range(self._nsamp):
            # prepare noise samples on-fly
            for i in range(self._nfreq):
                # noise realization
                noise_map[:3] = np.random.normal(size=(3,self._npix))*np.sqrt(self._variances[self._freqlist[i]])
                signal_map[:3] = self._signals[self._freqlist[i]] + noise_map[:3]
                # auto correlation
                ntmp = self._est.autoBP(noise_map[:3],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[self._freqlist[i]])
                stmp = self._est.autoBP(signal_map[:3],wsp=wsp_dict[(i,i)],fwhms=self._fwhms[self._freqlist[i]])
                # assign results
                for k in range(len(modes)):
                    noise_ps[s,k,i,i] = ntmp[1][k]
                    signal_ps[s,k,i,i] = stmp[1][k]
                # cross correlation
                for j in range(i+1,self._nfreq):
                    # noise realization
                    noise_map[3:] = np.random.normal(size=(3,self._npix))*np.sqrt(self._variances[self._freqlist[j]])
                    signal_map[3:] = self._signals[self._freqlist[j]] + noise_map[3:]
                    # cross correlation
                    ntmp = self._est.crosBP(noise_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                    stmp = self._est.crosBP(signal_map,wsp=wsp_dict[(i,j)],fwhms=[self._fwhms[self._freqlist[i]],self._fwhms[self._freqlist[j]]])
                    for k in range(len(modes)):
                        noise_ps[s,k,i,j] = ntmp[1][k]
                        noise_ps[s,k,j,i] = ntmp[1][k]
                        signal_ps[s,k,i,j] = stmp[1][k]
                        signal_ps[s,k,j,i] = stmp[1][k]
        return (modes, noise_ps, signal_ps)

    def method_noisy(self, shift, threshold, verbose):
        modes, noise_ps, signal_ps = self.method_noisy_raw()
        # get noise PS mean and rms
        noise_ps_mean = np.mean(noise_ps,axis=0)
        noise_ps_std = np.std(noise_ps,axis=0)
        noise_ps_std_diag = np.zeros((len(modes),self._nfreq),dtype=np.float32)
        for l in range(len(modes)):
            noise_ps_std_diag[l] = np.diag(noise_ps_std[l])
        # shift for each angular mode independently
        safe_shift = shift*np.mean(noise_ps_std_diag,axis=1)
        rslt = np.zeros((self._nsamp,len(modes)),dtype=np.float32)
        for s in range(self._nsamp):
            # send PS to ABS method
            spt = abssep(signal_ps[s]-noise_ps_mean,noise_ps_mean,noise_ps_std_diag,shift=safe_shift,threshold=threshold)
            rslt[s] = spt.run()
        if verbose:
            spt = abssep(np.mean(signal_ps,axis=0)-noise_ps_mean,noise_ps_mean,noise_ps_std_diag,shift=safe_shift,threshold=threshold)
            if self._debug:
                return (modes, rslt, spt.run_info())
            return (modes, np.mean(rslt,axis=0), np.std(rslt,axis=0), spt.run_info())
        if self._debug:
            return (modes, rslt)
        return (modes, np.mean(rslt,axis=0), np.std(rslt,axis=0))
