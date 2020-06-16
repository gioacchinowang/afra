"""auxiliary functions"""
import numpy as np


def binell(modes, bins):
    """
    Angular mode binning.
    
    Parameters
    ----------
    
    modes : list
        Input angular mode list.
        
    bins : integer
        Output angular mode bin number.
        
    Return
    ------
    
    result : list
        List of central angular modes for each bin.
    """
    assert isinstance(modes, list)
    assert isinstance(bins, int)
    result = list()
    lres = len(modes)%bins
    lmod = len(modes)//bins
    # binned average for each single spectrum
    for i in range(bins):
        begin = min(lres,i)+i*lmod
        end = min(lres,i) + (i+1)*lmod + int(i < lres)
        result.append(0.5*(modes[begin]+modes[end-1]))
    return result
    

def bincps(cps, modes, bins):
    """
    Binned average of CROSS-power-spectrum (band power).
    
    Parameters
    ----------
    
    cps : numpy.ndarray
        Cross power spectrum in shape (N_ell, N_freq, N_freq).
        
    modes : list
        Input angular mode list.
        
    bins : integer
        Output angular mode bin number.
    
    Returns
    -------
    
    result : numpy.ndarray
        Binned cross band power in shape (N_bins, N_freq, N_freq).
    """
    assert isinstance(cps, np.ndarray)
    assert isinstance(modes, list)
    assert isinstance(bins, int)
    assert (cps.shape[0] == len(modes))
    assert (cps.shape[1] == cps.shape[2])
    lres = len(modes)%bins
    lmod = len(modes)//bins
    result = np.empty((bins, cps.shape[1], cps.shape[2]))
    cps_cp = cps.copy()  # avoid mem issue
    for i in range(len(modes)):
        cps_cp[i] *= 2.*np.pi/(modes[i]*(modes[i]+1))
    # binned average for each single spectrum
    for i in range(bins):
        begin = min(lres,i)+i*lmod
        end = min(lres,i) + (i+1)*lmod + int(i < lres)
        effl = 0.5*(modes[begin]+modes[end-1])
        result[i,:,:] = np.mean(cps_cp[begin:end,:,:], axis=0)*0.5*effl*(effl+1)/np.pi
    return result


def binaps(aps, modes, bins):
    """
    Binned average of AUTO-power-spectrum (band power).
    
    Parameters
    ----------
    
    aps : numpy.ndarray
        Auto power spectrum in shape (N_ell, N_freq).
        
    modes : list
        Input angular mode list.
        
    bins : integer
        Output angular mode bin number.
    
    Returns
    -------
    
    result : numpy.ndarray
        Binned auto band power in shape (N_bins, N_freq).
    """
    assert isinstance(aps, np.ndarray)
    assert (aps.shape[0] == len(modes))
    lres = len(modes)%bins
    lmod = len(modes)//bins
    # allocate results
    result = np.empty((bins, aps.shape[1]))
    aps_cp = aps.copy()
    for i in range(len(modes)):
        aps_cp[i] *= 2.*np.pi/(modes[i]*(modes[i]+1))
    # binned average for each single spectrum
    for i in range(bins):
        begin = min(lres,i)+i*lmod
        end = min(lres,i) + (i+1)*lmod + int(i < lres)
        effl = 0.5*(modes[begin]+modes[end-1])
        result[i,:] = np.mean(aps_cp[begin:end,:], axis=0)*0.5*effl*(effl+1)/np.pi
    return result
    
    
def oas_cov(sample):
    """
    OAS covariance estimator.
    
    Parameters
    ----------
    sample : numpy.ndarray
        ensemble of observables, in shape (# ensemble,# data)
        
    Returns
    -------
    covariance matrix : numpy.ndarray
        covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(sample, np.ndarray)
    _n, _p = sample.shape
    assert (_n > 0 and _p > 0)
    if _n == 1:
        return np.zeros((_p, _p))
    _m = np.mean(sample, axis=0)
    _u = sample - _m
    _s = np.dot(_u.T, _u) / _n
    _trs = np.trace(_s)
    # IMAGINE implementation
    '''
    _trs2 = np.trace(np.dot(_s, _s))
    _numerator = (1 - 2. / _p) * _trs2 + _trs * _trs
    _denominator = (_n + 1. - 2. / _p) * (_trs2 - (_trs*_trs) / _p)
    '''
    # skylearn implementation
    _mu = (_trs / _p)
    _alpha = np.mean(_s ** 2)
    _numerator = _alpha + _mu ** 2
    _denominator = (_n + 1.) * (_alpha - (_mu ** 2) / _p)
    #
    _rho = 1. if _denominator == 0 else min(1., _numerator / _denominator)
    return (1. - _rho) * _s + np.eye(_p) * _rho * _trs / _p
    
    
def vecp(cps):
    """vectorize cross-power-spectrum band power
    with repeated symetric elements trimed
	
    Parameters
    ----------
	
    cps : numpy.ndarray
    cross-PS with dimension (# modes, # freq, # freq)
    """
    assert isinstance(cps, np.ndarray)
    assert (len(cps.shape) == 3)
    assert (cps.shape[1] == cps.shape[2])
    _nmode = cps.shape[0]
    _nfreq = cps.shape[1]
    _dof = _nfreq*(_nfreq+1)//2
    _rslt = np.zeros(_nmode*_dof)
    for _l in range(_nmode):
        _trimed = np.triu(cps[_l],k=0)
        _rslt[_l*_dof:(_l+1)*_dof] = _trimed[_trimed!=0]
    return _rslt

def vecs(cps):
    """reshuffle vecp results with the correct order:

        TT TE TB
           EE EB
              BB
    for each angular mode and each frequency pairs in a row
    """
    assert isinstance(cps, (list,tuple))
    _ntype = len(cps)
    _len = len(cps[0])
    _rslt = np.zeros(_ntype*_len)
    for _i in range(_len):
        for _j in range(_ntype):
            _rslt[_i*_ntype+_j] = cps[_j][_i]
    return _rslt

def hl_g(x):
    """HL likelihood g(x) function"""
    assert (x > 0.)
    return np.sign(x-1.)*np.sqrt(2.*(x-np.log(x)-1.))
