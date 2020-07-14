"""auxiliary functions"""
import numpy as np
from afra.tools.ps_estimator import pstimator


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
    n, p = sample.shape
    assert (n > 0 and p > 0)
    if n == 1:
        return np.zeros((p, p))
    m = np.mean(sample, axis=0)
    u = sample - m
    s = np.dot(u.T, u) / n
    trs = np.trace(s)
    # IMAGINE implementation
    """
    trs2 = np.trace(np.dot(s, s))
    numerator = (1 - 2. / p) * trs2 + trs * trs
    denominator = (n + 1. - 2. / p) * (trs2 - (trs*trs) / p)
    """
    # skylearn implementation
    mu = (trs / p)
    alpha = np.mean(s ** 2)
    numerator = alpha + mu ** 2
    denominator = (n + 1.) * (alpha - (mu ** 2) / p)
    #
    rho = 1. if denominator == 0 else min(1., numerator / denominator)
    return (1. - rho) * s + np.eye(p) * rho * trs / p
    
    
def vec_simple(cps):
    """vectorize cross-power-spectrum band power
    with repeated symetric elements trimed
	
    Parameters
    ----------
	
    cps : numpy.ndarray
        cross-PS with dimension (# sample, # modes, # freq, # freq)
        or                      (# modes, # freq, # freq)

    Returns
    -------
    vectorized cps : numpy.ndarray
    """
    assert isinstance(cps, np.ndarray)
    assert (cps.shape[-1] == cps.shape[-2])
    nfreq = cps.shape[-2]
    nmode = cps.shape[-3]
    dof = nfreq*(nfreq+1)//2
    if (len(cps.shape) == 3):
        rslt = np.zeros(nmode*dof)
        for l in range(nmode):
            trimed = np.triu(cps[l],k=0)
            rslt[l*dof:(l+1)*dof] = trimed[trimed!=0]
        return rslt
    elif (len(cps.shape) == 4):
        nsamp = cps.shape[0]
        rslt = np.zeros((nsamp,nmode*dof))
        for s in range(nsamp):
            for l in range(nmode):
                trimed = np.triu(cps[s,l],k=0)
                rslt[s,l*dof:(l+1)*dof] = trimed[trimed!=0]
        return rslt
    else:
        raise ValueError('unsupported input shape')


def bp_window(ps_estimator,lmax,offset=1):
    """window function matrix for converting PS into band-powers

    Parameters
    ----------

    ps_estimator

    lmax : int
        maximal multipole for Cl

    offset : int
        discard the first offset multipole bins
    """
    assert isinstance(ps_estimator, pstimator)
    assert isinstance(lmax, int)
    assert (lmax >= ps_estimator.lmax)
    compress = np.zeros((len(ps_estimator.modes)-offset,lmax))
    for i in range(len(ps_estimator.modes)-offset):
        lrange = np.array(ps_estimator._b.get_ell_list(i+offset))
        factor = 0.5*lrange*(lrange+1)/np.pi
        w = np.array(ps_estimator._b.get_weight_list(i+offset))
        compress[i,lrange] = w*factor
    return compress
