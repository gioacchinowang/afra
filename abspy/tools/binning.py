import numpy as np
from copy import deepcopy

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
    Binned average of CROSS-power-spectrum and convert it into CROSS-Dl (band power).
    
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
    cps_cp = deepcopy(cps)  # avoid mem issue
    # binned average for each single spectrum
    for i in range(bins):
        begin = min(lres,i)+i*lmod
        end = min(lres,i) + (i+1)*lmod + int(i < lres)
        # convert Cl into Dl for each single spectrum
        effl = 0.5*(modes[begin]+modes[end-1])
        result[i,:,:] = np.mean(cps_cp[begin:end,:,:], axis=0)*0.5*effl*(effl+1)/np.pi
    return result

def binaps(aps, modes, bins):
    """
    Binned average of AUTO-power-spectrum Cl and convert it into AUTO-Dl (band power).
    
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
    aps_cp = deepcopy(aps)
    # binned average for each single spectrum
    for i in range(bins):
        begin = min(lres,i)+i*lmod
        end = min(lres,i) + (i+1)*lmod + int(i < lres)
        effl = 0.5*(modes[begin]+modes[end-1])
        # convert Cl into Dl for each single spectrum
        result[i,:] = np.mean(aps_cp[begin:end,:], axis=0)*0.5*effl*(effl+1)/np.pi
    return result
