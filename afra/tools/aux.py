"""auxiliary functions"""
import numpy as np
import healpy as hp
from scipy.linalg import sqrtm
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


def unity_mapper(x, r=[0.,1.]):
    """
    Maps x from [0, 1] into the interval [a, b]

    Parameters
    ----------
    x : float
        the variable to be mapped
    range : list,tuple
        the lower and upper parameter value limits

    Returns
    -------
    numpy.float64
        The mapped parameter value
    """
    assert isinstance(r, (list,tuple))
    assert (len(r) == 2)
    return x * (r[1]-r[0]) + r[0]


def vec_gauss(cps):
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


def vec_hl(cps,cps_hat,cps_fid):
    """with measured cps_hat, fiducial cps_fid, modeled cps
    """
    assert isinstance(cps, np.ndarray)
    assert (cps.shape[-1] == cps.shape[-2])
    nfreq = cps.shape[-2]
    nmode = cps.shape[-3]
    dof = nfreq*(nfreq+1)//2
    rslt = np.zeros(nmode*dof)
    for l in range(cps.shape[0]):
        c_h = cps_hat[l]
        c_f = sqrtm(cps_fid[l])
        c_inv = sqrtm(np.linalg.pinv(cps[l]))
        res = np.dot(np.conjugate(c_inv), np.dot(c_h, c_inv))
        [d, u] = np.linalg.eigh(res)
        #assert (all(d>=0))
        # real symmetric matrices are diagnalized by orthogonal matrices (M^t M = 1)
        # this makes a diagonal matrix by applying g(x) to the eigenvalues, equation 10 in Barkats et al
        gd = np.diag( np.sign(d - 1.) * np.sqrt(2. * (d - np.log(d) - 1.)) )
        # multiplying from right to left
        x = np.dot(gd, np.dot(np.transpose(u),c_f))
        x = np.dot(c_f, np.dot(u,x))
        trimed = np.triu(x,k=0)
        rslt[l*dof:(l+1)*dof] = trimed[trimed!=0]
    return rslt


def bp_window(ps_estimator):
    """
    "top-hat" window function matrix 
    for converting PS into band-powers

    Parameters
    ----------

    ps_estimator
        the wrapped-in power-spectrum estimator
    """
    assert isinstance(ps_estimator, pstimator)
    compress = np.zeros((len(ps_estimator.modes),3*ps_estimator.nside))
    for i in range(len(ps_estimator.modes)):
        lrange = np.array(ps_estimator._b.get_ell_list(i))
        factor = 0.5*lrange*(lrange+1)/np.pi
        w = np.array(ps_estimator._b.get_weight_list(i))
        compress[i,lrange] = w*factor
    return compress


def Mbpconv_t(nside,mask=None,ensemble=10):
    """
    matrix for converting true PS to masked PS,
    works for T mode to T mode conversion,
    Cl_masked = M.dott(Cl_true)
    """
    if mask is None:
        mask = np.ones(12*nside**2)
    else:
        assert isinstance(mask, np.ndarray)
        assert (len(mask) == 12*nside**2)
    lmax = 3*nside
    result = np.zeros((lmax,lmax))
    # do not count monopole & dipole
    for l_pivot in range(2,lmax):
        cl_in = np.zeros(lmax)
        cl_out = np.zeros(lmax)
        cl_in[l_pivot] = 1.
        for k in range(ensemble):
            intmap = hp.synfast(cl_in,nside=nside,new=True,verbose=False)
            intmap *= mask
            cl_out += hp.anafast(intmap)
        result[:,l_pivot] = cl_out/ensemble
    return result


def Mbpconv_eb(nside,mask=None,ensemble=10):
    """
    matrix for converting true PS to masked PS,
    works for E&B mode to E mode conversion,
    [Cl_masked_ee,Cl_masked_bb] = M.dott(np.r_[Cl_true_ee,Cl_true_bb]).reshape(2,-1)
    """
    if mask is None:
        mask = np.ones(12*nside**2)
    else:
        assert isinstance(mask, np.ndarray)
        assert (len(mask) == 12*nside**2)
    lmax = 3*nside
    result = np.zeros((2*lmax,2*lmax))
    # do not count monopole & dipole
    for l_pivot in range(2,lmax):
        cl_in = np.zeros((2,4,lmax))
        cl_out = np.zeros((4,lmax))
        cl_in[0,1,l_pivot] = 1.  # E mode injection
        cl_in[1,2,l_pivot] = 1.  # B mode injection
        for k in range(ensemble):
            intmap_e = hp.synfast(cl_in[0],nside=nside,new=True,verbose=False)
            intmap_b = hp.synfast(cl_in[1],nside=nside,new=True,verbose=False)
            intmap_e *= mask
            intmap_b *= mask
            tmp_cl_e = hp.anafast(intmap_e)
            tmp_cl_b = hp.anafast(intmap_b)
            cl_out[0] += tmp_cl_e[1]  # E to E
            cl_out[1] += tmp_cl_b[1]  # B to E
            cl_out[2] += tmp_cl_e[2]  # E to B
            cl_out[3] += tmp_cl_b[2]  # B to B
        result[:lmax,l_pivot] = cl_out[0]/ensemble
        result[:lmax,l_pivot+lmax] = cl_out[1]/ensemble
        result[lmax:,l_pivot] = cl_out[2]/ensemble
        result[lmax:,l_pivot+lmax] = cl_out[3]/ensemble
    return result
