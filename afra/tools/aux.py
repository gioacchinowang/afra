"""auxiliary functions"""

import numpy as np
import healpy as hp
from scipy.linalg import sqrtm
from afra.tools.ps_estimator import pstimator


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


def emp_cov(sample):
    assert isinstance(sample, np.ndarray)
    n, p = sample.shape
    assert (n > 0 and p > 0)
    if n == 1:
        return np.zeros((p, p))
    m = np.mean(sample, axis=0)
    u = sample - m
    s = np.dot(u.T, u) / n
    return s


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
        cross-PS with dimension (# sample, # types, # modes, # freq, # freq)
        or                      (# types, # modes, # freq, # freq)
    
    Returns
    -------
    
    vectorized cps : numpy.ndarray
    """
    assert isinstance(cps, np.ndarray)
    assert (cps.shape[-1] == cps.shape[-2])
    nfreq = cps.shape[-2]
    nmode = cps.shape[-3]
    ntype = cps.shape[-4]
    dof = nfreq*(nfreq+1)//2  # distinctive elements at each mode
    triu_idx = np.triu_indices(nfreq)
    if (len(cps.shape) == 4):
        rslt = np.zeros(ntype*nmode*dof)
        for t in range(ntype):
            for l in range(nmode):
                rslt[(t*nmode+l)*dof:(t*nmode+l+1)*dof] = cps[t,l][triu_idx]
        return rslt
    elif (len(cps.shape) == 5):
        nsamp = cps.shape[0]
        rslt = np.zeros((nsamp,ntype*nmode*dof))
        for s in range(nsamp):
            for t in range(ntype):
                for l in range(nmode):
                    rslt[s,(t*nmode+l)*dof:(t*nmode+l+1)*dof] = cps[s,t,l][triu_idx]
        return rslt
    else:
        raise ValueError('unsupported input shape')


def vec_hl(cps,cps_hat,cps_fid):
    """with measured cps_hat, fiducial cps_fid, modeled cps
    """
    assert (len(cps.shape) == 4)
    assert isinstance(cps, np.ndarray)
    assert (cps.shape[-1] == cps.shape[-2])
    nfreq = cps.shape[-2]
    nmode = cps.shape[-3]
    ntype = cps.shape[-4]
    dof = nfreq*(nfreq+1)//2
    triu_idx = np.triu_indices(nfreq)
    rslt = np.ones(ntype*nmode*dof)
    for t in range(ntype):
        for l in range(nmode):
            c_h = cps_hat[t,l]
            c_f = sqrtm(cps_fid[t,l])
            c_inv = sqrtm(np.linalg.inv(cps[t,l]))
            res = np.dot(np.conjugate(c_inv), np.dot(c_h, c_inv))
            [d, u] = np.linalg.eigh(res)
            assert (any(d>=0))
            #if (any(d<0)):
            #    rslt[(t*nmode+l)*dof:(t*nmode+l+1)*dof] *= np.nan_to_num(np.inf)
            #else:
            # real symmetric matrices are diagnalized by orthogonal matrices (M^t M = 1)
            # this makes a diagonal matrix by applying g(x) to the eigenvalues, equation 10 in Barkats et al
            gd = np.diag( np.sign(d - 1.) * np.sqrt(2. * (d - np.log(d) - 1.)) )
            # multiplying from right to left
            x = np.dot(gd, np.dot(np.transpose(u),c_f))
            x = np.dot(np.conjugate(c_f), np.dot(np.conjugate(u),x))
            rslt[(t*nmode+l)*dof:(t*nmode+l+1)*dof] = x[triu_idx]
    return rslt


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
