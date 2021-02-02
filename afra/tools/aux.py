import numpy as np
import healpy as hp
from scipy.linalg import sqrtm


def empcov(sample):
    """
    empirical covariance matrix estimation
    + conditional Hartlap correction

    Parameters
    ----------
    sample:
        input sample
    """
    assert isinstance(sample, np.ndarray)
    n,p = sample.shape
    assert (n>1 and p>0)
    u = sample-np.mean(sample,axis=0)
    s = np.dot(u.T,u)/(n-1)
    # with Hartlap correction under condition p<0.12n
    if (0.12*n > p):
        s *= (n-1)/(n-p-2)
    return s


def oascov(sample, bsize=1):
    """
    OAS shrinkage covariance matrix estimation
    + conditional Hartlap correction
    + block diagonal adjustment
    
    Parameters
    ---------
    sample:
        input sample
    
    bsize:
        block size
    """
    assert isinstance(sample,np.ndarray)
    n,p = sample.shape
    assert (n>1 and p>0)
    # empirical estimation
    u = sample-np.mean(sample,axis=0)
    s = np.dot(u.T,u)/(n-1)
    # with Hartlap correction under condition p<0.12n
    if (0.12*n > p):
        s *= (n-1)/(n-p-2)
    # rho is block invariant
    mu = (np.trace(s)/p)
    alpha = np.mean(s**2)
    numerator = alpha+mu**2
    denominator = (n+1.)*(alpha-(mu**2)/p)
    rho = 1. if denominator == 0 else min(1., numerator/denominator)
    return (1.-rho)*s+np.kron(np.eye(p//bsize),np.ones((bsize,bsize)))*rho*mu


def jkncov(sample):
    """
    empirical covariance matrix estimation
    + jackknife correction
    + conditional Hartlap correction

    Parameters
    ---------
    sample:
        input sample
    """
    assert isinstance(sample,np.ndarray)
    n,p = sample.shape
    assert (n>1 and p>0)
    # empirical
    u = sample-np.mean(sample,axis=0)
    s = n*np.dot(u.T,u)/(n-1)
    # jackknife
    for i in range(n):
        s -= np.cov(np.delete(sample,i,0),rowvar=False)*(n-1)/n
    # with Hartlap correction under condition p<0.12n
    if (0.12*n > p):
        s *= (n-1)/(n-p-2)
    return s


def umap(x, r=[0.,1.]):
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


def gvec(cps):
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
    dof_arr = np.arange(dof)
    triu_idx = np.triu_indices(nfreq)
    if (cps.ndim == 4):  # single sample
        rslt = np.zeros(ntype*nmode*dof)
        for t in range(ntype):
            for l in range(nmode):
                rslt[(t*nmode+l)*dof:(t*nmode+l+1)*dof] = cps[t,l][triu_idx]  # type leading
        return rslt
    elif (cps.ndim == 5):  # ensemble
        nsamp = cps.shape[0]
        rslt = np.zeros((nsamp,ntype*nmode*dof))
        for s in range(nsamp):
            for t in range(ntype):
                for l in range(nmode):
                    rslt[s,(t*nmode+l)*dof:(t*nmode+l+1)*dof] = cps[s,t,l][triu_idx]  # type leading
        return rslt
    else:
        raise ValueError('unsupported input shape')


def hvec(cps,cps_hat,cps_fid):
    """with measured cps_hat, fiducial cps_fid, modeled cps
    """
    assert (cps.ndim == 4)
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
            res = np.matmul(np.conjugate(c_inv), np.matmul(c_h, c_inv))
            d,u = np.linalg.eigh(res)
            assert (any(d>=0))
            #if (any(d<0)):
            #    rslt[(t*nmode+l)*dof:(t*nmode+l+1)*dof] *= np.nan_to_num(np.inf)
            gd = np.diag(np.sign(d-1.)*np.sqrt(2.*(d-np.log(d)-1.)))
            x = np.matmul(gd, np.matmul(np.transpose(u),c_f))
            x = np.matmul(np.conjugate(c_f), np.matmul(np.conjugate(u),x))
            rslt[(t*nmode+l)*dof:(t*nmode+l+1)*dof] = x[triu_idx]  # type leading
    return rslt
