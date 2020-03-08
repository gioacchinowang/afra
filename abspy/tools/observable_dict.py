"""
For convenience we define dictionary of data as
ObservableDict from which one can define define the classes 
Measurements and Masks, which can be used to store:

    * measured data sets
    * mask "maps" (but actually mask lists)

Conventions for observables entries
    * **Synchrotron emission**: `('sync',str(freq),str(Healpix-Nside),X)`
        where X stands for:
            * 'I' - total intensity (in unit K-cmb)
            * 'Q' - Stokes Q (in unit K-cmb, IAU convention)
            * 'U' - Stokes U (in unit K-cmb, IAU convention)

    Remarks:
        * `str(freq)`, polarisation-related-flag are redundant for
        non-CMB type data so we put 'nan' instead
        * `str(nside)` stores either Healpix Nisde
        
Masking convention
    fully masked erea associated with pixel value 0
    fully unmasked area with pixel value 1
"""

import numpy as np
from copy import deepcopy
import logging as log
from abspy.tools.icy_decorator import icy


@icy
class ObservableDict(object):
    
    def __init__(self):
        self._archive = dict()

    @property
    def archive(self):
        return self._archive

    def keys(self):
        return self._archive.keys()

    def __getitem__(self, key):
        return self._archive[key]

    def append(self, name, data):
        """
        Adds/updates name and data.

        Parameters
        ----------
        
        name : string tuple
            Should follow the convention:
            ``(data-name,str(data-freq),str(HEALPix-Nside),str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U' or other customized tags.
            
        data : list, tuple, numpy.ndarray
            Distributed/copied ndarray/Observable.
            
        plain : bool
            If True, means unstructured data.
            If False (default case), means HEALPix-like sky map.
        """
        pass

    def apply_mask(self, mask_dict):
        """
        Parameters
        ----------
        
        mask_dict : ObservableDict.Masks
            Masks object.
        """
        pass


@icy
class Masks(ObservableDict):
    
    def __init__(self):
        super(Masks, self).__init__()

    def append(self, name, new, plain=False):
        """
        Adds/updates name and data.

        Parameters
        ----------
        
        name : str tuple
            Should follow the convention:
            ``(data-name, str(data-freq), str(HEALPix-Nside), str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U' or other customized tags.
            
        data : list, tuple, numpy.ndarray
            A Healpix map to be appended.
        """
        log.debug('@ ObservableDict::Masks::append')
        assert (len(name) == 4)
        assert isinstance(new, (list,tuple,np.ndarray))
        assert (len(new) == 12*np.uint(name[2])**2)
        self._archive.update({name: np.array(new)})


@icy
class Measurements(ObservableDict):
    
    def __init__(self):
        super(Measurements, self).__init__()

    def append(self, name, new):
        """
        Adds/updates name and data

        Parameters
        ----------
        
        name : str tuple
            Should follow the convention:
            ``(data-name, str(data-freq), str(HEALPix-Nside), str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U' or other customized tags.
            
        data : list, tuple, numpy.ndarray
            Healpix map to be appended
        """
        log.debug('@ ObservableDict::Measurements::append')
        assert (len(name) == 4)
        assert isinstance(new, (list,tuple,np.ndarray))
        assert (len(new) == 12*np.uint(name[2])**2)
        self._archive.update({name: np.array(new)})
    
    def apply_mask(self, mask_dict=None):
        """
        Applying mask, by matching name tags of masks and measurements.
        
        Parameters
        ----------
        
        mask_dict : ObservableDict.Masks
            Masks object.
        """
        log.debug('@ ObservableDict::Measurements::apply_mask')
        if mask_dict is None:
            pass
        else:
            assert isinstance(mask_dict, Masks)
            for name, msk in mask_dict._archive.items():
                if name in self._archive.keys():
                    masked = deepcopy(self._archive[name])
                    for ptr in range(len(msk)):
                        masked[ptr] *= msk[ptr]
                    self._archive.update({name: masked})


@icy
class Covariances(ObservableDict):
    
    def __init__(self):
        super(Covariances, self).__init__()
        
    def append(self, name, new):
        """
        Adds/updates name and data
        
        Parameters
        ----------
        
        name : str tuple
            Should follow the convention:
            ``(data-name, str(data-freq), str(HEALPix-Nside), _str(ext))``.
            If data is independent from frequency, set 'nan'.
            `ext` can be 'I','Q','U' or other customized tags.
            
        new : numpy.ndarray
            A covariance matrix to be appended.
        """
        log.debug('@ ObservableDict::Covariances::append')
        assert (len(name) == 4)
        assert isinstance(new, np.ndarray)
        assert (len(new.shape) == 2)
        assert (new.shape[0] == new.shape[1])
        assert (new.shape[0] == 12*np.uint(name[2])**2)
        self._archive.update({name: np.array(new)})
        
    def apply_mask(self, mask_dict=None):
        """
        Applying mask, by matching name tags of masks and covariance matrices.
        
        Parameters
        ----------
        
        mask_dict : ObservableDict.Masks
            Masks object.
        """
        log.debug('@ ObservableDict::Covariances::apply_mask')
        if mask_dict is None:
            pass
        else:
            for name, msk in mask_dict._archive.items():
                if name in self._archive.keys():
                    masked = deepcopy(self._archive[name])
                    for ptr in range(len(msk)):
                        masked[:,ptr] *= msk[ptr]
                        masked[ptr,:] *= msk[ptr]
                    self._archive.update({name: masked})
                    
    def apply_mask_drct(self, name, msk):
        """
        An alternative way of applying mask,
        by deirectly assign the mask to the specific covariance matrix,
        instead of by matching name tags.
        
        Parameters
        ----------
        
        name : string tuple
            ObservableDict name tag.
        
        msk : numpy.ndarray
            The mask map to be applied.
        """
        log.debug('@ observabld_dict::Covariances::apply_mask_drct')
        masked = deepcopy(self._archive[name])
        for ptr in range(len(msk)):
            masked[:,ptr] *= msk[ptr]
            masked[ptr,:] *= msk[ptr]
        self._archive.update({name: masked})