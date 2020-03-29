import healpy as hp
import numpy as np

class PurifyB(object):
   
    
    def __init__(self, maps, mask):
         
        '''
        Parameters
        ----------
        
        mask : binary, 1 and 0;
        maps: IQU maps; The Mask_01() and hp.map2alm() require IQU as input.
        
        Example
        ----------
        
        purify = PurifyB(cmb_i, ali_ma_512)
        map_purified = purify.lin_resi()
        '''
        
        self.mask = mask; self.maps = maps; self.nside = hp.npix2nside(len(mask))
        self.pix_list = np.arange(len(mask)); self.mask_sum = sum(mask)
        self.mask_index = self.pix_list[np.where(mask < 1)] # the pixel index of the masked index
        self.avai_index = self.pix_list[np.where(mask == 1)] # the pixel index of the available index
        
    def Mask_01(self, maps_raw):
    
        '''
        Mask the maps. The masked values are equal to 0.
        maps_raw: the maps to be masked, the shape of which must be (Nf, 3, npix), (3, npix), or (npix).

        '''
        _maps = np.copy(maps_raw)

        _ndim = len(_maps.shape)

        if _ndim > 2:  ### (Nf, 3, npix)
            for i in range(_maps.shape[0]):
                for j in range(3):
                    _maps[i,j][self.mask_index] = 0
        elif _ndim == 2: ### (3, npix)
            for j in range(_maps.shape[0]):
                _maps[j][self.mask_index] = 0

        else: ### (npix)
            _maps[self.mask_index] = 0
        
        return _maps

    def lin_resi(self):
        
        '''
        Main function of this class to correct the E to B leakage.
        '''
        
        ### get the template of E to B leakage 
        
        alm_ma = hp.map2alm(self.Mask_01(self.maps)) #alms of the masked maps

        B0 = hp.alm2map(alm_ma[2], nside = self.nside, verbose = False) # corrupted B map
        
        alm_ma[0] = 0; alm_ma[2] = 0; map_E = hp.alm2map(alm_ma, nside = self.nside, verbose = False) # IQU of corrupted E mode only

        alm_new = hp.map2alm(self.Mask_01(map_E)) # alms of the IUQ from only E-mode 

        BT = hp.alm2map(alm_new[2], nside = self.nside, verbose = False) # template of E to B leakage 
        
        ### compute the residual of linear fit
        
        x = BT[self.avai_index]; y = B0[self.avai_index]
        
        mx  = sum(x)/self.mask_sum; my  = sum(y)/self.mask_sum;
        cxx = sum((x-mx)*(x-mx)); cxy = sum((y-my)*(x-mx))
        a1  = cxy/cxx 
        a0  = my - mx*a1 
        resi  = y - a0 - a1*x

        map0 = np.zeros(12*self.nside**2);
        map0[self.avai_index] = resi

        return map0
