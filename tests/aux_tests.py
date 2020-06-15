import unittest
import numpy as np
from abspy.tools.aux import *


class TestBinning(unittest.TestCase):

    def test_vecp(self):
        input_cps = np.random.rand(2,3,3)
        output_vecp = vecp(input_cps)
        manual_vecp = np.zeros(12)
        tmp = np.triu(input_cps[0])
        manual_vecp[:6] = tmp[tmp!=0]
        tmp = np.triu(input_cps[1])
        manual_vecp[6:] = tmp[tmp!=0]
        self.assertListEqual(list(output_vecp),list(manual_vecp))
		
    def test_binning(self):
        # prepare D_ell
        test_cdl = np.random.rand(10,3,3)
        test_cdl_noise = np.random.rand(10,3,3)
        test_cdl_sigma = np.random.rand(10,3)
        modes = [*range(2,12)]
        binsize = 3
        # calculate with functions
        binned_ell = binell(modes,binsize)
        self.assertListEqual(binned_ell, [3.5,7.,10.])
        binned_cdl = bincps(test_cdl,modes,binsize)
        binned_ndl = bincps(test_cdl_noise,modes,binsize)
        binned_rdl = binaps(test_cdl_sigma,modes,binsize)
        # calculate manually
        # convert into C_ell
        for i in range(len(modes)):
            scale = 0.5*modes[i]*(modes[i]+1)/np.pi
            test_cdl[i] /= scale
            test_cdl_noise[i] /= scale
            test_cdl_sigma[i] /= scale
        check_cdl = np.empty((binsize,test_cdl.shape[1],test_cdl.shape[2]))
        check_ndl = np.empty((binsize,test_cdl.shape[1],test_cdl.shape[2]))
        check_rdl = np.empty((binsize,test_cdl.shape[1]))
        facto = [0.5*3.5*(3.5+1)/np.pi, 0.5*7*(7+1)/np.pi, 0.5*10*(10+1)/np.pi]
        for i in range(3):
            check_rdl[:,i] = [np.mean(test_cdl_sigma[0:4,i])*facto[0],
                     np.mean(test_cdl_sigma[4:7,i])*facto[1],
                     np.mean(test_cdl_sigma[7:10,i])*facto[2]]
            for j in range(3):
                check_cdl[:,i,j] = [np.mean(test_cdl[0:4,i,j])*facto[0],
                         np.mean(test_cdl[4:7,i,j])*facto[1],
                         np.mean(test_cdl[7:10,i,j])*facto[2]]
                check_ndl[:,i,j] = [np.mean(test_cdl_noise[0:4,i,j])*facto[0],
                         np.mean(test_cdl_noise[4:7,i,j])*facto[1],
                         np.mean(test_cdl_noise[7:10,i,j])*facto[2]]
        # comparison
        for i in range(check_cdl.shape[0]):
            for k in range(check_cdl.shape[2]):
                self.assertAlmostEqual(binned_rdl[i,k], check_rdl[i,k])
                for j in range(check_cdl.shape[1]):
                    self.assertAlmostEqual(binned_cdl[i,j,k], check_cdl[i,j,k])
                    self.assertAlmostEqual(binned_ndl[i,j,k], check_ndl[i,j,k])
		
	
if __name__ == '__main__':
    unittest.main()
