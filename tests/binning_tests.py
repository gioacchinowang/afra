import unittest
import numpy as np
from abspy import binell, bincps, binaps


class TestBinning(unittest.TestCase):

    def test_binning(self):
        test_ccl = np.random.rand(10,3,3)
        test_ccl_noise = np.random.rand(10,3,3)
        test_ccl_sigma = np.random.rand(10,3)
        modes = [*range(0,10)]
        binsize = 3
        #
        test_ell = binell(modes,binsize)
        self.assertListEqual(test_ell, [1.5,5.,8.])
        test_cdl = bincps(test_ccl,modes,binsize)
        test_ndl = bincps(test_ccl_noise,modes,binsize)
        test_rdl = binaps(test_ccl_sigma,modes,binsize)
        # calculate manually
        check_cdl = np.empty((binsize,test_ccl.shape[1],test_ccl.shape[2]))
        check_ndl = np.empty((binsize,test_ccl.shape[1],test_ccl.shape[2]))
        check_rdl = np.empty((binsize,test_ccl.shape[1]))
        facto = [0.5*1.5*(1.5+1)/np.pi, 0.5*5*(5+1)/np.pi, 0.5*8*(8+1)/np.pi]
        for i in range(test_ndl.shape[1]):
            check_rdl[:,i] = [np.mean(test_ccl_sigma[0:4,i])*facto[0],
                     np.mean(test_ccl_sigma[4:7,i])*facto[1],
                     np.mean(test_ccl_sigma[7:10,i])*facto[2]]
            for j in range(test_cdl.shape[2]):
                check_cdl[:,i,j] = [np.mean(test_ccl[0:4,i,j])*facto[0],
                         np.mean(test_ccl[4:7,i,j])*facto[1],
                         np.mean(test_ccl[7:10,i,j])*facto[2]]
                check_ndl[:,i,j] = [np.mean(test_ccl_noise[0:4,i,j])*facto[0],
                         np.mean(test_ccl_noise[4:7,i,j])*facto[1],
                         np.mean(test_ccl_noise[7:10,i,j])*facto[2]]
        for i in range(check_cdl.shape[0]):
            for k in range(check_cdl.shape[2]):
                self.assertAlmostEqual(test_rdl[i,k], check_rdl[i,k])
                for j in range(check_cdl.shape[1]):
                    self.assertAlmostEqual(test_cdl[i,j,k], check_cdl[i,j,k])
                    self.assertAlmostEqual(test_ndl[i,j,k], check_ndl[i,j,k])

if __name__ == '__main__':
    unittest.main()
