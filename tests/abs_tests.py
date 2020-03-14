import unittest
import numpy as np
from abspy.methods.abs import abssep

class TestSeparator(unittest.TestCase):
    
    def test_init(self):
        test_signal = np.random.rand(192,3,3)
        test_noise = np.random.rand(192,3,3)
        test_sigma = np.random.rand(192,3)
        test_bins = 3
        test_sep = abssep(test_signal,
                          test_noise,
                          test_sigma,
                          test_bins)
        for i in range(test_signal.shape[0]):
            self.assertListEqual(list(test_sep.sigma[i]), list(test_sigma[i]))
            for j in range(test_signal.shape[1]):
                self.assertEqual(list(test_sep.signal[i,j]), list(test_signal[i,j]))
                self.assertEqual(list(test_sep.noise[i,j]), list(test_noise[i,j]))
        self.assertEqual(test_sep._lsize, test_signal.shape[0])
        self.assertEqual(test_sep._fsize, test_signal.shape[1])
        self.assertEqual(test_sep.bins, test_bins)
        self.assertEqual(test_sep.shift, 0.0)
        self.assertEqual(test_sep.threshold, 0.0)
        #
        test_shift = 23.0
        test_threshold = 2.3
        test_sep = abssep(test_signal,
                          test_noise,
                          test_sigma,
                          bins=test_bins,
                          modes=None,
                          shift=test_shift,
                          threshold=test_threshold)
        self.assertEqual(test_sep.shift, test_shift)
        self.assertEqual(test_sep.threshold, test_threshold)
        #
        test_modes = [*range(2,16,2)]
        test_signal = np.random.rand(7,3,3)
        test_noise = np.random.rand(7,3,3)
        test_sigma = np.random.rand(7,3)
        test_sep = abssep(test_signal,
                          test_noise,
                          test_sigma,
                          bins=test_bins,
                          modes=test_modes,
                          shift=test_shift,
                          threshold=test_threshold)
        self.assertListEqual(test_sep.binell, [4., 9., 13.])
    
    def test_binning(self):
        test_ccl = np.random.rand(10,3,3)
        test_ccl_noise = np.random.rand(10,3,3)
        test_ccl_sigma = np.random.rand(10,3)
        binsize = 3
        test_sep = abssep(test_ccl,
                          test_ccl_noise,
                          test_ccl_sigma,
                          binsize)
        test_ell = test_sep.binell
        self.assertListEqual(test_ell, [1.5,5.,8.])
        test_cdl = test_sep.bincps(test_ccl)
        test_ndl = test_sep.bincps(test_ccl_noise)
        test_rdl = test_sep.binaps(test_ccl_sigma)
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
    
    def test_sainity(self):
        np.random.seed(234)
        test_ccl = np.random.rand(128,3,3)
        test_ccl_noise = np.random.rand(128,3,3)*0.01
        test_ccl_sigma = np.random.rand(128,3)*0.001
        binsize = 3
        test_sep = abssep(test_ccl,
                          test_ccl_noise,
                          test_ccl_sigma,
                          binsize)
        test_result = test_sep()

if __name__ == '__main__':
    unittest.main()
