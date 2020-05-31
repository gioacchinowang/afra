import unittest
import numpy as np
from abspy import abssep


class TestSeparator(unittest.TestCase):
    
    def test_init(self):
        test_signal = np.random.rand(5,3,3)
        test_noise = np.random.rand(5,3,3)
        test_sigma = np.random.rand(5,3)
        test_sep = abssep(test_signal,
                          test_noise,
                          test_sigma)
        for i in range(test_signal.shape[0]):
            self.assertListEqual(list(test_sep.sigma[i]), list(test_sigma[i]))
            for j in range(test_signal.shape[1]):
                self.assertEqual(list(test_sep.signal[i,j]), list(test_signal[i,j]))
                self.assertEqual(list(test_sep.noise[i,j]), list(test_noise[i,j]))
        self.assertEqual(test_sep._lsize, test_signal.shape[0])
        self.assertEqual(test_sep._fsize, test_signal.shape[1])
        self.assertListEqual(list(test_sep.shift), [0.]*5)
        self.assertEqual(test_sep.threshold, None)
        #
        test_shift = np.random.rand(5)
        test_threshold = 2.3
        test_sep = abssep(test_signal,
                          test_noise,
                          test_sigma,
                          shift=test_shift,
                          threshold=test_threshold)
        self.assertListEqual(list(test_sep.shift), list(test_shift))
        self.assertEqual(test_sep.threshold, test_threshold)
    
    def test_sainity(self):
        np.random.seed(234)
        test_ccl = np.random.rand(5,3,3)
        test_ccl_noise = np.random.rand(5,3,3)*0.01
        test_ccl_sigma = np.random.rand(5,3)*0.001
        test_sep = abssep(test_ccl,
                          test_ccl_noise,
                          test_ccl_sigma)
        test_result = test_sep.run()

if __name__ == '__main__':
    unittest.main()
