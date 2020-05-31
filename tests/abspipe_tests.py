import unittest
import numpy as np
from abspy import abspipe


class TestABSpipe(unittest.TestCase):
    
    def test_init(self):
        test_signal = np.random.rand(4,1,192)
        test_pipe = abspipe(test_signal,
                            nfreq=4,
                            nmap=1,
                            nside=4)
        self.assertEqual(test_pipe.nfreq, 4)
        self.assertEqual(test_pipe.nmap, 1)
        self.assertEqual(test_pipe.nside, 4)
        self.assertEqual(test_pipe._npix, 192)
        
    def test_sainity_Tmap(self):
        np.random.seed(234)
        test_signal = np.random.rand(4,1,192)
        test_pipe = abspipe(test_signal,
                            nfreq=4,
                            nmap=1,
                            nside=4)
        test_result = test_pipe(psbin=1)
        
    def test_sainity_QUmap(self):
        np.random.seed(234)
        test_signal = np.random.rand(4,2,192)
        test_pipe = abspipe(test_signal,
                            nfreq=4,
                            nmap=2,
                            nside=4)
        test_result = test_pipe(psbin=1)

if __name__ == '__main__':
    unittest.main()
