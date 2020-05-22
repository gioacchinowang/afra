"""
Galactic foreground models

- fgmodel::syncmodel
    synchrotron cross-correlation band power
    
- fgmodel::dustmodel
    thermal dust cross-correlation band power
    
- fdmodel::syncdustmodel
    synchrotron-dust correlation contribution band power
"""

import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy


@icy
class syncmodel(object):
    
    def __init__(self, ellist=list(), freq_ref=30.):
        self.ellist = ellist
        self._params = dict()
        # default reference frequency
        self.freq_ref = freq_ref  # GHz
    
    @property
    def ellist(self):
        """angular mode list"""
        return self._ellist
    
    @property
    def freq_ref(self):
        return self._freq_ref
        
    @property
    def params(self):
        return self._params
    
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "amp_s_x" amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for i in range(len(self._ellist)):
            plist.append('amp_s_'+str(self._ellist[i]))
        plist.append('beta_s')
        return plist
        
    @ellist.setter
    def ellist(self, ellist):
        assert isinstance(ellist, list)
        self._ellist = ellist
    
    @freq_ref.setter
    def freq_ref(self, freq_ref):
        assert isinstance(freq_ref, float)
        self._freq_ref = freq_ref
    
    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params[name] = pdict[name]
    
    def bandpower(self, freq_list):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freq_list : float
            list of frequency in GHz
            
        freq_ref : float
            synchrotron template reference frequency
        """
        assert isinstance(freq_list, (list,tuple))
        rslt = np.zeros((len(self._ellist),len(freq_list),len(freq_list)))
        beta_val_s = self.params['beta_s']
        for l in range(len(self._ellist)):
            amp_val_s = self.params['amp_s_'+str(self._ellist[l])]
            for i in range(len(freq_list)):
                rslt[l,i,i] = amp_val_s * (freq_list[i]*freq_list[i]/ self._freq_ref**2)**beta_val_s
                for j in range(i+1,len(freq_list)):
                    tmp = amp_val_s * (freq_list[i]*freq_list[j]/ self._freq_ref**2)**beta_val_s
                    rslt[l,i,j] = tmp
                    rslt[l,j,i] = tmp
        return rslt


@icy
class dustmodel(object):
    
    def __init__(self, ellist=list(), freq_ref=353.):
        self.ellist = ellist
        self._params = dict()
        # default reference frequency
        self.freq_ref = freq_ref  # GHz
        
    @property
    def ellist(self):
        """angular mode list"""
        return self._ellist
        
    @property
    def freq_ref(self):
        return self._freq_ref
        
    @property
    def params(self):
        return self._params
        
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "amp_s_x" amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for i in range(len(self._ellist)):
            plist.append('amp_d_'+str(self._ellist[i]))
        plist.append('beta_d')
        return plist
        
    @ellist.setter
    def ellist(self, ellist):
        assert isinstance(ellist, list)
        self._ellist = ellist
        
    @freq_ref.setter
    def freq_ref(self, freq_ref):
        assert isinstance(freq_ref, float)
        self._freq_ref = freq_ref
    
    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params[name] = pdict[name]
        
    def bandpower(self, freq_list):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freq_list : float
            list of frequency in GHz
            
        freq_ref : float
            synchrotron template reference frequency
        """
        assert isinstance(freq_list, (list,tuple))
        def Bratio(_freq):
            _td = 19.6  # thermal dust temp
            _hGk = 0.04799340833334541  # h*GHz/k_B
            return (np.exp(_hGk*self._freq_ref/_td)-1.)/(np.exp(_hGk*_freq/_td)-1.)
        rslt = np.zeros((len(self._ellist),len(freq_list),len(freq_list)))
        beta_val_d = self.params['beta_d']
        for l in range(len(self._ellist)):
            amp_val_d = self.params['amp_d_'+str(self._ellist[l])]
            for i in range(len(freq_list)):
                rslt[l,i,i] = amp_val_d * (freq_list[i]*freq_list[i]/ self._freq_ref**2)**(beta_val_d+1) * Bratio(freq_list[i]) * Bratio(freq_list[i])
                for j in range(i+1,len(freq_list)):
                    tmp = amp_val_d * (freq_list[i]*freq_list[j]/ self._freq_ref**2)**(beta_val_d+1) * Bratio(freq_list[i]) * Bratio(freq_list[j])
                    rslt[l,i,j] = tmp
                    rslt[l,j,i] = tmp
        return rslt


@icy
class syncdustmodel(object):
    
    def __init__(self, ellist=list(), freq_ref=[30.,353.]):
        self.ellist = ellist
        self._params = dict()
        # default reference frequency
        self.freq_ref = freq_ref  # GHz
        
    @property
    def ellist(self):
        """angular mode list"""
        return self._ellist
        
    @property
    def freq_ref(self):
        return self._freq_ref
        
    @property
    def params(self):
        return self._params
        
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "amp_s_x" amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for i in range(len(self._ellist)):
            plist.append('amp_s_'+str(self._ellist[i]))
            plist.append('amp_d_'+str(self._ellist[i]))
        plist.append('beta_s')
        plist.append('beta_d')
        plist.append('rho')
        return plist
        
    @ellist.setter
    def ellist(self, ellist):
        assert isinstance(ellist, list)
        self._ellist = ellist
    
    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params[name] = pdict[name]
    
    @freq_ref.setter
    def freq_ref(self, freq_ref):
        assert (len(freq_ref) == 2)  # need two refernece frequencies
        assert (freq_ref[0] < freq_ref[1])  # 1st for sync, 2nd for dust
        self._freq_ref = freq_ref
        
    def bandpower(self, freq_list):
        """synchrotron, dust + cross corr power-spectrum
        
        Parameters
        ----------
        
        freq1 : float
            frequency in GHz
        
        freq2 : float
            frequency in GHz
        """
        assert isinstance(freq_list, (list,tuple))
        def Bratio(_freq):
            _td = 19.6  # thermal dust temp
            _hGk = 0.04799340833334541  # h*GHz/k_B
            return (np.exp(_hGk*self._freq_ref[1]/_td)-1.)/(np.exp(_hGk*_freq/_td)-1.)
        rslt = np.zeros((len(self._ellist),len(freq_list),len(freq_list)))
        beta_val_s = self.params['beta_s']
        beta_val_d = self.params['beta_d']
        rho_val = self.params['rho']
        for l in range(len(self._ellist)):
            amp_val_s = self.params['amp_s_'+str(self._ellist[l])]
            amp_val_d = self.params['amp_d_'+str(self._ellist[l])]
            for i in range(len(freq_list)):
                rslt[l,i,i] = amp_val_s * (freq_list[i]*freq_list[i]/self._freq_ref[0]**2)**beta_val_s
                rslt[l,i,i] += amp_val_d * (freq_list[i]*freq_list[i]/self._freq_ref[1]**2)**(beta_val_d+1)
                rslt[l,i,i] += rho_val * np.sqrt(amp_val_s*amp_val_d) * ( (freq_list[i]/self._freq_ref[0])**beta_val_s * (freq_list[i]/self.freq_ref[1])**(beta_val_d+1) * Bratio(freq_list[i]) +  (freq_list[i]/self.freq_ref[0])**beta_val_s * (freq_list[i]/self.freq_ref[1])**(beta_val_d+1) * Bratio(freq_list[i]))
                for j in range(i+1,len(freq_list)):
                    tmp = amp_val_s * (freq_list[i]*freq_list[j]/self._freq_ref[0]**2)**beta_val_s
                    tmp += amp_val_d * (freq_list[i]*freq_list[j]/self._freq_ref[1]**2)**(beta_val_d+1)
                    tmp += rho_val * np.sqrt(amp_val_s*amp_val_d) * ( (freq_list[i]/self._freq_ref[0])**beta_val_s * (freq_list[j]/self.freq_ref[1])**(beta_val_d+1) * Bratio(freq_list[j]) +  (freq_list[j]/self.freq_ref[0])**beta_val_s * (freq_list[i]/self.freq_ref[1])**(beta_val_d+1) * Bratio(freq_list[i]))
                    rslt[l,i,j] = tmp
                    rslt[l,j,i] = tmp
        return rslt
