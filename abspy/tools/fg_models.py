"""
Galactic foreground models

- syncmodel
    synchrotron cross-correlation band power
    
- dustmodel
    thermal dust cross-correlation band power
    
- syncdustmodel
    synchrotron-dust correlation contribution band power
"""

import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy


@icy
class fgmodel(object):

    def __init__(self, modes=list(), refs=None):
        self.modes = modes
        self.refs = refs
        self._params = dict()
        
    @property
    def modes(self):
        """angular mode list"""
        return self._modes
    
    @property
    def refs(self):
        return self._refs
        
    @property
    def params(self):
        return self._params
        
    @modes.setter
    def modes(self, modes):
        assert isinstance(modes, list)
        self._modes = modes
    
    @refs.setter
    def refs(self, refs):
        self._refs = refs
        
    def reset(self, pdict):
        """(re)set parameters"""
        assert isinstance(pdict, dict)
        for name in pdict.keys():
            if name in self.param_list:
                self._params[name] = pdict[name]


@icy
class syncmodel(fgmodel):
    
    def __init__(self, modes=list(), refs=30.):
        super(syncmodel, self).__init__(modes,refs)
    
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_s_x" amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for i in range(len(self._modes)):
            plist.append('bp_s_'+str(self._modes[i]))
        plist.append('beta_s')
        return plist
        
    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        for i in _tmp[:-1]:
            prange[i] = [0.,1.e+4]
        prange['beta_s'] = [-5.,0.]
        return prange
    
    def bandpower(self, freqs):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freqs : float
            list of frequency in GHz
        """
        assert isinstance(freqs, (list,tuple))
        assert isinstance(self._refs, float)
        rslt = np.zeros((len(self._modes),len(freqs),len(freqs)))
        beta_val_s = self.params['beta_s']
        for l in range(len(self._modes)):
            bp_s = self.params['bp_s_'+str(self._modes[l])]
            for i in range(len(freqs)):
                rslt[l,i,i] = bp_s * (freqs[i]*freqs[i]/ self._refs**2)**beta_val_s
                for j in range(i+1,len(freqs)):
                    tmp = bp_s * (freqs[i]*freqs[j]/ self._refs**2)**beta_val_s
                    rslt[l,i,j] = tmp
                    rslt[l,j,i] = tmp
        return rslt


@icy
class dustmodel(fgmodel):
    
    def __init__(self, modes=list(), refs=353.):
        super(dustmodel, self).__init__(modes,refs)
    
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_s_x" amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for i in range(len(self._modes)):
            plist.append('bp_d_'+str(self._modes[i]))
        plist.append('beta_d')
        return plist
        
    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        for i in _tmp[:-1]:
            prange[i] = [0.,1.e+4]
        prange['beta_d'] = [0.,5.]
        return prange
        
    def bandpower(self, freqs):
        """synchrotron model cross-(frequency)-power-spectrum
        in shape (ell #, freq #, freq #)
        
        Parameters
        ----------
            
        freqs : float
            list of frequency in GHz
        """
        assert isinstance(freqs, (list,tuple))
        assert isinstance(self._refs, float)
        def Bratio(_freq):
            _td = 19.6  # thermal dust temp
            _hGk = 0.04799340833334541  # h*GHz/k_B
            return (np.exp(_hGk*self._refs/_td)-1.)/(np.exp(_hGk*_freq/_td)-1.)
        rslt = np.zeros((len(self._modes),len(freqs),len(freqs)))
        beta_val_d = self.params['beta_d']
        for l in range(len(self._modes)):
            bp_d = self.params['bp_d_'+str(self._modes[l])]
            for i in range(len(freqs)):
                rslt[l,i,i] = bp_d * (freqs[i]*freqs[i]/ self._refs**2)**(beta_val_d+1) * Bratio(freqs[i]) * Bratio(freqs[i])
                for j in range(i+1,len(freqs)):
                    tmp = bp_d * (freqs[i]*freqs[j]/ self._refs**2)**(beta_val_d+1) * Bratio(freqs[i]) * Bratio(freqs[j])
                    rslt[l,i,j] = tmp
                    rslt[l,j,i] = tmp
        return rslt


@icy
class syncdustmodel(fgmodel):
    
    def __init__(self, modes=list(), refs=[30.,353.]):
        super(syncdustmodel, self).__init__(modes,refs)
    
    @property
    def param_list(self):
        """parameters are set as
        - bandpower "bp_s_x" amplitude at reference frequency x
        - bandpower frequency scaling parameters
        """
        plist = list()
        for i in range(len(self._modes)):
            plist.append('bp_s_'+str(self._modes[i]))
            plist.append('bp_d_'+str(self._modes[i]))
        plist.append('beta_s')
        plist.append('beta_d')
        plist.append('rho')
        return plist
        
    @property
    def param_range(self):
        """parameter sampling range,
        in python dict
        {param name : [low limit, high limit]
        """
        prange = dict()
        _tmp = self.param_list
        for i in _tmp[:-3]:
            prange[i] = [0.,1.e+4]
        prange['beta_s'] = [-5.,0.]
        prange['beta_d'] = [0.,5.]
        prange['rho'] = [-1.,1.]
        return prange
    
    def bandpower(self, freqs):
        """synchrotron, dust + cross corr power-spectrum
        
        Parameters
        ----------
        
        freqs : float
            list of frequency in GHz
        """
        assert isinstance(freqs, (list,tuple))
        assert (len(self._refs) == 2)
        assert (self._refs[0] < self._refs[1])
        def Bratio(_freq):
            _td = 19.6  # thermal dust temp
            _hGk = 0.04799340833334541  # h*GHz/k_B
            return (np.exp(_hGk*self._refs[1]/_td)-1.)/(np.exp(_hGk*_freq/_td)-1.)
        rslt = np.zeros((len(self._modes),len(freqs),len(freqs)))
        beta_val_s = self.params['beta_s']
        beta_val_d = self.params['beta_d']
        rho_val = self.params['rho']
        for l in range(len(self._modes)):
            bp_s = self.params['bp_s_'+str(self._modes[l])]
            bp_d = self.params['bp_d_'+str(self._modes[l])]
            for i in range(len(freqs)):
                rslt[l,i,i] = bp_s * (freqs[i]*freqs[i]/self._refs[0]**2)**beta_val_s
                rslt[l,i,i] += bp_d * (freqs[i]*freqs[i]/self._refs[1]**2)**(beta_val_d+1)
                rslt[l,i,i] += rho_val * np.sqrt(bp_s*bp_d) * ( (freqs[i]/self._refs[0])**beta_val_s * (freqs[i]/self.refs[1])**(beta_val_d+1) * Bratio(freqs[i]) +  (freqs[i]/self.refs[0])**beta_val_s * (freqs[i]/self.refs[1])**(beta_val_d+1) * Bratio(freqs[i]))
                for j in range(i+1,len(freqs)):
                    tmp = bp_s * (freqs[i]*freqs[j]/self._refs[0]**2)**beta_val_s
                    tmp += bp_d * (freqs[i]*freqs[j]/self._refs[1]**2)**(beta_val_d+1)
                    tmp += rho_val * np.sqrt(bp_s*bp_d) * ( (freqs[i]/self._refs[0])**beta_val_s * (freqs[j]/self.refs[1])**(beta_val_d+1) * Bratio(freqs[j]) +  (freqs[j]/self.refs[0])**beta_val_s * (freqs[i]/self.refs[1])**(beta_val_d+1) * Bratio(freqs[i]))
                    rslt[l,i,j] = tmp
                    rslt[l,j,i] = tmp
        return rslt
