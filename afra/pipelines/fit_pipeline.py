import numpy as np
from afra.pipelines.pipeline import pipe
from afra.methods.fit import *
from afra.tools.icy_decorator import icy


@icy
class fitpipe(pipe):
    
    def __init__(self, data, noises=None, mask=None, beams=None, targets='T',
                 fiducials=None, fiducial_beams=None,
                 templates=None, template_noises=None, template_beams=None,
                 foreground=None, background=None,
                 likelihood='gauss', filt=None):
        super(fitpipe, self).__init__(data,noises,mask,beams,targets,fiducials,fiducial_beams,templates,template_noises,template_beams,foreground,background,likelihood,filt)
        # analyse select dict
        self._anadict = {'gauss':self.analyse_gauss, 'hl':self.analyse_hl}
        # Bayesian engine to be assigned
        self.engine = None

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        if engine is not None:
            assert isinstance(engine, fit)
        self._engine = engine

    def run(self, aposcale=6., psbin=20, lmin=None, lmax=None, kwargs=dict()):
        self.preprocess(aposcale,psbin,lmin,lmax)
        return self.analyse(kwargs)

    def analyse(self, kwargs=dict()):
        return self._anadict[self._likelihood](kwargs)

    def analyse_gauss(self, kwargs=dict()):
        # gauss likelihood
        self.engine = gaussfit(self._data_bp,np.mean(self._fiducial_bp,axis=0),np.mean(self._noise_bp,axis=0),self._covmat,self._background_obj,self._foreground_obj)
        if (len(self._paramrange)):
            self._engine.rerange(self._paramrange)
        result = self._engine.run(kwargs)
        self._paramlist = sorted(self._engine.activelist)
        return result

    def analyse_hl(self, kwargs=dict()):
        # noise offset improved HL likelihood
        offset_bp = np.mean(self._noise_bp,axis=0)  # effecive offset (1503.01347, 2010.01139)
        for l in range(offset_bp.shape[1]):
            offset_bp[:,l,:,:] *= np.sqrt(self._estimator.modes[l]+0.5)
        self.engine = hlfit(self._data_bp,np.mean(self._fiducial_bp,axis=0),np.mean(self._noise_bp,axis=0),self._covmat,self._background_obj,self._foreground_obj,offset_bp)
        if (len(self._paramrange)):
            self._engine.rerange(self._paramrange)
        result = self._engine.run(kwargs)
        self._paramlist = sorted(self._engine.activelist)
        return result
