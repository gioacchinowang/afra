import healpy as hp
import pysm
import numpy as np
import pysm.units as u

NSIDE = 128


MODEL = ['s1','d1']  # pysm3 model name list
FWHM = [0.5, 0.3, 0.2, 0.08]  # FWHM in deg for beam effect
FREQ = [30, 95, 150, 353]

sky = pysm.Sky(nside=NSIDE, preset_strings=MODEL)
mname = '_'
for i in MODEL:
    mname += i
for i in range(len(FREQ)):
    maps = sky.get_emission(FREQ[i]*u.GHz)
    maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(FREQ[i]*u.GHz))
    maps = pysm.apply_smoothing_and_coord_transform(maps,fwhm=FWHM[i]*u.deg)
    hp.write_map('pysm'+mname+'_'+str(FREQ[i])+'.fits',maps.astype(np.float32),dtype=np.float32)

import camb
from camb import model, initialpower

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.05)
pars.set_for_lmax(2500, lens_potential_accuracy=1);
pars.WantTensors = True
#calculate results for these parameters
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
ls = np.arange(totCL.shape[0])
coefficient = (2.*np.pi)/ls/(ls+1)
totCL[0] = 0.
for i in range(1,len(ls)):
    totCL[i] *= coefficient[i]
hp.write_cl('camb_cls.fits', np.transpose(totCL), dtype=np.float32)
