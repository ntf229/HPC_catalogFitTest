# Prospector-alpha fit of galaxies from the NIHAO-SKIRT-Catalog

import os
import argparse
import numpy as np
import fitsio

# have not implemented variable filter errors (see fit.py)
def runProspector():
    if args.sampling == "mcmc":
        os.system('python '+codePath+'/python/alphaTemplate.py --emcee --nwalkers='+args.nwalkers+
                  ' --path='+fitPath+' --filters='+filter_list+' --niter='+args.niter)
    elif args.sampling == "dynesty":
        os.system('python '+codePath+'/python/alphaTemplate.py --dynesty --path='+fitPath+
                  ' --filters='+filter_list)

parser = argparse.ArgumentParser()
parser.add_argument("--sampling") # mcmc or dynesty 
parser.add_argument("--fitType") # bands/spectra to use in Prospector fit (eg. GSWLC1)
parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit (MCMC parameter)
parser.add_argument("--niter") # number of steps taken by each walker (MCMC parameter)
parser.add_argument("--galaxy") # name of galaxy 
args = parser.parse_args()

if args.fitType == 'GSWLC1':
    print('Using GWSLC1')
    filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV'
elif args.fitType == 'DustPedia':
    print('Using DustPedia')
    filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV,'\
                  'twomass_J,twomass_H,twomass_Ks,wise_w1,wise_w2,wise_w3,wise_w4,'\
                  'spitzer_irac_ch1,spitzer_irac_ch2,spitzer_irac_ch3,spitzer_irac_ch4,'\
                  'spitzer_mips_24,spitzer_mips_70,spitzer_mips_160,herschel_pacs_70,'\
                  'herschel_pacs_100,herschel_pacs_160,herschel_spire_250,'\
                  'herschel_spire_350,herschel_spire_500'

codePath = "/home/ntf229/catalogFitTest/"
catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
fitPath = '/scratch/ntf229/catalogFitTest/ProspectorAlphaFits/'

os.system('mkdir -p '+fitPath)

overwrite = True

# load catalog data
summary = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SUMMARY')
wave = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='WAVE')
spectrum = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPEC')
spectrum_nodust = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST')

names = summary['name']
stellarMass = summary['stellar_mass']
SFR = summary['sfr']
dustMass = summary['dust_mass']
axisRatios = summary['axis_ratio']
Av = summary['Av']
bands = summary['bands'][0]
flux = summary['flux']
flux_noDust = summary['flux_nodust']

nameMask = names == args.galaxy
faceIndex = np.argmax(axisRatios[nameMask])
edgeIndex = np.argmin(axisRatios[nameMask])

faceSpectrum = spectrum[nameMask][faceIndex]
#faceSpectrumNoDust = spectrum_nodust[nameMask][faceIndex]
edgeSpectrum = spectrum[nameMask][edgeIndex]
#edgeSpectrumNoDust = spectrum_nodust[nameMask][edgeIndex]

if args.sampling == "mcmc":
    fitPath += 'mcmc/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
elif args.sampling == "dynesty":
    fitPath += 'dynesty/'+args.fitType+'/'+args.galaxy+'/'
else:
    print('invalid sampling type')
    exit()

os.system('mkdir -p '+fitPath)

# face-on
np.save(fitPath+'wave.npy', wave)
np.save(fitPath+'spec.npy', faceSpectrum) # in Jy

if os.path.exists(fitPath+'faceFit.h5'):
    if overwrite:
        os.remove(fitPath+'faceFit.h5')
        print('removed old faceFit.h5 file, running Prospector fit')
        runProspector()
    else:
        print('Prospector faceFit.h5 file already exists, skipping fit')
else:
    print('no existing faceFit.h5 file, running Prospector fit')
    runProspector()

os.system('rm '+fitPath+'spec.npy')
os.system('mv '+fitPath+'fit.h5 '+fitPath+'faceFit.h5')

# edge-on
np.save(fitPath+'wave.npy', wave)
np.save(fitPath+'spec.npy', edgeSpectrum) # in Jy

if os.path.exists(fitPath+'edgeFit.h5'):
    if overwrite:
        os.remove(fitPath+'edgeFit.h5')
        print('removed old edgeFit.h5 file, running Prospector fit')
        runProspector()
    else:
        print('Prospector edgeFit.h5 file already exists, skipping fit')
else:
    print('no existing edgeFit.h5 file, running Prospector fit')
    runProspector()

os.system('rm '+fitPath+'wave.npy')
os.system('rm '+fitPath+'spec.npy')
os.system('mv '+fitPath+'fit.h5 '+fitPath+'edgeFit.h5')

print('Done')
