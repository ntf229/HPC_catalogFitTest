# Prospector fit of galaxies from the NIHAO-SKIRT-Catalog

import os
import argparse
import numpy as np
import fitsio
from prospect.models import transforms

def runProspector():
    if args.sampling == "mcmc":
        os.system('python '+codePath+'/python/trueAlphaSFH.py --emcee --dust='+args.dust+' --dust_type='+
                  args.dust_type+' --nwalkers='+args.nwalkers+' --path='+fitPath+
                  ' --filters='+filter_list+' --niter='+args.niter+
                  ' --filter_errors='+filter_errors+' --total_mass='+str(total_mass)+
                  ' --z_frac0='+str(z_fraction[0])+' --z_frac1='+str(z_fraction[1])+
                  ' --z_frac2='+str(z_fraction[2])+' --z_frac3='+str(z_fraction[3])+
                  ' --z_frac4='+str(z_fraction[4]))
    elif args.sampling == "dynesty":
        os.system('python '+codePath+'/python/trueAlphaSFH.py --dynesty --dust='+args.dust+' --dust_type='+
                  args.dust_type+' --path='+fitPath+
                  ' --filters='+filter_list+
                  ' --filter_errors='+filter_errors+' --total_mass='+total_mass+
                  ' --z_frac0='+str(z_fraction[0])+' --z_frac1='+str(z_fraction[1])+
                  ' --z_frac2='+str(z_fraction[2])+' --z_frac3='+str(z_fraction[3])+
                  ' --z_frac4='+str(z_fraction[4]))

def calcCatalogSFH(sfh, ages):
    # returns SFH in terms of total_mass and z_frac (Prospector parameters)
    ages /= 1e9 # convert from years to Gyrs
    fullAges = np.append(ages, 10**(np.log10(ages[1]) - np.log10(ages[0]) + np.log10(ages[-1]))) # includes right-most $
    bins = np.asarray([0.0, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6]) # in Gyrs
    SFH = np.zeros((len(bins) - 1))
    index = 0 # SFH index
    for i in range(len(fullAges) - 1):
        if i == (len(fullAges) - 2): # last loop
            SFH[index] += sfh[i] # already in M_sun
            #SFH[index] /= (bins[index+1] - bins[index])*1e9 # convert from M_sun to M_sun / year
            break
        if fullAges[i+1] < bins[index+1]: # +1 = right bin edge
            #SFH[index] += sfh[i]*(fullAges[i+1] - fullAges[i])*1e9 # in M_sun
            SFH[index] += sfh[i] # already in M_sun
        else:
            #SFH[index] /= (bins[index+1] - bins[index])*1e9 # convert from M_sun to M_sun / year
            # keep in M_sun 
            index += 1
    alphaBinEdges = np.array([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
    alpha_agelims = (np.log10(alphaBinEdges) + 9) # convert from Gyrs to log10(years)
    agebins = np.array([alpha_agelims[:-1], alpha_agelims[1:]])
    agebins = agebins.T
    total_mass, z_fraction = transforms.masses_to_zfrac(mass=SFH, agebins=agebins)
    return total_mass, z_fraction

parser = argparse.ArgumentParser()
parser.add_argument("--sampling") # mcmc or dynesty 
parser.add_argument("--dust") # include dust; True or False
parser.add_argument("--dust_type") # power_law, milky_way, calzetti, witt_and_gordon, kriek_and_conroy
parser.add_argument("--fitType") # bands/spectra to use in Prospector fit (eg. GSWLC1)
parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit (MCMC parameter)
parser.add_argument("--niter") # number of steps taken by each walker (MCMC parameter)
parser.add_argument("--galaxy") # name of galaxy 
args = parser.parse_args()

if args.fitType == 'GSWLC1':
    print('Using GWSLC1')
    filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV'
    filter_errors = '0.11178,0.03482,0.03078,0.03707,0.06125,0.08322,0.04755'
elif args.fitType == 'DustPedia':
    print('Using DustPedia')
    filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV,'\
                  'twomass_J,twomass_H,twomass_Ks,wise_w1,wise_w2,wise_w3,wise_w4,'\
                  'spitzer_irac_ch1,spitzer_irac_ch2,spitzer_irac_ch3,spitzer_irac_ch4,'\
                  'spitzer_mips_24,spitzer_mips_70,spitzer_mips_160,herschel_pacs_70,'\
                  'herschel_pacs_100,herschel_pacs_160,herschel_spire_250,'\
                  'herschel_spire_350,herschel_spire_500'
    filter_errors = '0.11178,0.03482,0.03078,0.03707,0.06125,0.08322,0.04755,'\
                    '0.12024,0.25264,0.20118,0.04411,0.07867,0.13981,0.23473,'\
                    '0.04156,0.06143,0.10021,0.06306,'\
                    '0.14266,0.16143,0.19578,0.24518,'\
                    '0.29042,0.27249,0.16354,'\
                    '0.20634,0.25484'

codePath = "/home/ntf229/catalogFitTest/"
catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
noDustCatalogPath = '/scratch/ntf229/ageSmooth-noSF-NSC/' # no IR emission in the no-dust data
fitPath = '/scratch/ntf229/catalogFitTest/ProspectorFits/TrueAlphaSFH/'

os.system('mkdir -p '+fitPath)

overwrite = False

# load catalog data
galaxies = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='GALAXIES')
summary = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SUMMARY')
wave = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='WAVE')
spectrum = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPEC')
#spectrum_nodust = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST') # SF
spectrum_nodust = fitsio.read(noDustCatalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST') # noSF
wave_nodust = fitsio.read(noDustCatalogPath+'nihao-integrated-seds.fits', ext='WAVE') # noSF

names = summary['name']
singleNames = galaxies['name']
stellarMass = summary['stellar_mass']
SFR = summary['sfr']
sfh = galaxies['sfh'] # M_sun per year
ages = galaxies['ages'] # years
dustMass = summary['dust_mass']
axisRatios = summary['axis_ratio']
Av = summary['Av']
bands = summary['bands'][0]
#flux = summary['flux']
#flux_noDust = summary['flux_nodust']

nameMask = names == args.galaxy
singleNameMask = singleNames == args.galaxy
faceIndex = np.argmax(axisRatios[nameMask])
edgeIndex = np.argmin(axisRatios[nameMask])

faceSpectrum = spectrum[nameMask][faceIndex]
faceSpectrumNoDust = spectrum_nodust[nameMask][faceIndex]
edgeSpectrum = spectrum[nameMask][edgeIndex]
edgeSpectrumNoDust = spectrum_nodust[nameMask][edgeIndex]

total_mass, z_fraction = calcCatalogSFH(sfh[singleNameMask, :][0], ages[singleNameMask, :][0])

if args.sampling == "mcmc":
    if eval(args.dust):
        fitPath += 'mcmc/dust/'+args.dust_type+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
    else:
        fitPath += 'mcmc/noDust/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
elif args.sampling == "dynesty":
    if eval(args.dust):
        fitPath += 'dynesty/dust/'+args.dust_type+'/'+args.fitType+'/'+args.galaxy+'/'
    else:
        fitPath += 'dynesty/noDust/'+args.fitType+'/'+args.galaxy+'/'
else:
    print('invalid sampling type')
    exit()

os.system('mkdir -p '+fitPath)

# face-on
if eval(args.dust):
    np.save(fitPath+'wave.npy', wave)
    np.save(fitPath+'spec.npy', faceSpectrum) # in Jy
else:
    np.save(fitPath+'wave.npy', wave_nodust)
    np.save(fitPath+'spec.npy', faceSpectrumNoDust) # in Jy

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
if eval(args.dust):
    np.save(fitPath+'wave.npy', wave)
    np.save(fitPath+'spec.npy', edgeSpectrum) # in Jy
else:
    np.save(fitPath+'wave.npy', wave_nodust)
    np.save(fitPath+'spec.npy', edgeSpectrumNoDust) # in Jy

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
