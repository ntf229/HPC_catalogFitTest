import prospect.io.read_results as reader
import os
import numpy as np
import argparse
import fitsio
import matplotlib.pyplot as plt
from sedpy.observate import load_filters, getSED
from prospect.sources import CSPSpecBasis
from prospect.sources import FastStepBasis
from prospect.models import transforms

def getSpectrum(bestParams, obs, sps, model):
    fake_obs = obs.copy()
    fake_obs['spectrum'] = None
    #fake_obs['wavelength'] = wave # catalog wave
    fake_obs['wavelength'] = fspsWave # fsps full range
    spec, phot, x = model.predict(bestParams, obs=fake_obs, sps=sps)
    spec *= 3631 # convert from Maggies to Jy
    phot *= 3631
    return spec, phot

def orderParams(params1, labels1, labels2):
    ''' This function rearranges params1 so that
    its labels are in the order of labels2.
    They don't have to be the same length, but
    labels1 has to contain all elements of labels2 '''
    params = np.zeros(len(labels2))
    for i in range(len(labels2)):
        for j in range(len(labels1)):
            if labels2[i] == labels1[j]:
                params[i] = params1[j]
                break
    return params

def plotFit():
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(wave_eff), np.log10(phot), label='Prospector', linewidth=0, s=200, marker='s', alpha=0.5)
    plt.scatter(np.log10(wave_eff), np.log10(catalogFlux), label='Catalog', linewidth=0, s=200, marker='o', alpha=0.5)
    plt.xlabel(r'$\log_{10}(\lambda_{eff} \, [\AA])$', fontsize=28)
    plt.ylabel(r'$\log_{10}(f_{\nu} \, [Jy])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    plt.savefig(plotPath+dataType+'_fits/'+galaxy+'_fit.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()

#parser = argparse.ArgumentParser()
#parser.add_argument("--sampling") # mcmc or dynesty
#parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit
#parser.add_argument("--niter") # number of steps taken by each walker
#args = parser.parse_args()

catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
noDustCatalogPath = '/scratch/ntf229/ageSmooth-noSF-NSC/' # no IR emission in the no-dust data

# load catalog data
galaxies = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='GALAXIES')
summary = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SUMMARY')
#wave = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='WAVE')
#catalogSpectrum = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPEC')
#spectrum_nodust = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST') # SF
catalogSpectrum_nodust = fitsio.read(noDustCatalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST') # noSF
wave_nodust = fitsio.read(noDustCatalogPath+'nihao-integrated-seds.fits', ext='WAVE') # noSF

names = summary['name']
singleNames = galaxies['name']
stellarMass = summary['stellar_mass']
singleStellarMass = galaxies['stellar_mass']
SFR = summary['sfr']
sfh = galaxies['sfh'] # M_sun per year
ages = galaxies['ages'] # years
dustMass = summary['dust_mass']
axisRatios = summary['axis_ratio']
Av = summary['Av']
bands = summary['bands'][0]
flux = summary['flux']
flux_noDust = summary['flux_nodust']

fitPath_GSWLC1 = '/scratch/ntf229/catalogFitTest/ProspectorFits/TrueAlphaSFH/mcmc/noDust/GSWLC1/numWalkers16/numIter500/'
fitPath_DustPedia = '/scratch/ntf229/catalogFitTest/ProspectorFits/TrueAlphaSFH/mcmc/noDust/DustPedia/numWalkers16/numIter500/'
codePath = '/home/ntf229/catalogFitTest/'
plotPath = '/scratch/ntf229/catalogFitTest/ProspectorFits/TrueAlphaSFH_plots/'
os.system('mkdir -p '+plotPath+'GSWLC1_fits/')
os.system('mkdir -p '+plotPath+'DustPedia_fits/')

dataTypes = ['GSWLC1', 'DustPedia']

fspsWave = np.load(codePath+'python/full_rf_wavelengths.npy')

fitSFHs = np.zeros((6, 2)) # [SFH bin, orientation (face or edge)]

sps = FastStepBasis()

alphaBinEdges = np.asarray([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
alpha_agelims = np.log10(alphaBinEdges) + 9
agebins = np.array([alpha_agelims[:-1], alpha_agelims[1:]])
agebins = agebins.T

faceMasses = np.zeros((len(singleNames), 2))
faceMetals = np.zeros((len(singleNames), 2))
trueMasses = np.zeros(len(singleNames))

for d in range(len(dataTypes)):
    dataType = dataTypes[d]
    if dataType == 'GSWLC1':
        fitPath = fitPath_GSWLC1
        filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV'
    else:
        fitPath = fitPath_DustPedia
        filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV,'\
                      'twomass_J,twomass_H,twomass_Ks,wise_w1,wise_w2,wise_w3,wise_w4,'\
                      'spitzer_irac_ch1,spitzer_irac_ch2,spitzer_irac_ch3,spitzer_irac_ch4,'\
                      'spitzer_mips_24,spitzer_mips_70,spitzer_mips_160,herschel_pacs_70,'\
                      'herschel_pacs_100,herschel_pacs_160,herschel_spire_250,'\
                      'herschel_spire_350,herschel_spire_500'
    filter_list = filter_list.split(',')
    filterlist = load_filters(filter_list)
    for g in range(len(singleNames)):
        # mask current galaxy
        galaxy = singleNames[g]
        nameMask = names == galaxy
        singleNameMask = singleNames == galaxy
        faceIndex = np.argmax(axisRatios[nameMask])
        edgeIndex = np.argmin(axisRatios[nameMask])
        catalogMass = stellarMass[nameMask][0]
        catalogSFR = SFR[nameMask][0]
        trueMasses[g] = catalogMass
        # calculate catalog broadband photometry from spectrum (no-dust)
        f_lambda_cgs = (1/33333) * (1/(wave_nodust**2)) * catalogSpectrum_nodust[nameMask][faceIndex]
        mags = getSED(wave_nodust, f_lambda_cgs, filterlist=filterlist) # AB magnitudes
        maggies = 10**(-0.4*mags) # convert to maggies
        catalogFlux = maggies * 3631
        # load fits
        res, obs, model = reader.results_from(fitPath+galaxy+'/faceFit.h5', dangerous=False)
        wave_eff = [f.wave_effective for f in res['obs']['filters']]
        model = reader.read_model(fitPath+galaxy+'/model')[0]
        theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
        best = res["bestfit"]
        bestParams = np.asarray(best["parameter"])
        spec, phot = getSpectrum(bestParams, obs, sps, model) # in Jy, catalog wave
        faceMasses[g, d] = bestParams[theta_labels == 'total_mass']
        faceMetals[g, d] = bestParams[theta_labels == 'logzsol']
        #plotFit()

plt.figure(figsize=(10,8))
plt.scatter(np.log10(trueMasses), np.log10(faceMasses[:,0]), label='GSWLC1', linewidth=0, s=200, marker='s', alpha=0.5)
plt.scatter(np.log10(trueMasses), np.log10(faceMasses[:,1]), label='DustPedia', linewidth=0, s=200, marker='o', alpha=0.5)
xlim = plt.xlim()
ylim = plt.ylim()
minimum = np.amin([xlim,ylim])
maximum = np.amax([xlim,ylim])
plt.plot([minimum, maximum], [minimum, maximum], color='k', alpha=0.3)
plt.xlim([minimum, maximum])
plt.ylim([minimum, maximum])
plt.xlabel(r'$\log_{10}(True \, Mass \, [M_{\odot}])$', fontsize=28)
plt.ylabel(r'$\log_{10}(Inferred \, Mass \, [M_{\odot}])$',fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=28)
plt.savefig(plotPath+'masses.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
plt.close()
