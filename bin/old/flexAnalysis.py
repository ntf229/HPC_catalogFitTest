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

def getCatalogFluxes(catalogWave, catalogSpectrum, filterlist):
    faceIndex = np.argmax(axisRatios[nameMask])
    edgeIndex = np.argmin(axisRatios[nameMask])
    # face-on
    f_lambda_cgs = (1/33333) * (1/(catalogWave**2)) * catalogSpectrum[nameMask][faceIndex]
    mags = getSED(catalogWave, f_lambda_cgs, filterlist=filterlist) # AB magnitudes
    maggies = 10**(-0.4*mags) # convert to maggies
    catalogFaceFlux = maggies * 3631
    # edge-on 
    f_lambda_cgs = (1/33333) * (1/(catalogWave**2)) * catalogSpectrum[nameMask][edgeIndex]
    mags = getSED(catalogWave, f_lambda_cgs, filterlist=filterlist) # AB magnitudes
    maggies = 10**(-0.4*mags) # convert to maggies
    catalogEdgeFlux = maggies * 3631
    return catalogFaceFlux, catalogEdgeFlux

def getFitData(path):
    # face-on
    res, obs, model = reader.results_from(path+'faceFit.h5', dangerous=False)
    model = reader.read_model(path+'model')[0]
    theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
    best = res["bestfit"]
    faceBestParams = np.asarray(best["parameter"])
    faceSpec, facePhot = getSpectrum(faceBestParams, obs, sps, model) # in Jy, catalog wave
    #print('face param names:', theta_labels)
    #print('face param values:', faceBestParams)
    # edge-on
    res, obs, model = reader.results_from(path+'edgeFit.h5', dangerous=False)
    model = reader.read_model(path+'model')[0]
    theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
    best = res["bestfit"]
    edgeBestParams = np.asarray(best["parameter"])
    edgeSpec, edgePhot = getSpectrum(edgeBestParams, obs, sps, model) # in Jy, catalog wave
    #print('edge param names:', theta_labels)
    #print('edge param values:', edgeBestParams)
    return faceBestParams, edgeBestParams, theta_labels, facePhot, edgePhot

def getFitSFR(params, labels):
    #mass = params[labels == 'total_mass']
    zFracArray = np.zeros(5)
    zFracArray[0] = params[labels == 'z_fraction_1']
    zFracArray[1] = params[labels == 'z_fraction_2']
    zFracArray[2] = params[labels == 'z_fraction_3']
    zFracArray[3] = params[labels == 'z_fraction_4']
    zFracArray[4] = params[labels == 'z_fraction_5']
    SFR = transforms.zfrac_to_masses(mass, z_fraction=zFracArray, agebins = agebins)[0] / 1e8
    return SFR

def plotFit(dustModel, method, wave_eff, catalogFlux, phot):
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(wave_eff), np.log10(phot), label='Prospector', linewidth=0, s=200, marker='s', alpha=0.5)
    plt.scatter(np.log10(wave_eff), np.log10(catalogFlux), label='Catalog', linewidth=0, s=200, marker='o', alpha=0.5)
    plt.xlabel(r'$\log_{10}(\lambda_{eff} \, [\AA])$', fontsize=28)
    plt.ylabel(r'$\log_{10}(f_{\nu} \, [Jy])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    if dustModel == None:
        path = plotPath+'fits/'+method+'/noDust/'
        os.system('mkdir -p '+path)
        plt.savefig(path+galaxy+'_fit.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
    else:
        path = plotPath+'fits/'+method+'/'+dustModel+'/'
        os.system('mkdir -p '+path)
        plt.savefig(path+galaxy+'_fit.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--sampling") # mcmc or dynesty
#parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit
#parser.add_argument("--niter") # number of steps taken by each walker
args = parser.parse_args()

catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
noDustCatalogPath = '/scratch/ntf229/ageSmooth-noSF-NSC/' # no IR emission in the no-dust data

# load catalog data
galaxies = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='GALAXIES')
summary = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SUMMARY')
wave = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='WAVE')
catalogSpectrum = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPEC')
#spectrum_nodust = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST') # SF
catalogSpectrum_nodust = fitsio.read(noDustCatalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST') # noSF
wave_nodust = fitsio.read(noDustCatalogPath+'nihao-integrated-seds.fits', ext='WAVE') # noSF

names = summary['name']
singleNames = galaxies['name']
stellarMass = summary['stellar_mass']
singleStellarMass = galaxies['stellar_mass']
SFR = summary['sfr']
singleSFR = galaxies['sfr']
sfh = galaxies['sfh'] # M_sun per year
ages = galaxies['ages'] # years
dustMass = summary['dust_mass']
axisRatios = summary['axis_ratio']
Av = summary['Av']
bands = summary['bands'][0]
flux = summary['flux']
flux_noDust = summary['flux_nodust']

fitPath = '/scratch/ntf229/catalogFitTest/ProspectorFlexFit/'
codePath = '/home/ntf229/catalogFitTest/'
plotPath = '/scratch/ntf229/catalogFitTest/ProspectorFlexAnalysis/'
dataPath = plotPath+'data/'
os.system('mkdir -p '+dataPath)

storeData = False 
plotConvergence = True

dataTypes = ['GSWLC1', 'DustPedia']
methods = ['SFH', 'normSFH', 'totalMass', 'wild']
dustModels = ['Calzetti', 'Cardelli', 'PowerLaw', 'KriekAndConroy']
orientations = ['Face-on', 'Edge-on']
dust = ['dust', 'noDust']

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

filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV'
filter_list = filter_list.split(',')
filterlist_GSWLC1 = load_filters(filter_list)

filter_list = 'sdss_u0,sdss_g0,sdss_r0,sdss_i0,sdss_z0,galex_FUV,galex_NUV,'\
              'twomass_J,twomass_H,twomass_Ks,wise_w1,wise_w2,wise_w3,wise_w4,'\
              'spitzer_irac_ch1,spitzer_irac_ch2,spitzer_irac_ch3,spitzer_irac_ch4,'\
              'spitzer_mips_24,spitzer_mips_70,spitzer_mips_160,herschel_pacs_70,'\
              'herschel_pacs_100,herschel_pacs_160,herschel_spire_250,'\
              'herschel_spire_350,herschel_spire_500'
filter_list = filter_list.split(',')
filterlist_DustPedia = load_filters(filter_list)

#DustPediaToGSWLC1Mask = np.zeros(len(filterlist_GSWLC1), dtype=np.int8)
#for i in range(len(filterlist_GSWLC1)):
#    for j in range(len(filterlist_DustPedia)):
#        if filterlist_DustPedia[j] == filterlist_GSWLC1[i]:
#            DustPediaToGSWLC1Mask[i] = j
#            break
#DustPediaToGSWLC1Mask = np.array([0,1,2,3,4,5,6], dtype=np.int8)
#print('DustPediaToGSWLC1Mask:', DustPediaToGSWLC1Mask)
#print('filterlist_DustPedia[DustPediaToGSWLC1Mask]:', filterlist_DustPedia[DustPediaToGSWLC1Mask])
#exit()

numIter = '5000'

# get effective wavelengths
res, obs, model = reader.results_from(fitPath+'SFH/mcmc/dust/calzetti/GSWLC1/numWalkers64/numIter'+
    numIter+'/'+singleNames[0]+'/faceFit.h5', dangerous=False)
wave_eff_GSWLC1 = [f.wave_effective for f in res['obs']['filters']]
res, obs, model = reader.results_from(fitPath+'SFH/mcmc/dust/calzetti/DustPedia/numWalkers64/numIter'+
    numIter+'/'+singleNames[0]+'/faceFit.h5', dangerous=False)
wave_eff_DustPedia = [f.wave_effective for f in res['obs']['filters']]

if storeData:
    # catalog data
    catalogFluxesGSWLC1 = np.zeros((len(singleNames), 2, 2, 
        len(filterlist_GSWLC1))) # galaxy, [face, edge], [dust, no-dust], filter 
    catalogFluxesDustPedia = np.zeros((len(singleNames), 2, 2, 
        len(filterlist_DustPedia))) # galaxy, [face, edge], [dust, no-dust], filter 
    # fit fluxes 
    dustFitFluxesGSWLC1 = np.zeros((len(singleNames), 2, len(methods), len(dustModels), 
        len(filterlist_GSWLC1))) # galaxy, [face, edge], method, dust model, filter 
    dustFitFluxesDustPedia = np.zeros((len(singleNames), 2, len(methods), len(dustModels), 
        len(filterlist_DustPedia))) # galaxy, [face, edge], method, dust model, filter 
    noDustFitFluxesGSWLC1 = np.zeros((len(singleNames), 2, len(methods)-1, 
        len(filterlist_GSWLC1))) # galaxy, [face, edge], method (no SFH), filter 
    noDustFitFluxesDustPedia = np.zeros((len(singleNames), 2, len(methods)-1, 
        len(filterlist_DustPedia))) # galaxy, [face, edge], method (no SFH), filter 
    # fixed SFH 
    dustParamsCalzetti_SFH = np.zeros((len(singleNames), 2, 2, 5)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsCardelli_SFH = np.zeros((len(singleNames), 2, 2, 9)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsPowerLaw_SFH = np.zeros((len(singleNames), 2, 2, 8)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsKriekAndConroy_SFH = np.zeros((len(singleNames), 2, 2, 8)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    # fixed normalized SFH (free total mass)
    dustParamsCalzetti_normSFH = np.zeros((len(singleNames), 2, 2, 6)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsCardelli_normSFH = np.zeros((len(singleNames), 2, 2, 10)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsPowerLaw_normSFH = np.zeros((len(singleNames), 2, 2, 9)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsKriekAndConroy_normSFH = np.zeros((len(singleNames), 2, 2, 9)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    noDustParams_normSFH = np.zeros((len(singleNames), 2, 2, 2)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    # fixed total mass 
    dustParamsCalzetti_totalMass = np.zeros((len(singleNames), 2, 2, 10)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsCardelli_totalMass = np.zeros((len(singleNames), 2, 2, 14)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsPowerLaw_totalMass = np.zeros((len(singleNames), 2, 2, 13)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsKriekAndConroy_totalMass = np.zeros((len(singleNames), 2, 2, 13)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    noDustParams_totalMass = np.zeros((len(singleNames), 2, 2, 6)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    # wild (all parameters free)
    dustParamsCalzetti_wild = np.zeros((len(singleNames), 2, 2, 11)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsCardelli_wild = np.zeros((len(singleNames), 2, 2, 15)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsPowerLaw_wild = np.zeros((len(singleNames), 2, 2, 14)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    dustParamsKriekAndConroy_wild = np.zeros((len(singleNames), 2, 2, 14)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    noDustParams_wild = np.zeros((len(singleNames), 2, 2, 7)) # galaxy, [face, edge], [GSWLC1, DustPedia], paramter
    # parameter names
    # fixed SFH 
    dustParamNamesCalzetti_SFH = np.zeros(5) # change to 5 after refitting 
    dustParamNamesCardelli_SFH = np.zeros(9)
    dustParamNamesPowerLaw_SFH = np.zeros(8) 
    dustParamNamesKriekAndConroy_SFH = np.zeros(8) 
    # fixed normalized SFH (free total mass)
    dustParamNamesCalzetti_normSFH = np.zeros(6) 
    dustParamNamesCardelli_normSFH = np.zeros(10) 
    dustParamNamesPowerLaw_normSFH = np.zeros(9) 
    dustParamNamesKriekAndConroy_normSFH = np.zeros(9) 
    noDustParamNames_normSFH = np.zeros(2)
    # fixed total mass 
    dustParamNamesCalzetti_totalMass = np.zeros(10)
    dustParamNamesCardelli_totalMass = np.zeros(14)
    dustParamNamesPowerLaw_totalMass = np.zeros(13) 
    dustParamNamesKriekAndConroy_totalMass = np.zeros(13) 
    noDustParamNames_totalMass = np.zeros(6) 
    # wild (all parameters free)
    dustParamNamesCalzetti_wild = np.zeros(11) 
    dustParamNamesCardelli_wild = np.zeros(15) 
    dustParamNamesPowerLaw_wild = np.zeros(14)
    dustParamNamesKriekAndConroy_wild = np.zeros(14) 
    noDustParamNames_wild = np.zeros(7) 
    for g in range(len(singleNames)):
        galaxy = singleNames[g]
        nameMask = names == galaxy
        catalogMass = stellarMass[nameMask][0]
        catalogSFR = SFR[nameMask][0]
        trueMasses[g] = catalogMass
        catalogFluxesGSWLC1[g, 0, 0, :], catalogFluxesGSWLC1[g, 1, 0, :] = getCatalogFluxes(
            wave, catalogSpectrum, filterlist_GSWLC1)
        catalogFluxesGSWLC1[g, 0, 1, :], catalogFluxesGSWLC1[g, 1, 1, :] = getCatalogFluxes(
            wave_nodust, catalogSpectrum_nodust, filterlist_GSWLC1)
        catalogFluxesDustPedia[g, 0, 0, :], catalogFluxesDustPedia[g, 1, 0, :] = getCatalogFluxes(
            wave, catalogSpectrum, filterlist_DustPedia)
        catalogFluxesDustPedia[g, 0, 1, :], catalogFluxesDustPedia[g, 1, 1, :] = getCatalogFluxes(
            wave_nodust, catalogSpectrum_nodust, filterlist_DustPedia)
        # *** Load fits ***
        # ** SFH **
        # Calzetti 
        (dustParamsCalzetti_SFH[g, 0, 0, :], dustParamsCalzetti_SFH[g, 1, 0, :], 
            dustParamNamesCalzetti_SFH, dustFitFluxesGSWLC1[g, 0, 0, 0, :], 
            dustFitFluxesGSWLC1[g, 1, 0, 0, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/calzetti/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCalzetti_SFH[g, 0, 1, :], dustParamsCalzetti_SFH[g, 1, 1, :], 
            dustParamNamesCalzetti_SFH, dustFitFluxesDustPedia[g, 0, 0, 0, :],  
            dustFitFluxesDustPedia[g, 1, 0, 0, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/calzetti/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Cardelli
        (dustParamsCardelli_SFH[g, 0, 0, :], dustParamsCardelli_SFH[g, 1, 0, :], 
            dustParamNamesCardelli_SFH, dustFitFluxesGSWLC1[g, 0, 0, 1, :], 
            dustFitFluxesGSWLC1[g, 1, 0, 1, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/cardelli/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCardelli_SFH[g, 0, 1, :], dustParamsCardelli_SFH[g, 1, 1, :], 
            dustParamNamesCardelli_SFH, dustFitFluxesDustPedia[g, 0, 0, 1, :],  
            dustFitFluxesDustPedia[g, 1, 0, 1, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/cardelli/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Power Law
        (dustParamsPowerLaw_SFH[g, 0, 0, :], dustParamsPowerLaw_SFH[g, 1, 0, :], 
            dustParamNamesPowerLaw_SFH, dustFitFluxesGSWLC1[g, 0, 0, 2, :], 
            dustFitFluxesGSWLC1[g, 1, 0, 2, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/power_law/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsPowerLaw_SFH[g, 0, 1, :], dustParamsPowerLaw_SFH[g, 1, 1, :], 
            dustParamNamesPowerLaw_SFH, dustFitFluxesDustPedia[g, 0, 0, 2, :],  
            dustFitFluxesDustPedia[g, 1, 0, 2, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/power_law/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Kriek and Conroy
        (dustParamsKriekAndConroy_SFH[g, 0, 0, :], dustParamsKriekAndConroy_SFH[g, 1, 0, :], 
            dustParamNamesKriekAndConroy_SFH, dustFitFluxesGSWLC1[g, 0, 0, 3, :], 
            dustFitFluxesGSWLC1[g, 1, 0, 3, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/kriek_and_conroy/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsKriekAndConroy_SFH[g, 0, 1, :], dustParamsKriekAndConroy_SFH[g, 1, 1, :], 
            dustParamNamesKriekAndConroy_SFH, dustFitFluxesDustPedia[g, 0, 0, 3, :],  
            dustFitFluxesDustPedia[g, 1, 0, 3, :]) = getFitData(fitPath+
            'SFH/mcmc/dust/kriek_and_conroy/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # ** normSFH **
        # no-dust 
        (noDustParams_normSFH[g, 0, 0, :], noDustParams_normSFH[g, 1, 0, :], 
            noDustParamNames_normSFH, noDustFitFluxesGSWLC1[g, 0, 0, :],  
            noDustFitFluxesGSWLC1[g, 1, 0, :]) = getFitData(fitPath+
            'normSFH/mcmc/noDust/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (noDustParams_normSFH[g, 0, 1, :], noDustParams_normSFH[g, 1, 1, :], 
            noDustParamNames_normSFH, noDustFitFluxesDustPedia[g, 0, 0, :], 
            noDustFitFluxesDustPedia[g, 1, 0, :]) = getFitData(fitPath+
            'normSFH/mcmc/noDust/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Calzetti 
        (dustParamsCalzetti_normSFH[g, 0, 0, :], dustParamsCalzetti_normSFH[g, 1, 0, :], 
            dustParamNamesCalzetti_normSFH, dustFitFluxesGSWLC1[g, 0, 1, 0, :],  
            dustFitFluxesGSWLC1[g, 1, 1, 0, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/calzetti/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCalzetti_normSFH[g, 0, 1, :], dustParamsCalzetti_normSFH[g, 1, 1, :], 
            dustParamNamesCalzetti_normSFH, dustFitFluxesDustPedia[g, 0, 1, 0, :],  
            dustFitFluxesDustPedia[g, 1, 1, 0, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/calzetti/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Cardelli
        (dustParamsCardelli_normSFH[g, 0, 0, :], dustParamsCardelli_normSFH[g, 1, 0, :], 
            dustParamNamesCardelli_normSFH, dustFitFluxesGSWLC1[g, 0, 1, 1, :],  
            dustFitFluxesGSWLC1[g, 1, 1, 1, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/cardelli/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCardelli_normSFH[g, 0, 1, :], dustParamsCardelli_normSFH[g, 1, 1, :], 
            dustParamNamesCardelli_normSFH, dustFitFluxesDustPedia[g, 0, 1, 1, :],  
            dustFitFluxesDustPedia[g, 1, 1, 1, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/cardelli/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Power Law
        (dustParamsPowerLaw_normSFH[g, 0, 0, :], dustParamsPowerLaw_normSFH[g, 1, 0, :], 
            dustParamNamesPowerLaw_normSFH, dustFitFluxesGSWLC1[g, 0, 1, 2, :],  
            dustFitFluxesGSWLC1[g, 1, 1, 2, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/power_law/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsPowerLaw_normSFH[g, 0, 1, :], dustParamsPowerLaw_normSFH[g, 1, 1, :], 
            dustParamNamesPowerLaw_normSFH, dustFitFluxesDustPedia[g, 0, 1, 2, :],  
            dustFitFluxesDustPedia[g, 1, 1, 2, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/power_law/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Kriek and Conroy
        (dustParamsKriekAndConroy_normSFH[g, 0, 0, :], dustParamsKriekAndConroy_normSFH[g, 1, 0, :], 
            dustParamNamesKriekAndConroy_normSFH, dustFitFluxesGSWLC1[g, 0, 1, 3, :],  
            dustFitFluxesGSWLC1[g, 1, 1, 3, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/kriek_and_conroy/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsKriekAndConroy_normSFH[g, 0, 1, :], dustParamsKriekAndConroy_normSFH[g, 1, 1, :], 
            dustParamNamesKriekAndConroy_normSFH, dustFitFluxesDustPedia[g, 0, 1, 3, :],  
            dustFitFluxesDustPedia[g, 1, 1, 3, :]) = getFitData(fitPath+
            'normSFH/mcmc/dust/kriek_and_conroy/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # ** totalMass **
        # no-dust
        (noDustParams_totalMass[g, 0, 0, :], noDustParams_totalMass[g, 1, 0, :], 
            noDustParamNames_totalMass, noDustFitFluxesGSWLC1[g, 0, 1, :],  
            noDustFitFluxesGSWLC1[g, 1, 1, :]) = getFitData(fitPath+
            'totalMass/mcmc/noDust/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (noDustParams_totalMass[g, 0, 1, :], noDustParams_totalMass[g, 1, 1, :], 
            noDustParamNames_totalMass, noDustFitFluxesDustPedia[g, 0, 1, :],  
            noDustFitFluxesDustPedia[g, 1, 1, :]) = getFitData(fitPath+
            'totalMass/mcmc/noDust/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Calzetti 
        (dustParamsCalzetti_totalMass[g, 0, 0, :], dustParamsCalzetti_totalMass[g, 1, 0, :], 
            dustParamNamesCalzetti_totalMass, dustFitFluxesGSWLC1[g, 0, 2, 0, :],  
            dustFitFluxesGSWLC1[g, 1, 2, 0, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/calzetti/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCalzetti_totalMass[g, 0, 1, :], dustParamsCalzetti_totalMass[g, 1, 1, :], 
            dustParamNamesCalzetti_totalMass, dustFitFluxesDustPedia[g, 0, 2, 0, :],  
            dustFitFluxesDustPedia[g, 1, 2, 0, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/calzetti/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Cardelli
        (dustParamsCardelli_totalMass[g, 0, 0, :], dustParamsCardelli_totalMass[g, 1, 0, :], 
            dustParamNamesCardelli_totalMass, dustFitFluxesGSWLC1[g, 0, 2, 1, :],  
            dustFitFluxesGSWLC1[g, 1, 2, 1, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/cardelli/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCardelli_totalMass[g, 0, 1, :], dustParamsCardelli_totalMass[g, 1, 1, :], 
            dustParamNamesCardelli_totalMass, dustFitFluxesDustPedia[g, 0, 2, 1, :],  
            dustFitFluxesDustPedia[g, 1, 2, 1, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/cardelli/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Power Law
        (dustParamsPowerLaw_totalMass[g, 0, 0, :], dustParamsPowerLaw_totalMass[g, 1, 0, :], 
            dustParamNamesPowerLaw_totalMass, dustFitFluxesGSWLC1[g, 0, 2, 2, :],  
            dustFitFluxesGSWLC1[g, 1, 2, 2, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/power_law/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsPowerLaw_totalMass[g, 0, 1, :], dustParamsPowerLaw_totalMass[g, 1, 1, :], 
            dustParamNamesPowerLaw_totalMass, dustFitFluxesDustPedia[g, 0, 2, 2, :],  
            dustFitFluxesDustPedia[g, 1, 2, 2, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/power_law/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Kriek and Conroy
        (dustParamsKriekAndConroy_totalMass[g, 0, 0, :], dustParamsKriekAndConroy_totalMass[g, 1, 0, :], 
            dustParamNamesKriekAndConroy_totalMass, dustFitFluxesGSWLC1[g, 0, 2, 3, :],  
            dustFitFluxesGSWLC1[g, 1, 2, 3, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/kriek_and_conroy/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsKriekAndConroy_totalMass[g, 0, 1, :], dustParamsKriekAndConroy_totalMass[g, 1, 1, :], 
            dustParamNamesKriekAndConroy_totalMass, dustFitFluxesDustPedia[g, 0, 2, 3, :],  
            dustFitFluxesDustPedia[g, 1, 2, 3, :]) = getFitData(fitPath+
            'totalMass/mcmc/dust/kriek_and_conroy/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # ** wild **
        # no-dust
        (noDustParams_wild[g, 0, 0, :], noDustParams_wild[g, 1, 0, :], 
            noDustParamNames_wild, noDustFitFluxesGSWLC1[g, 0, 2, :],  
            noDustFitFluxesGSWLC1[g, 1, 2, :]) = getFitData(fitPath+
            'wild/mcmc/noDust/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (noDustParams_wild[g, 0, 1, :], noDustParams_wild[g, 1, 1, :], 
            noDustParamNames_wild, noDustFitFluxesDustPedia[g, 0, 2, :],  
            noDustFitFluxesDustPedia[g, 1, 2, :]) = getFitData(fitPath+
            'wild/mcmc/noDust/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Calzetti 
        (dustParamsCalzetti_wild[g, 0, 0, :], dustParamsCalzetti_wild[g, 1, 0, :], 
            dustParamNamesCalzetti_wild, dustFitFluxesGSWLC1[g, 0, 3, 0, :],  
            dustFitFluxesGSWLC1[g, 1, 3, 0, :]) = getFitData(fitPath+
            'wild/mcmc/dust/calzetti/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCalzetti_wild[g, 0, 1, :], dustParamsCalzetti_wild[g, 1, 1, :], 
            dustParamNamesCalzetti_wild, dustFitFluxesDustPedia[g, 0, 3, 0, :],  
            dustFitFluxesDustPedia[g, 1, 3, 0, :]) = getFitData(fitPath+
            'wild/mcmc/dust/calzetti/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Cardelli
        (dustParamsCardelli_wild[g, 0, 0, :], dustParamsCardelli_wild[g, 1, 0, :], 
            dustParamNamesCardelli_wild, dustFitFluxesGSWLC1[g, 0, 3, 1, :],  
            dustFitFluxesGSWLC1[g, 1, 3, 1, :]) = getFitData(fitPath+
            'wild/mcmc/dust/cardelli/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsCardelli_wild[g, 0, 1, :], dustParamsCardelli_wild[g, 1, 1, :], 
            dustParamNamesCardelli_wild, dustFitFluxesDustPedia[g, 0, 3, 1, :],  
            dustFitFluxesDustPedia[g, 1, 3, 1, :]) = getFitData(fitPath+
            'wild/mcmc/dust/cardelli/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Power Law
        (dustParamsPowerLaw_wild[g, 0, 0, :], dustParamsPowerLaw_wild[g, 1, 0, :], 
            dustParamNamesPowerLaw_wild, dustFitFluxesGSWLC1[g, 0, 3, 2, :],  
            dustFitFluxesGSWLC1[g, 1, 3, 2, :]) = getFitData(fitPath+
            'wild/mcmc/dust/power_law/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsPowerLaw_wild[g, 0, 1, :], dustParamsPowerLaw_wild[g, 1, 1, :], 
            dustParamNamesPowerLaw_wild, dustFitFluxesDustPedia[g, 0, 3, 2, :],  
            dustFitFluxesDustPedia[g, 1, 3, 2, :]) = getFitData(fitPath+
            'wild/mcmc/dust/power_law/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        # Kriek and Conroy
        (dustParamsKriekAndConroy_wild[g, 0, 0, :], dustParamsKriekAndConroy_wild[g, 1, 0, :], 
            dustParamNamesKriekAndConroy_wild, dustFitFluxesGSWLC1[g, 0, 3, 3, :],  
            dustFitFluxesGSWLC1[g, 1, 3, 3, :]) = getFitData(fitPath+
            'wild/mcmc/dust/kriek_and_conroy/GSWLC1/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
        (dustParamsKriekAndConroy_wild[g, 0, 1, :], dustParamsKriekAndConroy_wild[g, 1, 1, :], 
            dustParamNamesKriekAndConroy_wild, dustFitFluxesDustPedia[g, 0, 3, 3, :],  
            dustFitFluxesDustPedia[g, 1, 3, 3, :]) = getFitData(fitPath+
            'wild/mcmc/dust/kriek_and_conroy/DustPedia/numWalkers64/numIter'+numIter+'/'+galaxy+'/')
    # save data as numpy arrays
    np.save(dataPath+'catalogFluxesGSWLC1.npy', catalogFluxesGSWLC1)
    np.save(dataPath+'catalogFluxesDustPedia.npy', catalogFluxesDustPedia)
    np.save(dataPath+'dustFitFluxesGSWLC1.npy', dustFitFluxesGSWLC1) 
    np.save(dataPath+'dustFitFluxesDustPedia.npy', dustFitFluxesDustPedia)
    np.save(dataPath+'noDustFitFluxesGSWLC1.npy', noDustFitFluxesGSWLC1) 
    np.save(dataPath+'noDustFitFluxesDustPedia.npy', noDustFitFluxesDustPedia)
    np.save(dataPath+'dustParamsCalzetti_SFH.npy', dustParamsCalzetti_SFH)
    np.save(dataPath+'dustParamsCardelli_SFH.npy', dustParamsCardelli_SFH)
    np.save(dataPath+'dustParamsPowerLaw_SFH.npy', dustParamsPowerLaw_SFH) 
    np.save(dataPath+'dustParamsKriekAndConroy_SFH.npy', dustParamsKriekAndConroy_SFH) 
    np.save(dataPath+'dustParamsCalzetti_normSFH.npy', dustParamsCalzetti_normSFH) 
    np.save(dataPath+'dustParamsCardelli_normSFH.npy', dustParamsCardelli_normSFH) 
    np.save(dataPath+'dustParamsPowerLaw_normSFH.npy', dustParamsPowerLaw_normSFH)
    np.save(dataPath+'dustParamsKriekAndConroy_normSFH.npy', dustParamsKriekAndConroy_normSFH)
    np.save(dataPath+'noDustParams_normSFH.npy', noDustParams_normSFH) 
    np.save(dataPath+'dustParamsCalzetti_totalMass.npy', dustParamsCalzetti_totalMass)
    np.save(dataPath+'dustParamsCardelli_totalMass.npy', dustParamsCardelli_totalMass) 
    np.save(dataPath+'dustParamsPowerLaw_totalMass.npy', dustParamsPowerLaw_totalMass)
    np.save(dataPath+'dustParamsKriekAndConroy_totalMass.npy', dustParamsKriekAndConroy_totalMass)
    np.save(dataPath+'noDustParams_totalMass.npy', noDustParams_totalMass) 
    np.save(dataPath+'dustParamsCalzetti_wild.npy', dustParamsCalzetti_wild)
    np.save(dataPath+'dustParamsCardelli_wild.npy', dustParamsCardelli_wild) 
    np.save(dataPath+'dustParamsPowerLaw_wild.npy', dustParamsPowerLaw_wild)
    np.save(dataPath+'dustParamsKriekAndConroy_wild.npy', dustParamsKriekAndConroy_wild)
    np.save(dataPath+'noDustParams_wild.npy', noDustParams_wild) 
    np.save(dataPath+'dustParamNamesCalzetti_SFH.npy', dustParamNamesCalzetti_SFH)
    np.save(dataPath+'dustParamNamesCardelli_SFH.npy', dustParamNamesCardelli_SFH)
    np.save(dataPath+'dustParamNamesPowerLaw_SFH.npy', dustParamNamesPowerLaw_SFH)
    np.save(dataPath+'dustParamNamesKriekAndConroy_SFH.npy', dustParamNamesKriekAndConroy_SFH)
    np.save(dataPath+'dustParamNamesCalzetti_normSFH.npy', dustParamNamesCalzetti_normSFH)
    np.save(dataPath+'dustParamNamesCardelli_normSFH.npy', dustParamNamesCardelli_normSFH)
    np.save(dataPath+'dustParamNamesPowerLaw_normSFH.npy', dustParamNamesPowerLaw_normSFH)
    np.save(dataPath+'dustParamNamesKriekAndConroy_normSFH.npy', dustParamNamesKriekAndConroy_normSFH)
    np.save(dataPath+'noDustParamNames_normSFH.npy', noDustParamNames_normSFH)
    np.save(dataPath+'dustParamNamesCalzetti_totalMass.npy', dustParamNamesCalzetti_totalMass)
    np.save(dataPath+'dustParamNamesCardelli_totalMass.npy', dustParamNamesCardelli_totalMass)
    np.save(dataPath+'dustParamNamesPowerLaw_totalMass.npy', dustParamNamesPowerLaw_totalMass)
    np.save(dataPath+'dustParamNamesKriekAndConroy_totalMass.npy', dustParamNamesKriekAndConroy_totalMass) 
    np.save(dataPath+'noDustParamNames_totalMass.npy', noDustParamNames_totalMass)
    np.save(dataPath+'dustParamNamesCalzetti_wild.npy', dustParamNamesCalzetti_wild)
    np.save(dataPath+'dustParamNamesCardelli_wild.npy', dustParamNamesCardelli_wild)
    np.save(dataPath+'dustParamNamesPowerLaw_wild.npy', dustParamNamesPowerLaw_wild)
    np.save(dataPath+'dustParamNamesKriekAndConroy_wild.npy', dustParamNamesKriekAndConroy_wild)
    np.save(dataPath+'noDustParamNames_wild.npy', noDustParamNames_wild)
else:
    catalogFluxesGSWLC1 = np.load(dataPath+'catalogFluxesGSWLC1.npy')
    catalogFluxesDustPedia = np.load(dataPath+'catalogFluxesDustPedia.npy')
    dustFitFluxesGSWLC1 = np.load(dataPath+'dustFitFluxesGSWLC1.npy')
    dustFitFluxesDustPedia = np.load(dataPath+'dustFitFluxesDustPedia.npy')
    noDustFitFluxesGSWLC1 = np.load(dataPath+'noDustFitFluxesGSWLC1.npy')
    noDustFitFluxesDustPedia = np.load(dataPath+'noDustFitFluxesDustPedia.npy')
    dustParamsCalzetti_SFH = np.load(dataPath+'dustParamsCalzetti_SFH.npy')
    dustParamsCardelli_SFH = np.load(dataPath+'dustParamsCardelli_SFH.npy')
    dustParamsPowerLaw_SFH = np.load(dataPath+'dustParamsPowerLaw_SFH.npy')
    dustParamsKriekAndConroy_SFH = np.load(dataPath+'dustParamsKriekAndConroy_SFH.npy')
    dustParamsCalzetti_normSFH = np.load(dataPath+'dustParamsCalzetti_normSFH.npy')
    dustParamsCardelli_normSFH = np.load(dataPath+'dustParamsCardelli_normSFH.npy')
    dustParamsPowerLaw_normSFH = np.load(dataPath+'dustParamsPowerLaw_normSFH.npy')
    dustParamsKriekAndConroy_normSFH = np.load(dataPath+'dustParamsKriekAndConroy_normSFH.npy')
    noDustParams_normSFH = np.load(dataPath+'noDustParams_normSFH.npy')
    dustParamsCalzetti_totalMass = np.load(dataPath+'dustParamsCalzetti_totalMass.npy')
    dustParamsCardelli_totalMass = np.load(dataPath+'dustParamsCardelli_totalMass.npy')
    dustParamsPowerLaw_totalMass = np.load(dataPath+'dustParamsPowerLaw_totalMass.npy')
    dustParamsKriekAndConroy_totalMass = np.load(dataPath+'dustParamsKriekAndConroy_totalMass.npy')
    noDustParams_totalMass = np.load(dataPath+'noDustParams_totalMass.npy')
    dustParamsCalzetti_wild = np.load(dataPath+'dustParamsCalzetti_wild.npy')
    dustParamsCardelli_wild = np.load(dataPath+'dustParamsCardelli_wild.npy')
    dustParamsPowerLaw_wild = np.load(dataPath+'dustParamsPowerLaw_wild.npy')
    dustParamsKriekAndConroy_wild = np.load(dataPath+'dustParamsKriekAndConroy_wild.npy')
    noDustParams_wild = np.load(dataPath+'noDustParams_wild.npy')
    dustParamNamesCalzetti_SFH = np.load(dataPath+'dustParamNamesCalzetti_SFH.npy')
    dustParamNamesCardelli_SFH = np.load(dataPath+'dustParamNamesCardelli_SFH.npy')
    dustParamNamesPowerLaw_SFH = np.load(dataPath+'dustParamNamesPowerLaw_SFH.npy')
    dustParamNamesKriekAndConroy_SFH = np.load(dataPath+'dustParamNamesKriekAndConroy_SFH.npy')
    dustParamNamesCalzetti_normSFH = np.load(dataPath+'dustParamNamesCalzetti_normSFH.npy')
    dustParamNamesCardelli_normSFH = np.load(dataPath+'dustParamNamesCardelli_normSFH.npy')
    dustParamNamesPowerLaw_normSFH = np.load(dataPath+'dustParamNamesPowerLaw_normSFH.npy')
    dustParamNamesKriekAndConroy_normSFH = np.load(dataPath+'dustParamNamesKriekAndConroy_normSFH.npy')
    noDustParamNames_normSFH = np.load(dataPath+'noDustParamNames_normSFH.npy')
    dustParamNamesCalzetti_totalMass = np.load(dataPath+'dustParamNamesCalzetti_totalMass.npy')
    dustParamNamesCardelli_totalMass = np.load(dataPath+'dustParamNamesCardelli_totalMass.npy')
    dustParamNamesPowerLaw_totalMass = np.load(dataPath+'dustParamNamesPowerLaw_totalMass.npy')
    dustParamNamesKriekAndConroy_totalMass = np.load(dataPath+'dustParamNamesKriekAndConroy_totalMass.npy')
    noDustParamNames_totalMass = np.load(dataPath+'noDustParamNames_totalMass.npy')
    dustParamNamesCalzetti_wild = np.load(dataPath+'dustParamNamesCalzetti_wild.npy')
    dustParamNamesCardelli_wild = np.load(dataPath+'dustParamNamesCardelli_wild.npy')
    dustParamNamesPowerLaw_wild = np.load(dataPath+'dustParamNamesPowerLaw_wild.npy')
    dustParamNamesKriekAndConroy_wild = np.load(dataPath+'dustParamNamesKriekAndConroy_wild.npy')
    noDustParamNames_wild = np.load(dataPath+'noDustParamNames_wild.npy') 

# dataTypes = ['GSWLC1', 'DustPedia']
# methods = ['SFH', 'normSFH', 'totalMass', 'wild']
# dustModels = ['Calzetti', 'Cardelli', 'PowerLaw', 'KriekAndConroy']
# orientations = ['Face-on', 'Edge-on']
# dust = ['dust', 'noDust']

#plotFit('Calzetti', 'SFH', wave_eff_GSWLC1, catalogFluxesGSWLC1[g, 0, 0, :], dustFitFluxesGSWLC1[g, 0, 0, 0, :]) # face-on

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr(y, c=5.0):
    # compute autocorrelation time for each walker and then average
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def plotConvergence(path, view):
    res, obs, model = reader.results_from(fitPath+path+view+'Fit.h5', dangerous=False)
    theta_labels = np.asarray(res["theta_labels"])
    chain = res["chain"] # chain shape: (numWalkers, numIter, numParams)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('plasma')
    color = iter(cmap(np.linspace(0, 0.9, len(theta_labels))))
    for t in range(len(theta_labels)):
        c=next(color)
        param = theta_labels[t]
        currentChain = chain[:, :, theta_labels == param][:,:,0]
        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(currentChain.shape[1]), 10)).astype(int)
        tau = np.empty(len(N))
        for i, n in enumerate(N):
            tau[i] = autocorr(currentChain[:, :n])
        plt.plot(np.log10(N), np.log10(tau), "o-", label=param, markersize=10, linewidth=3, c=c)
    ylim = plt.gca().get_ylim()
    plt.plot(np.log10(N), np.log10(N / 50.0), "--k", label=r"$\tau = N/50$", linewidth=3)
    plt.ylim(ylim)
    plt.xlabel(r"$\log_{10}(N)$", fontsize=28)
    plt.ylabel(r"$\log_{10}(\tau)$", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=12, ncol=2)
    os.system('mkdir -p '+plotPath+'convergence/'+path)
    plt.savefig(plotPath+'convergence/'+path+view+'_convergence.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

dustPaths = ['noDust/', 'dust/calzetti/', 'dust/cardelli/', 'dust/power_law/', 'dust/kriek_and_conroy/']

if plotConvergence:
    for i in range(len(methods)):
        for j in range(len(dustPaths)):
            if (i == 0) and (j == 0):
                continue # SFH needs dust turned on
            for k in range(len(dataTypes)):
                for l in range(len(singleNames)):
                    if singleStellarMass[l] < 10**(10.5):
                        continue
                    path = methods[i]+'/mcmc/'+dustPaths[j]+dataTypes[k]+'/numWalkers64/numIter5000/'+singleNames[l]+'/'
                    plotConvergence(path, 'face')
                    plotConvergence(path, 'edge')

exit() # DELETE THIS LATER

dustList = ['no-dust', 'Calzetti', 'Cardelli', 'PowerLaw', 'KriekAndConroy']
params = ['total_mass', 'SFR']

def plotTruth(method, dustType, dataType, param, catalogValues, fitValues):
    os.system('mkdir -p '+plotPath+method+'/'+dustType+'/')
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(catalogValues), np.log10(fitValues[:,0]), linewidth=0, s=200, marker='o', alpha=0.7, label='face-on')
    plt.scatter(np.log10(catalogValues), np.log10(fitValues[:,1]), linewidth=0, s=200, marker='o', alpha=0.7, label='edge-on')
    xlim = plt.xlim()
    ylim = plt.ylim()
    minimum = np.amin([xlim,ylim])
    maximum = np.amax([xlim,ylim])
    plt.plot([minimum, maximum], [minimum, maximum], color='k', alpha=0.3)
    plt.xlim([minimum, maximum])
    plt.ylim([minimum, maximum])
    if param == 'total_mass':
        plt.xlabel(r'$\log_{10}(True \, Mass \, [M_{\odot}])$', fontsize=28)
        plt.ylabel(r'$\log_{10}(Inferred \, Mass \, [M_{\odot}])$',fontsize=28)
    elif param == 'SFR':
        plt.xlabel(r'$\log_{10}(True \, SFR \, [M_{\odot} \, / \, yr])$', fontsize=28)
        plt.ylabel(r'$\log_{10}(Inferred \, SFR \, [M_{\odot} \, / \, yr])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    plt.title(dataType+', '+dustType, fontsize=28)
    plt.savefig(plotPath+method+'/'+dustType+'/'+dataType+'_'+param+'.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()

def plotGSWLC1Errors(method, dustType, fitErrors):
    # for DustPedia fits, plot the chi squared error of only the GSWLC1 filters
    # for GSWLC1 fits, plot the chi squared error of the GSWLC1 filters
    os.system('mkdir -p '+plotPath+'GSWLC1Errors/'+method+'/')
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(singleStellarMass), np.log10(fitErrors[:, 0, 0]), 
        linewidth=0, s=200, marker='o', alpha=0.7, color='blue', label='GSWLC1, face-on')
    plt.scatter(np.log10(singleStellarMass), np.log10(fitErrors[:, 1, 0]), 
        linewidth=0, s=200, marker='s', alpha=0.7, color='blue', label='GSWLC1, edge-on')
    plt.scatter(np.log10(singleStellarMass), np.log10(fitErrors[:, 0, 1]), 
        linewidth=0, s=200, marker='o', alpha=0.7, color='red', label='DustPedia, face-on')
    plt.scatter(np.log10(singleStellarMass), np.log10(fitErrors[:, 1, 1]), 
        linewidth=0, s=200, marker='s', alpha=0.7, color='red', label='DustPedia, edge-on')
    plt.xlabel(r'$\log_{10}(True \, Mass \, [M_{\odot}])$', fontsize=28)
    plt.ylabel('GSWLC1 '+r'$\log_{10}(RMSE)$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=20)
    plt.title(method+', '+dustType, fontsize=28)
    plt.savefig(plotPath+'GSWLC1Errors/'+method+'/'+dustType+'_GSWLC1Errors.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    
    

plotMass = np.array([False, True, False, True]) # don't plot mass for SFH or totalMass methods (fixed)
plotSFR = np.array([False, False, True, True]) # don't plot SFR for SFH or normSFH method (fixed)

for i in range(len(methods)):
    for j in range(len(dustList)):
        if (i == 0) and (j == 0):
            continue # SFH method needs dust turned on
        fitFilters = np.zeros((len(singleNames), 2, 2, len(filterlist_GSWLC1))) # galaxy, orientation, [GSWLC1, DP], GSWLC1 filters
        fitErrors = np.zeros((len(singleNames), 2, 2))
        catalogFilters = np.zeros((len(singleNames), 2, len(filterlist_GSWLC1))) # galaxy, orientation, GSWLC1 filters
        if j == 0: # no-dust
            catalogFilters[:, 0, :] = catalogFluxesGSWLC1[:, 0, 1, :] # no-dust
            catalogFilters[:, 1, :] = catalogFluxesGSWLC1[:, 1, 1, :] # no-dust
            fitFilters[:, 0, 0, :] = noDustFitFluxesGSWLC1[:, 0, int(i-1), :]
            fitFilters[:, 1, 0, :] = noDustFitFluxesGSWLC1[:, 1, int(i-1), :]
            fitFilters[:, 0, 1, :] = noDustFitFluxesDustPedia[:, 0, int(i-1), :7]
            fitFilters[:, 1, 1, :] = noDustFitFluxesDustPedia[:, 1, int(i-1), :7]
        else: # dust
            catalogFilters[:, 0, :] = catalogFluxesGSWLC1[:, 0, 0, :] # dust
            catalogFilters[:, 1, :] = catalogFluxesGSWLC1[:, 1, 0, :] # dust
            fitFilters[:, 0, 0, :] = dustFitFluxesGSWLC1[:, 0, i, int(j-1), :]
            fitFilters[:, 1, 0, :] = dustFitFluxesGSWLC1[:, 1, i, int(j-1), :]
            fitFilters[:, 0, 1, :] = dustFitFluxesDustPedia[:, 0, i, int(j-1), :7]
            fitFilters[:, 1, 1, :] = dustFitFluxesDustPedia[:, 1, i, int(j-1), :7]
        fitErrors[:, 0, 0] = np.mean(np.sqrt((catalogFilters[:, 0, :] - fitFilters[:, 0, 0, :])**2), axis=-1)
        fitErrors[:, 1, 0] = np.mean(np.sqrt((catalogFilters[:, 1, :] - fitFilters[:, 1, 0, :])**2), axis=-1)
        fitErrors[:, 0, 1] = np.mean(np.sqrt((catalogFilters[:, 0, :] - fitFilters[:, 0, 1, :])**2), axis=-1)
        fitErrors[:, 1, 1] = np.mean(np.sqrt((catalogFilters[:, 1, :] - fitFilters[:, 1, 1, :])**2), axis=-1)
        plotGSWLC1Errors(methods[i], dustList[j], fitErrors)
        for k in range(len(dataTypes)):
            if methods[i] == 'SFH':
                if dustList[j] == 'Calzetti':
                    fitData = dustParamsCalzetti_SFH[:,:,k,:]
                    paramNames = dustParamNamesCalzetti_SFH
                elif dustList[j] == 'Cardelli':
                    fitData = dustParamsCardelli_SFH[:,:,k,:]
                    paramNames = dustParamNamesCardelli_SFH
                elif dustList[j] == 'PowerLaw':
                    fitData = dustParamsPowerLaw_SFH[:,:,k,:]
                    paramNames = dustParamNamesPowerLaw_SFH
                elif dustList[j] == 'KriekAndConroy':
                    fitData = dustParamsKriekAndConroy_SFH[:,:,k,:]
                    paramNames = dustParamNamesKriekAndConroy_SFH
            elif methods[i] == 'normSFH':
                if dustList[j] == 'no-dust':
                    fitData = noDustParams_normSFH[:,:,k,:]
                    paramNames = noDustParamNames_normSFH
                elif dustList[j] == 'Calzetti':
                    fitData = dustParamsCalzetti_normSFH[:,:,k,:]
                    paramNames = dustParamNamesCalzetti_normSFH
                elif dustList[j] == 'Cardelli':
                    fitData = dustParamsCardelli_normSFH[:,:,k,:]
                    paramNames = dustParamNamesCardelli_normSFH
                elif dustList[j] == 'PowerLaw':
                    fitData = dustParamsPowerLaw_normSFH[:,:,k,:]
                    paramNames = dustParamNamesPowerLaw_normSFH
                elif dustList[j] == 'KriekAndConroy':
                    fitData = dustParamsKriekAndConroy_normSFH[:,:,k,:]
                    paramNames = dustParamNamesKriekAndConroy_normSFH
            elif methods[i] == 'totalMass':
                if dustList[j] == 'no-dust':
                    fitData = noDustParams_totalMass[:,:,k,:]
                    paramNames = noDustParamNames_totalMass
                elif dustList[j] == 'Calzetti':
                    fitData = dustParamsCalzetti_totalMass[:,:,k,:]
                    paramNames = dustParamNamesCalzetti_totalMass
                elif dustList[j] == 'Cardelli':
                    fitData = dustParamsCardelli_totalMass[:,:,k,:]
                    paramNames = dustParamNamesCardelli_totalMass
                elif dustList[j] == 'PowerLaw':
                    fitData = dustParamsPowerLaw_totalMass[:,:,k,:]
                    paramNames = dustParamNamesPowerLaw_totalMass
                elif dustList[j] == 'KriekAndConroy':
                    fitData = dustParamsKriekAndConroy_totalMass[:,:,k,:]
                    paramNames = dustParamNamesKriekAndConroy_totalMass
            elif methods[i] == 'wild':
                if dustList[j] == 'no-dust':
                    fitData = noDustParams_wild[:,:,k,:]
                    paramNames = noDustParamNames_wild
                elif dustList[j] == 'Calzetti':
                    fitData = dustParamsCalzetti_wild[:,:,k,:]
                    paramNames = dustParamNamesCalzetti_wild
                elif dustList[j] == 'Cardelli':
                    fitData = dustParamsCardelli_wild[:,:,k,:]
                    paramNames = dustParamNamesCardelli_wild
                elif dustList[j] == 'PowerLaw':
                    fitData = dustParamsPowerLaw_wild[:,:,k,:]
                    paramNames = dustParamNamesPowerLaw_wild
                elif dustList[j] == 'KriekAndConroy':
                    fitData = dustParamsKriekAndConroy_wild[:,:,k,:]
                    paramNames = dustParamNamesKriekAndConroy_wild
            if plotMass[i]:
                fitMass = fitData[:, :, paramNames == 'total_mass']
                print('fitMass shape:', fitMass.shape)
                print('paramNames:', paramNames)
                plotTruth(methods[i], dustList[j], dataTypes[k], 'total_mass', singleStellarMass, fitMass)
            if plotSFR[i]:
                fitSFR = np.zeros((len(singleSFR), 2))
                if i == 2:
                    masses0 = singleStellarMass
                    masses1 = singleStellarMass
                else:
                    masses0 = fitData[:, 0, paramNames == 'total_mass']
                    masses1 = fitData[:, 1, paramNames == 'total_mass']
                for g in range(len(singleSFR)):
                    mass = masses0[g]
                    fitSFR[g, 0] = getFitSFR(fitData[g,0,:], paramNames) # face-on
                    mass = masses1[g]
                    fitSFR[g, 1] = getFitSFR(fitData[g,1,:], paramNames) # edge-on
                #fitSFR = getFitSFR(fitData, paramNames)
                print('fitSFR shape:', fitSFR.shape)
                plotTruth(methods[i], dustList[j], dataTypes[k], 'SFR', singleSFR, fitSFR)
                    
print('done')





