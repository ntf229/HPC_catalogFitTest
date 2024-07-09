import prospect.io.read_results as reader
from prospect.models import transforms
import os
import numpy as np
import argparse
import fitsio
import matplotlib.pyplot as plt
from sedpy.observate import load_filters, getSED

def makePlots(catalogMass, catalogSFR, faceMass, faceSFR, edgeMass, edgeSFR):
    # mass plots
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(catalogMass), np.log10(faceMass), label="face-on", color='blue', alpha=0.5, linewidth=0, s=80)
    plt.scatter(np.log10(catalogMass), np.log10(edgeMass), label="edge-on", color='red', alpha=0.5, linewidth=0, s=80)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-20,20], [-20,20], alpha=0.3, color='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=28)
    plt.xlabel(r'$True \; \log_{10}(Mass \, / \, M_{\odot})$', fontsize=28)
    plt.ylabel(r'$Inferred \; \log_{10}(Mass \, / \, M_{\odot})$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+'mass.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    # SFR plots
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(catalogSFR), np.log10(faceSFR), label="face-on", color='blue', alpha=0.5, linewidth=0, s=80)
    plt.scatter(np.log10(catalogSFR), np.log10(edgeSFR), label="edge-on", color='red', alpha=0.5, linewidth=0, s=80)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-20,20], [-20,20], alpha=0.3, color='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=28)
    plt.xlabel(r'$True \; \log_{10}(SFR \, (M_{\odot}/yr))$', fontsize=28)
    plt.ylabel(r'$Inferred \; \log_{10}(SFR \, (M_{\odot}/yr))$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+'SFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    # sSFR plots
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(catalogSFR/catalogMass), np.log10(faceSFR/faceMass), label="face-on", color='blue', alpha=0.5, linewidth=0, s=80)
    plt.scatter(np.log10(catalogSFR/catalogMass), np.log10(edgeSFR/edgeMass), label="edge-on", color='red', alpha=0.5, linewidth=0, s=80)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-20,20], [-20,20], alpha=0.3, color='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(ticks=[-11, -10.5, -10, -9.5])
    plt.legend(fontsize=28)
    plt.xlabel(r'$True \; \log_{10}(sSFR \, (yr^{-1}))$', fontsize=28)
    plt.ylabel(r'$Inferred \; \log_{10}(sSFR \, (yr^{-1}))$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+'sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    # sSFR vs. stellar mass plots
    plt.figure(figsize=(10,8))
    plt.scatter(np.log10(catalogMass), np.log10(catalogSFR/catalogMass), label="catalog", color='k', alpha=0.5, linewidth=0, s=80, marker='X')
    plt.scatter(np.log10(faceMass), np.log10(faceSFR/faceMass), label="inferred face-on", color='blue', alpha=0.5, linewidth=0, s=80)
    plt.scatter(np.log10(edgeMass), np.log10(edgeSFR/edgeMass), label="inferred edge-on", color='red', alpha=0.5, linewidth=0, s=80)
    plt.legend(fontsize=28)
    plt.xlabel(r'$\log_{10}(Mass \, / \, M_{\odot})$', fontsize=28)
    plt.ylabel(r'$\log_{10}(sSFR \, (yr^{-1}))$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+'mass_sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def delayedTauSFR(tau, tage, mass, intTime=0.1):    
    ''' 
    intTime is integration time for calculating the SFR in Gyrs
    tau is the e-folding time of the SFH in Gyrs
    Delayed-tau SFH is given by t * exp(-t/tau) where t and tau are in Gyrs
    This is parametrized such that t=0 corresponds to the beginning of star-formation 
    To calculate the SFR, we integrate this function (after normalizeing) 
    from t=tage-intTime to t=tage
    '''
    norm = tau*(tau-np.exp(-tage/tau)*(tage+tau))
    SFR = tau*np.exp(-(tage-intTime)/tau)*((tage-intTime)+tau) - tau*np.exp(-tage/tau)*(tage+tau)
    SFR *= mass / norm # full integral equals total mass
    SFR /= intTime*1e9 # convert from mass to mass/year
    return SFR
    
parser = argparse.ArgumentParser()
parser.add_argument("--sampling") # mcmc or dynesty
parser.add_argument("--dust") # include dust; True or False
parser.add_argument("--dust_type") # power_law, milky_way, calzetti, witt_and_gordon, kriek_and_conroy
parser.add_argument("--sfh") # delayed-tau, prospector-alpha
parser.add_argument("--fitType") # bands/spectra to use in Prospector fit (eg. GSWLC1)
parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit
parser.add_argument("--niter") # number of steps taken by each walker
args = parser.parse_args()

codePath = "/home/ntf229/catalogFitTest/"
catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'

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

filter_list = filter_list.split(',')
filterlist = load_filters(filter_list)

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

galaxies = ['g1.88e10','g1.89e10','g1.90e10','g2.34e10','g2.63e10','g2.64e10','g2.80e10','g2.83e10','g2.94e10',
            'g3.44e10','g3.67e10','g3.93e10','g4.27e10','g4.48e10','g4.86e10','g4.94e10','g4.99e10',
            'g5.05e10','g6.12e10','g6.37e10','g6.77e10','g6.91e10','g7.12e10','g8.89e10','g9.59e10','g1.05e11',
            'g1.08e11','g1.37e11','g1.52e11','g1.57e11','g1.59e11','g1.64e11','g2.04e11','g2.19e11','g2.39e11',
            'g2.41e11','g2.42e11','g2.54e11','g3.06e11','g3.21e11','g3.23e11','g3.49e11','g3.55e11','g3.59e11',
            'g3.71e11','g4.90e11','g5.02e11','g5.31e11','g5.36e11','g5.38e11','g5.46e11','g5.55e11','g6.96e11',
            'g7.08e11','g7.44e11','g7.55e11','g7.66e11','g8.06e11','g8.13e11','g8.26e11','g8.28e11','g1.12e12',
            'g1.77e12','g1.92e12','g2.79e12']

catalogMass = np.zeros(len(galaxies))
catalogSFR = np.zeros(len(galaxies))
faceMass = np.zeros(len(galaxies))
faceSFR = np.zeros(len(galaxies))
edgeMass = np.zeros(len(galaxies))
edgeSFR = np.zeros(len(galaxies))

if args.sfh == 'prospector-alpha':
    alpha_agelims = np.asarray([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
    alpha_agelims = np.log10(alpha_agelims) + 9
    agebins = np.array([alpha_agelims[:-1], alpha_agelims[1:]])
    agebins = agebins.T

# loop over galaxies
for g in range(len(galaxies)):
    galaxy = galaxies[g]
    print('starting galaxy', galaxy)
    fitPath = '/scratch/ntf229/catalogFitTest/ProspectorFits/'
    plotPath = '/scratch/ntf229/catalogFitTest/ProspectorGlobalPlots/'
    nameMask = names == galaxy
    catalogMass[g] = stellarMass[nameMask][0]
    catalogSFR[g] = SFR[nameMask][0]
    faceIndex = np.argmax(axisRatios[nameMask])
    edgeIndex = np.argmin(axisRatios[nameMask])
    # set paths and select catalog spectra
    if args.sampling == 'mcmc':
        if eval(args.dust):
            fitPath += 'mcmc/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+galaxy+'/'
            plotPath += 'mcmc/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'
            faceSpectrum = spectrum[nameMask][faceIndex]
            edgeSpectrum = spectrum[nameMask][edgeIndex]
        else:
            fitPath += 'mcmc/noDust/'+args.sfh+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+galaxy+'/'
            plotPath += 'mcmc/noDust/'+args.sfh+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'
            faceSpectrum = spectrum_nodust[nameMask][faceIndex]
            edgeSpectrum = spectrum_nodust[nameMask][edgeIndex]
    elif args.sampling == 'dynesty':
        if eval(args.dust):
            fitPath += 'dynesty/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/'+galaxy+'/'
            plotPath += 'dynesty/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/'
            faceSpectrum = spectrum[nameMask][faceIndex]
            edgeSpectrum = spectrum[nameMask][edgeIndex]
        else:
            fitPath += 'dynesty/noDust/'+args.sfh+'/'+args.fitType+'/'+galaxy+'/'
            plotPath += 'dynesty/noDust/'+args.sfh+'/'+args.fitType+'/'
            faceSpectrum = spectrum_nodust[nameMask][faceIndex]
            edgeSpectrum = spectrum_nodust[nameMask][edgeIndex]
    else:
        print('invalid sampling type')
        exit()
    os.system('mkdir -p '+plotPath)
    # load sps 
    #if args.sfh == 'delayed-tau':
    #    from prospect.sources import CSPSpecBasis
    #    sps = CSPSpecBasis(zcontinuous=1,
    #                       compute_vega_mags=False)
    #elif args.sfh == 'prospector-alpha':
    #        from prospect.sources import FastStepBasis
    #        sps = FastStepBasis()
    # face-on plots
    res, obs, model = reader.results_from(fitPath+'faceFit.h5', dangerous=False)
    theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
    best = res["bestfit"]
    bestParams = np.asarray(best["parameter"])
    if args.sfh == 'delayed-tau':
        tau = bestParams[theta_labels == 'tau']
        tage = bestParams[theta_labels == 'tage']
        mass = bestParams[theta_labels == 'mass']
        faceMass[g] = mass
        faceSFR[g] = delayedTauSFR(tau, tage, mass, intTime=0.1)
    elif args.sfh == 'prospector-alpha':
        mass = bestParams[theta_labels == 'total_mass']
        faceMass[g] = mass
        zFracArray = np.zeros(5)
        zFracArray[0] = bestParams[theta_labels == 'z_fraction_1']
        zFracArray[1] = bestParams[theta_labels == 'z_fraction_2']
        zFracArray[2] = bestParams[theta_labels == 'z_fraction_3']
        zFracArray[3] = bestParams[theta_labels == 'z_fraction_4']
        zFracArray[4] = bestParams[theta_labels == 'z_fraction_5']
        faceSFR[g] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
                     agebins=agebins)[0] / 1e8
    # edge-on plots
    res, obs, model = reader.results_from(fitPath+'edgeFit.h5', dangerous=False)
    theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
    best = res["bestfit"]
    bestParams = np.asarray(best["parameter"])
    if args.sfh == 'delayed-tau':
        tau = bestParams[theta_labels == 'tau']
        tage = bestParams[theta_labels == 'tage']
        mass = bestParams[theta_labels == 'mass']
        edgeMass[g] = mass
        edgeSFR[g] = delayedTauSFR(tau, tage, mass, intTime=0.1)
    elif args.sfh == 'prospector-alpha':
        mass = bestParams[theta_labels == 'total_mass']
        edgeMass[g] = mass
        zFracArray = np.zeros(5)
        zFracArray[0] = bestParams[theta_labels == 'z_fraction_1']
        zFracArray[1] = bestParams[theta_labels == 'z_fraction_2']
        zFracArray[2] = bestParams[theta_labels == 'z_fraction_3']
        zFracArray[3] = bestParams[theta_labels == 'z_fraction_4']
        zFracArray[4] = bestParams[theta_labels == 'z_fraction_5']
        edgeSFR[g] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
                     agebins=agebins)[0] / 1e8

makePlots(catalogMass, catalogSFR, faceMass, faceSFR, edgeMass, edgeSFR)

print('Done')


