import prospect.io.read_results as reader
from prospect.models import transforms
import os
import numpy as np
import argparse
import fitsio
import matplotlib.pyplot as plt
from sedpy.observate import load_filters, getSED

# data = [galaxy, dust model, data selection, parameter, orientation]

def old_deltaPlots():
    # delta sSFR vs. delta mass plot
    plt.figure(figsize=(20,16))
    alpha = 0.4
    markers = ['o', 'v', '^', '<', '>']
    for d in range(len(dustModels)):
        if dustModels[d] == 'witt_and_gordon':
            continue
        for s in range(len(dataSelection)):
            if s == 0:
                colors = ['blue', 'red']
            else:
                colors = ['purple', 'orange']
            faceMass = data[:,d,s,0,0]
            faceSFR = data[:,d,s,1,0]
            edgeMass = data[:,d,s,0,1]
            edgeSFR = data[:,d,s,1,1]
            plt.scatter(np.log10(faceMass) - np.log10(catalogMass), 
                        np.log10(faceSFR/faceMass) - np.log10(catalogSFR/catalogMass), 
                        label=dustModelsClean[d]+' '+dataSelection[s] +" Face-on", color=colors[0], 
                        alpha=alpha, linewidth=0, s=400, marker=markers[d])
            plt.scatter(np.log10(edgeMass) - np.log10(catalogMass), 
                        np.log10(edgeSFR/edgeMass) - np.log10(catalogSFR/catalogMass), 
                        label=dustModelsClean[d]+' '+dataSelection[s] +" Edge-on", color=colors[1], 
                        alpha=alpha, linewidth=0, s=400, marker=markers[d])
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(xlim, [0,0], color='k', alpha=0.3)
    plt.plot([0,0], ylim, color='k', alpha=0.3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=28, loc=(1.02, 0.1))
    plt.xlabel(r'$\Delta\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
    plt.ylabel(r'$\Delta\log_{10}(sSFR \, [yr^{-1}])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+'delta_mass_sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig(plotPath+'zoomed_delta_mass_sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.savefig(plotPath+'zoomed2_delta_mass_sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def deltaPlots():
    # delta sSFR vs. delta mass plot
    alpha = 0.5
    plt.figure(figsize=(10,8))
    for s in range(len(dataSelection)):
        if s == 0:
            colors = ['blue', 'red']
        else:
            colors = ['purple', 'orange']
        faceMass = data[:,s,0,0]
        faceSFR = data[:,s,1,0]
        edgeMass = data[:,s,0,1]
        edgeSFR = data[:,s,1,1]
        plt.scatter(np.log10(faceMass) - np.log10(catalogMass), 
                    np.log10(faceSFR/faceMass) - np.log10(catalogSFR/catalogMass), 
                    label=dataSelection[s]+" Face-on", color=colors[0], 
                    alpha=alpha, linewidth=0, s=200, marker='o')
        plt.scatter(np.log10(edgeMass) - np.log10(catalogMass), 
                    np.log10(edgeSFR/edgeMass) - np.log10(catalogSFR/catalogMass), 
                    label=dataSelection[s]+" Edge-on", color=colors[1], 
                    alpha=alpha, linewidth=0, s=200, marker='o')
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(xlim, [0,0], color='k', alpha=0.3)
    plt.plot([0,0], ylim, color='k', alpha=0.3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=20)
    plt.xlabel(r'$\Delta\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
    plt.ylabel(r'$\Delta\log_{10}(sSFR \, [yr^{-1}])$', fontsize=28)
    plt.title('Prospector-alpha', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+'delta_mass_sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def truthPlots():
    for s in range(len(dataSelection)):
        faceMass = data[:,s,0,0]
        faceSFR = data[:,s,1,0]
        edgeMass = data[:,s,0,1]
        edgeSFR = data[:,s,1,1]
        # mass plots
        plt.figure(figsize=(10,8))
        plt.scatter(np.log10(catalogMass), np.log10(faceMass), label="face-on", color='blue', alpha=0.5, linewidth=0, s=100)
        plt.scatter(np.log10(catalogMass), np.log10(edgeMass), label="edge-on", color='red', alpha=0.5, linewidth=0, s=100)
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.plot([-20,20], [-20,20], alpha=0.3, color='k')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(fontsize=20)
        plt.xlabel(r'$True \; \log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
        plt.ylabel(r'$Inferred \; \log_{10}(Mass \, [M_{\odot}])$',fontsize=28)
        plt.title(dataSelection[s], fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.savefig(plotPath+dataSelection[s]+'_mass.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        # SFR plots
        plt.figure(figsize=(10,8))
        plt.scatter(np.log10(catalogSFR), np.log10(faceSFR), label="face-on", color='blue', alpha=0.5, linewidth=0, s=100)
        plt.scatter(np.log10(catalogSFR), np.log10(edgeSFR), label="edge-on", color='red', alpha=0.5, linewidth=0, s=100)
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.plot([-20,20], [-20,20], alpha=0.3, color='k')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(fontsize=20)
        plt.xlabel(r'$True \; \log_{10}(SFR \, [M_{\odot}/yr])$', fontsize=28)
        plt.ylabel(r'$Inferred \; \log_{10}(SFR \, [M_{\odot}/yr])$',fontsize=28)
        plt.title(dataSelection[s], fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.savefig(plotPath+dataSelection[s]+'_SFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        # sSFR plots
        plt.figure(figsize=(10,8))
        plt.scatter(np.log10(catalogSFR/catalogMass), np.log10(faceSFR/faceMass), label="face-on", color='blue', alpha=0.5, linewidth=0, s=100)
        plt.scatter(np.log10(catalogSFR/catalogMass), np.log10(edgeSFR/edgeMass), label="edge-on", color='red', alpha=0.5, linewidth=0, s=100)
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.plot([-20,20], [-20,20], alpha=0.3, color='k')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(ticks=[-11, -10.5, -10, -9.5])
        plt.legend(fontsize=20)
        plt.xlabel(r'$True \; \log_{10}(sSFR \, [yr^{-1}])$', fontsize=28)
        plt.ylabel(r'$Inferred \; \log_{10}(sSFR \, [yr^{-1}])$',fontsize=28)
        plt.title(dataSelection[s], fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.savefig(plotPath+dataSelection[s]+'_sSFR.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
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
parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit
parser.add_argument("--niter") # number of steps taken by each walker
args = parser.parse_args()

codePath = "/home/ntf229/catalogFitTest/"
catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
plotPath = '/scratch/ntf229/catalogFitTest/ProspectorAlphaGlobal/'
dataPath = '/scratch/ntf229/catalogFitTest/ProspectorAlphaGlobalData/'

if args.sampling == 'mcmc':
    plotPath += 'mcmc/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'
    dataPath += 'mcmc/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'
elif args.sampling == 'dynesty':
    plotPath += 'dynesty/dust/'+args.sfh+'/'
    dataPath += 'dynesty/dust/'+args.sfh+'/'

os.system('mkdir -p '+plotPath)
os.system('mkdir -p '+dataPath)

#dustModels = ['calzetti', 'power_law', 'milky_way', 'witt_and_gordon', 'kriek_and_conroy']
#dustModelsClean = ['Calzetti', 'Power Law', 'Milky Way', 'Witt and Gordon', 'Kriek and Conroy']
dataSelection = ['GSWLC1', 'DustPedia']
params = ['Mass', 'SFR']
orientations = ['Face', 'Edge']

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

# loop over galaxies fill catalog arrays
for g in range(len(galaxies)):
    galaxy = galaxies[g]
    nameMask = names == galaxy
    catalogMass[g] = stellarMass[nameMask][0]
    catalogSFR[g] = SFR[nameMask][0]

overwrite = False
if overwrite == False:
    if os.path.exists(dataPath+'data.npy'):
        print('loading data')
        data = np.load(dataPath+'data.npy')
        deltaPlots()
        #deltaPlotsSeparate()
        truthPlots()
        exit()

data = np.zeros((len(galaxies), len(dataSelection), len(params), len(orientations)))

# loop over dust models, data selection, and galaxies 
for s in range(len(dataSelection)):
    faceMass = np.zeros(len(galaxies))
    faceSFR = np.zeros(len(galaxies))
    edgeMass = np.zeros(len(galaxies))
    edgeSFR = np.zeros(len(galaxies))
    for g in range(len(galaxies)):
        galaxy = galaxies[g]
        fitPath = '/scratch/ntf229/catalogFitTest/ProspectorAlphaFits/'
        # set paths and select catalog spectra
        if args.sampling == 'mcmc':
            fitPath += 'mcmc/'+dataSelection[s]+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+galaxy+'/'
        elif args.sampling == 'dynesty':
            fitPath += 'dynesty/'+dataSelection[s]+'/'+galaxy+'/'
        else:
            print('invalid sampling type')
            exit()
        # face-on plots
        res, obs, model = reader.results_from(fitPath+'faceFit.h5', dangerous=False)
        theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
        best = res["bestfit"]
        bestParams = np.asarray(best["parameter"])
        #tau = bestParams[theta_labels == 'tau']
        #tage = bestParams[theta_labels == 'tage']
        mass = bestParams[theta_labels == 'total_mass']
        faceMass[g] = mass
        #alpha_agelims = np.zeros((7,2))
        alpha_agelims = np.asarray([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
        alpha_agelims = np.log10(alpha_agelims) + 9
        agebins = np.array([alpha_agelims[:-1], alpha_agelims[1:]])
        agebins = agebins.T
        zFracArray = np.zeros(5)
        zFracArray[0] = bestParams[theta_labels == 'z_fraction_1']
        zFracArray[1] = bestParams[theta_labels == 'z_fraction_2']
        zFracArray[2] = bestParams[theta_labels == 'z_fraction_3']
        zFracArray[3] = bestParams[theta_labels == 'z_fraction_4']
        zFracArray[4] = bestParams[theta_labels == 'z_fraction_5']
        #remaining = 1 - np.sum(zFracArray[0:5])
        #print('remaining:', remaining)
        #zFracArray[5] = remaining
        faceSFR[g] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray, 
                     agebins=agebins)[0] / 1e8
        #faceSFR[g] = bestParams[theta_labels == 'z_fraction_1'] * mass / 1e8
        #if args.sfh == 'delayed-tau':
        #    faceSFR[g] = delayedTauSFR(tau, tage, mass, intTime=0.1)
        # edge-on plots
        res, obs, model = reader.results_from(fitPath+'edgeFit.h5', dangerous=False)
        theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
        best = res["bestfit"]
        bestParams = np.asarray(best["parameter"])
        #tau = bestParams[theta_labels == 'tau']
        #tage = bestParams[theta_labels == 'tage']
        mass = bestParams[theta_labels == 'total_mass']
        edgeMass[g] = mass
        zFracArray = np.zeros(5)
        zFracArray[0] = bestParams[theta_labels == 'z_fraction_1']
        zFracArray[1] = bestParams[theta_labels == 'z_fraction_2']
        zFracArray[2] = bestParams[theta_labels == 'z_fraction_3']
        zFracArray[3] = bestParams[theta_labels == 'z_fraction_4']
        zFracArray[4] = bestParams[theta_labels == 'z_fraction_5']
        #remaining = 1 - np.sum(zFracArray[0:5])
        #print('remaining:', remaining)
        #zFracArray[5] = remaining
        edgeSFR[g] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray, 
                     agebins=agebins)[0] / 1e8
        #edgeSFR[g] = bestParams[theta_labels == 'z_fraction_1'] * mass / 1e8
        #if args.sfh == 'delayed-tau':
        #    edgeSFR[g] = delayedTauSFR(tau, tage, mass, intTime=0.1)
    data[:, s, 0, 0] = faceMass
    data[:, s, 1, 0] = faceSFR
    data[:, s, 0, 1] = edgeMass
    data[:, s, 1, 1] = edgeSFR

np.save(dataPath+'data.npy', data)

deltaPlots()
#deltaPlotsSeparate()
truthPlots()

print('Done')


