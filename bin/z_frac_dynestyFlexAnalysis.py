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
    spec = np.array(spec * 3631) # convert from Maggies to Jy
    phot = np.array(phot * 3631)
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
    weights = res["weights"]
    model = reader.read_model(path+'model')[0]
    faceObs = obs["maggies"] * 3631 # convert from maggies to Jy
    faceUnc = obs["maggies_unc"] * 3631 # convert from maggies to Jy
    theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
    imax = np.argmax(res['lnprobability'])
    faceBestParams = res["chain"][imax, :]
    faceSpec, facePhot = getSpectrum(faceBestParams, obs, sps, model) # in Jy, catalog wave
    faceBestParamsUnc = np.zeros((len(theta_labels), 2))
    for i in range(len(theta_labels)):
        param = res["chain"][:, theta_labels == theta_labels[i]][:,0]
        face_lower_one_sigma, face_upper_one_sigma = equalTailedInterval(param, weights, 0.68)
        #face_lower_two_sigma, face_upper_two_sigma = equalTailedInterval(param, res['lnprobability'], weights, 0.95)
        faceBestParamsUnc[i,0] = faceBestParams[theta_labels == theta_labels[i]]-face_lower_one_sigma
        faceBestParamsUnc[i,1] = face_upper_one_sigma-faceBestParams[theta_labels == theta_labels[i]]
    # edge-on
    res, obs, model = reader.results_from(path+'edgeFit.h5', dangerous=False)
    weights = res["weights"]
    model = reader.read_model(path+'model')[0]
    edgeObs = obs["maggies"] * 3631 # convert from maggies to Jy
    edgeUnc = obs["maggies_unc"] * 3631 # convert from maggies to Jy
    theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
    imax = np.argmax(res['lnprobability'])
    edgeBestParams = res["chain"][imax, :]
    edgeSpec, edgePhot = getSpectrum(edgeBestParams, obs, sps, model) # in Jy
    edgeBestParamsUnc = np.zeros((len(theta_labels), 2))
    for i in range(len(theta_labels)):
        param = res["chain"][:, theta_labels == theta_labels[i]][:,0]
        edge_lower_one_sigma, edge_upper_one_sigma = equalTailedInterval(param, weights, 0.68)
        #edge_lower_two_sigma, edge_upper_two_sigma = equalTailedInterval(param, res['lnprobability'], weights, 0.95)
        edgeBestParamsUnc[i,0] = edgeBestParams[theta_labels == theta_labels[i]]-edge_lower_one_sigma
        edgeBestParamsUnc[i,1] = edge_upper_one_sigma-edgeBestParams[theta_labels == theta_labels[i]]
    return (faceBestParams, faceBestParamsUnc, edgeBestParams, edgeBestParamsUnc, theta_labels, 
            facePhot, edgePhot, faceObs, faceUnc, edgeObs, edgeUnc, faceSpec, edgeSpec)

def getFitSFR(params, labels, mass):
    zFracArray = np.zeros(5)
    zFracArray[0] = params[labels == 'z_fraction_1']
    zFracArray[1] = params[labels == 'z_fraction_2']
    zFracArray[2] = params[labels == 'z_fraction_3']
    zFracArray[3] = params[labels == 'z_fraction_4']
    zFracArray[4] = params[labels == 'z_fraction_5']
    SFR = transforms.zfrac_to_masses(mass, z_fraction=zFracArray, agebins = agebins)[0] / 1e8
    return SFR

def plotFit(dustModel, method, wave_eff, catalogFlux, catalogWave, catalogSpec, phot, unc, fitWave, fitSpec, orientation):
    if len(catalogFlux) == len(filterlist_GSWLC1):
        dataType = 'GSWLC1'
    else:
        dataType = 'DustPedia'
    yerrs = np.zeros((2, len(catalogFlux)))
    waveUpperLim = []
    fluxUpperLim = []
    for i in range(len(catalogFlux)):
        if (catalogFlux[i] - unc[i]) <= 0:
            yerrs[:,i] = np.asarray([0., np.log10(catalogFlux[i] + unc[i]) - np.log10(catalogFlux[i])])
            waveUpperLim.append(wave_eff[i])
            fluxUpperLim.append(catalogFlux[i])
        else:
            yerrs[:,i] = np.asarray([np.log10(catalogFlux[i]) - np.log10(catalogFlux[i] - unc[i]), 
                         np.log10(catalogFlux[i] + unc[i]) - np.log10(catalogFlux[i])])
    catalogWaveMask = (catalogWave >= np.amin(wave_eff)) & (catalogWave <= np.amax(wave_eff))
    fitWaveMask = (fitWave >= np.amin(wave_eff)) & (fitWave <= np.amax(wave_eff))
    plt.figure(figsize=(10,8))
    #plt.scatter(np.log10(wave_eff), np.log10(phot), label='Prospector', linewidth=0, s=200, marker='s', alpha=0.5)
    plt.plot(np.log10(catalogWave[catalogWaveMask]), np.log10(catalogSpec[catalogWaveMask]), alpha=0.3, color='k')
    plt.errorbar(np.log10(wave_eff), np.log10(catalogFlux), label='Catalog', 
    			 xerr=None, yerr=yerrs, elinewidth=1, marker='s',
    			 markersize=12, linewidth=0, alpha=0.5, markeredgewidth=0.0, color='k')
    plt.plot(np.log10(fitWave[fitWaveMask]), np.log10(fitSpec[fitWaveMask]), alpha=0.3, color='green')
    plt.scatter(np.log10(wave_eff), np.log10(phot), label='Fit', linewidth=0, s=200, marker='o', alpha=0.5, color='green')
    ylim = plt.ylim()
    minimum = np.amin(ylim)
    plt.vlines(np.log10(waveUpperLim), np.ones(len(waveUpperLim))*minimum, np.log10(fluxUpperLim),
               color='k', alpha=0.5, linewidth=1)
    plt.ylim(ylim)
    plt.xlabel(r'$\log_{10}(\lambda \, [\AA])$', fontsize=28)
    plt.ylabel(r'$\log_{10}(f_{\nu} \, [Jy])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    if dustModel == None:
        path = plotPath+'fits/'+method+'/noDust/'
    else:
        path = plotPath+'fits/'+method+'/'+dustModel+'/'
    os.system('mkdir -p '+path)
    plt.savefig(path+dataType+'_'+galaxy+'_'+orientation+'_fit.png', dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()

#def plotTruth(method, dustType, dataType, param, catalogValues, fitValues, fitUnc): # with error bars
def plotTruth(method, dustType, dataType, param, catalogValues, fitValues):
    # diagonal truth plots
    os.system('mkdir -p '+plotPath+'truthPlots/'+method+'/'+dustType+'/')
    if method == 'SFH':
        methodLabel = 'Fixed SFH'
    elif method == 'normSFH':
        methodLabel = 'Fixed sSFH'
    elif method == 'totalMass':
        methodLabel = 'Fixed Mass'
    else:
        methodLabel = 'Free SFH'
    if dustType == 'PowerLaw':
        dustLabel = 'Power Law'
    elif dustType == 'KriekAndConroy':
        dustLabel = 'K&C'
    elif dustType == 'no-dust':
        dustLabel = 'No Dust'
    else:
        dustLabel = dustType
    if dataType == 'DustPedia':
        dataLabel = 'UV-IR'
    else:
        dataLabel = 'UV-Optical'
    plt.figure(figsize=(10,8))
    #
    # with error bars
    # using equal-tailed interval, it's possible to get negative uncertainties
    #negUncMask = fitUnc < 0
    #if len(fitUnc) != 0:
    #    print('setting negative uncertainties to 0')
    #    fitUnc[negUncMask] = 0
    #faceYerrs = np.zeros((2, len(fitValues[:,0])))
    #edgeYerrs = np.zeros((2, len(fitValues[:,0])))
    #faceYerrs = np.asarray([np.log10(fitValues[:,0]) - np.log10(fitValues[:,0] - fitUnc[:,0,0]), 
    #                        np.log10(fitValues[:,0] + fitUnc[:,0,1]) - np.log10(fitValues[:,0])])
    #edgeYerrs = np.asarray([np.log10(fitValues[:,1]) - np.log10(fitValues[:,1] - fitUnc[:,1,0]), 
    #                        np.log10(fitValues[:,1] + fitUnc[:,1,1]) - np.log10(fitValues[:,1])])
    #plt.errorbar(np.log10(catalogValues), np.log10(fitValues[:,0]), yerr=faceYerrs, linewidth=0, 
    #             markersize=12, elinewidth=1, marker='o', alpha=0.5, label='face-on', color='blue')
    #plt.errorbar(np.log10(catalogValues), np.log10(fitValues[:,1]), yerr=edgeYerrs,  linewidth=0, 
    #             markersize=12, elinewidth=1, marker='s', alpha=0.5, label='edge-on', color='red')
    #
    # without error bars
    plt.scatter(np.log10(catalogValues), np.log10(fitValues[:,0]), linewidth=0, 
                 markersize=12, marker='o', alpha=0.5, label='face-on', color='blue')
    plt.scatter(np.log10(catalogValues), np.log10(fitValues[:,1]), linewidth=0, 
                 markersize=12, marker='s', alpha=0.5, label='edge-on', color='red')
    xlim = plt.xlim()
    ylim = plt.ylim()
    minimum = np.amin([xlim,ylim])
    maximum = np.amax([xlim,ylim])
    # plot diagonal line
    plt.plot([minimum, maximum], [minimum, maximum], color='k', alpha=0.3)
    plt.xlim([minimum, maximum])
    plt.ylim([minimum, maximum])
    if param == 'total_mass':
        plt.xlabel(r'$\log_{10}(\rm True \; Mass \; [M_{\odot}])$', fontsize=28)
        plt.ylabel(r'$\log_{10}(\rm Inferred \; Mass \; [M_{\odot}])$',fontsize=28)
    elif param == 'SFR':
        plt.xlabel(r'$\log_{10}(\rm True \; SFR \; [M_{\odot} \; / \; yr])$', fontsize=28)
        plt.ylabel(r'$\log_{10}(\rm Inferred \; SFR \; [M_{\odot} \; / \; yr])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    plt.title(methodLabel+', '+dustLabel+', '+dataLabel, fontsize=28)
    plt.savefig(plotPath+'truthPlots/'+method+'/'+dustType+'/'+method+'_'+dustType+'_'+dataType+'_'+param+'.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()

#def plotResiduals(method, dustType, dataType, param, catalogValues, fitValues, fitUnc): # with error bars
def plotResiduals(method, dustType, dataType, param, catalogValues, fitValues):
    os.system('mkdir -p '+plotPath+'residuals/')
    if method == 'SFH':
        methodLabel = 'Fixed SFH'
    elif method == 'normSFH':
        methodLabel = 'Fixed sSFH'
    elif method == 'totalMass':
        methodLabel = 'Fixed Mass'
    else:
        methodLabel = 'Free SFH'
    if dustType == 'PowerLaw':
        dustLabel = 'Power Law'
    elif dustType == 'KriekAndConroy':
        dustLabel = 'K&C'
    elif dustType == 'no-dust':
        dustLabel = 'No Dust'
    else:
        dustLabel = dustType
    if dataType == 'DustPedia':
        dataLabel = 'UV-IR'
    else:
        dataLabel = 'UV-Optical'
    plt.figure(figsize=(10,8))
    #
    # with error bars
    # using equal-tailed interval, it's possible to get negative uncertainties
    #negUncMask = fitUnc < 0
    #if len(fitUnc) != 0:
    #    print('setting negative uncertainties to 0')
    #    fitUnc[negUncMask] = 0
    #faceYerrs = np.zeros((2, len(fitValues[:,0])))
    #edgeYerrs = np.zeros((2, len(fitValues[:,0])))
    #faceYerrs = np.asarray([np.log10(fitValues[:,0]) - np.log10(fitValues[:,0] - fitUnc[:,0,0]), 
    #                        np.log10(fitValues[:,0] + fitUnc[:,0,1]) - np.log10(fitValues[:,0])])
    #edgeYerrs = np.asarray([np.log10(fitValues[:,1]) - np.log10(fitValues[:,1] - fitUnc[:,1,0]), 
    #                        np.log10(fitValues[:,1] + fitUnc[:,1,1]) - np.log10(fitValues[:,1])])
    #plt.errorbar(np.log10(catalogValues), np.log10(fitValues[:,0]) - np.log10(catalogValues), yerr=faceYerrs, linewidth=0, 
    #             markersize=12, elinewidth=1, marker='o', alpha=0.5, label='face-on', color='blue')
    #plt.errorbar(np.log10(catalogValues), np.log10(fitValues[:,1]) - np.log10(catalogValues), yerr=edgeYerrs,  linewidth=0, 
    #             markersize=12, elinewidth=1, marker='s', alpha=0.5, label='edge-on', color='red')
    #
    # without error bars
    plt.scatter(np.log10(catalogValues), np.log10(fitValues[:,0]) - np.log10(catalogValues), linewidth=0, 
                 s=200, marker='o', alpha=0.5, label='face-on', color='blue')
    plt.scatter(np.log10(catalogValues), np.log10(fitValues[:,1]) - np.log10(catalogValues), linewidth=0, 
                 s=200, marker='s', alpha=0.5, label='edge-on', color='red')
    # plot horizontal line
    plt.axhline(y = 0, color = 'k', linewidth=3, alpha=0.5)
    if param == 'total_mass':
        plt.xlabel(r'$\log_{10}(\rm True \; Mass \; [M_{\odot}])$', fontsize=28)
        plt.ylabel(r'$\rm Residual \; \log_{10}(\rm Mass \; [M_{\odot}])$',fontsize=28)
    elif param == 'SFR':
        plt.xlabel(r'$\log_{10}(\rm True \; SFR \; [M_{\odot} \; / \; yr])$', fontsize=28)
        plt.ylabel(r'$\rm Residual \; \log_{10}(\rm SFR \; [M_{\odot} \; / \; yr])$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    plt.title(methodLabel+', '+dustLabel+', '+dataLabel, fontsize=28)
    plt.savefig(plotPath+'residuals/'+method+'_'+dustType+'_'+dataType+'_'+param+'.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.25)
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
    plt.savefig(plotPath+'GSWLC1Errors/'+method+'/'+dustType+'_GSWLC1Errors.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()

#def weightedQuantile(posterior, weights, quantile):
#    this actually gives the iso-posterior surface containing 68% of the points
#    sort = np.argsort(posterior)
#    weights = weights[sort]
#    posterior = posterior[sort]
#    totalWeight = np.sum(weights)
#    for i in range(len(weights)):
#        qi = (np.sum(weights[:i]) / totalWeight)
#        if qi >= quantile:
#            return posterior[i]

def equalTailedInterval(param, weights, quantile):
    half = (1-quantile)/2 # 68% -> 16%
    sort = np.argsort(param)
    param = param[sort]
    weights = weights[sort]
    totalWeight = np.sum(weights)
    cdf = np.cumsum(weights) / totalWeight
    lower = param[cdf <= half][-1]
    upper = param[cdf >= (1-half)][0]
    return lower, upper

def plotMoments(method, param, lowMassBias, lowMassStd, highMassBias, highMassStd):
    # [dust, data selection, orientation]
    if method == 'SFH':
        methodLabel = 'Fixed SFH'
    elif method == 'normSFH':
        methodLabel = 'Fixed sSFH'
    elif method == 'totalMass':
        methodLabel = 'Fixed Mass'
    else:
        methodLabel = 'Free SFH'
    if param == 'Mass':
        xLabel = r'$\log_{10}(\rm Mass \; [M_{\odot}]) \; \rm Bias$'
        yLabel = r'$\rm Residual \; \log_{10}(\rm Mass \; [M_{\odot}]) \; \sigma$'
        #yLabel = r'$\log_{10}(\rm Mass \; [M_{\odot}]) \; \rm Variance$'
    else:
        xLabel = r'$\log_{10}(\rm SFR \; [M_{\odot} \; / \; yr]) \; \rm Bias$'
        yLabel = r'$\rm Residual \; \log_{10}(\rm SFR \; [M_{\odot} \; / \; yr]) \; \sigma$'
        #yLabel = r'$\log_{10}(\rm SFR \; [M_{\odot} \; / \; yr]) \; \rm Variance$'
    os.system('mkdir -p '+plotPath+'moments/')
    dataNames = ['UV-Optical', 'UV-IR']
    dustNames = ['No Dust', 'Calzetti', 'Cardelli', 'Power Law', 'K&C']
    dustSymbols = ['*', 's', '^', 'o', 'D']
    dustSymbolSizes = [300, 200, 200, 200, 200]
    for i in range(len(dataNames)):
        plt.figure(figsize=(10,8))
        for j in range(len(dustNames)):
            if method == 'normSFH':
                if dataNames[i] == 'UV-Optical':
                    if dustNames[j] == 'Power Law':
                        continue # still running
            plt.scatter(None, None, linewidth=0, s=dustSymbolSizes[j], 
                marker=dustSymbols[j], alpha=1, color='k', label=dustNames[j])
            # low mass
            plt.scatter(lowMassBias[j,i,0], lowMassStd[j,i,0], linewidth=0, s=dustSymbolSizes[j], 
                marker=dustSymbols[j], alpha=0.7, color='blue')
            plt.scatter(lowMassBias[j,i,1], lowMassStd[j,i,1], linewidth=0, s=dustSymbolSizes[j], 
                marker=dustSymbols[j], alpha=0.7, color='red')
            plt.plot([lowMassBias[j,i,0],lowMassBias[j,i,1]], [lowMassStd[j,i,0], lowMassStd[j,i,1]],
                color='k', alpha=0.3, linewidth=2)
            # high mass
            plt.scatter(highMassBias[j,i,0], highMassStd[j,i,0], linewidth=3, s=dustSymbolSizes[j], edgecolors='k', 
                marker=dustSymbols[j], alpha=0.7, color='blue')
            plt.scatter(highMassBias[j,i,1], highMassStd[j,i,1], linewidth=3, s=dustSymbolSizes[j], edgecolors='k',
                marker=dustSymbols[j], alpha=0.7, color='red')
            plt.plot([highMassBias[j,i,0],highMassBias[j,i,1]], [highMassStd[j,i,0], highMassStd[j,i,1]],
                color='k', alpha=0.3, linewidth=2)
        plt.axvline(x = 0, color = 'k', linewidth=3, alpha=0.5)
        plt.axhline(y = 0, color = 'k', linewidth=3, alpha=0.5)
        ylim = plt.ylim()
        plt.ylim([-0.05 * np.amax(ylim), np.amax(ylim)])
        plt.xlabel(xLabel, fontsize=28)
        plt.ylabel(yLabel, fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabel+', '+dataNames[i], fontsize=28)
        plt.savefig(plotPath+'moments/'+param+'_'+method+'_'+dataNames[i]+'_moments.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()

def plotAllMoments(methods, massBias, massStd, SFRBias, SFRStd):
    plotMass = np.array([False, True, False, True]) # don't plot mass for SFH or totalMass methods (fixed)
    plotSFR = np.array([False, False, True, True]) # don't plot SFR for SFH or normSFH method (fixed)
    methodLabels = ['Fixed SFH', 'Fixed sSFH', 'Fixed Mass', 'Free SFH']
    params = ['Mass', 'SFR']
    xLabels = [r'$\log_{10}(\rm Mass \; [M_{\odot}]) \; \rm Bias$', 
        r'$\log_{10}(\rm SFR \; [M_{\odot} \; / \; yr]) \; \rm Bias$']
    yLabels = [r'$\rm Residual \; \log_{10}(\rm Mass \; [M_{\odot}]) \; \sigma$',
        r'$\rm Residual \; \log_{10}(\rm SFR \; [M_{\odot} \; / \; yr]) \; \sigma$']
    os.system('mkdir -p '+plotPath+'allMoments/')
    dataNames = ['UV-Optical', 'UV-IR']
    dustNames = ['No Dust', 'Calzetti', 'Cardelli', 'Power Law', 'K&C']
    dustSymbols = ['*', 's', '^', 'o', 'D']
    dustSymbolSizes = [600, 200, 300, 300, 200]
    length = np.amax(massBias) - np.amin(massBias)
    xRangeMass = [np.amin(massBias) - 0.05*length, np.amax(massBias) + 0.05*length]
    length = np.amax(massStd) - np.amin(massStd)
    yRangeMass = [-0.05*length, np.amax(massStd) + 0.05*length]
    length = np.amax(SFRBias) - np.amin(SFRBias)
    xRangeSFR = [np.amin(SFRBias) - 0.05*length, np.amax(SFRBias) + 0.05*length]
    length = np.amax(SFRStd) - np.amin(SFRStd)
    yRangeSFR = [-0.05*length, np.amax(SFRStd) + 0.05*length]
    for m in range(len(methods)):
        if m == 1:
            massPlotIndex = 0
        elif m == 2:
            SFRPlotIndex = 0
        elif m == 3:
            massPlotIndex = 1
            SFRPlotIndex = 1
        for i in range(len(dataNames)):
            for p in range(2): # Mass, SFR
                if p == 0:
                    if not plotMass[m]:
                        continue
                    else:
                        bias = massBias[massPlotIndex,:,:,:]
                        std = massStd[massPlotIndex,:,:,:]
                        xRange = xRangeMass
                        yRange = yRangeMass
                else: 
                    if not plotSFR[m]:
                        continue
                    else:
                        bias = SFRBias[SFRPlotIndex,:,:,:]
                        std = SFRStd[SFRPlotIndex,:,:,:]
                        xRange = xRangeSFR
                        yRange = yRangeSFR
                plt.figure(figsize=(10,8))
                for j in range(len(dustNames)):
                    if methodLabels[m] == 'Fixed sSFH':
                        if dataNames[i] == 'UV-Optical':
                            if dustNames[j] == 'Power Law':
                                continue # still running
                    plt.scatter(None, None, linewidth=0, s=dustSymbolSizes[j], 
                        marker=dustSymbols[j], alpha=1, color='k', label=dustNames[j])
                    # high mass
                    plt.scatter(bias[j,i,0], std[j,i,0], linewidth=0, s=dustSymbolSizes[j], edgecolors='k', 
                        marker=dustSymbols[j], alpha=0.7, color='blue')
                    plt.scatter(bias[j,i,1], std[j,i,1], linewidth=0, s=dustSymbolSizes[j], edgecolors='k',
                        marker=dustSymbols[j], alpha=0.7, color='red')
                    plt.plot([bias[j,i,0], bias[j,i,1]], [std[j,i,0], std[j,i,1]],
                        color='k', alpha=0.3, linewidth=2)
                plt.axvline(x = 0, color = 'k', linewidth=3, alpha=0.5)
                plt.axhline(y = 0, color = 'k', linewidth=3, alpha=0.5)
                plt.xlim(xRange)
                plt.ylim(yRange)
                plt.xlabel(xLabels[p], fontsize=28)
                plt.ylabel(yLabels[p], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=4)
                plt.legend(fontsize=20)
                plt.title(methodLabels[m]+', '+dataNames[i], fontsize=28)
                plt.savefig(plotPath+'allMoments/'+params[p]+'_'+methods[m]+'_'+dataNames[i]+'_moments.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.25)
                plt.close()

# chi2_UV_Optical = np.zeros((len(singleStellarMass), len(methods), len(dustList), numView, len(filterlist_GSWLC1), numData))
def plotChi2(chi2_UV_IR, chi2_UV_Optical, wave_eff_DustPedia, wave_eff_GSWLC1):
    os.system('mkdir -p '+plotPath+'chi2/')
    methodLabels = ['Fixed SFH', 'Fixed sSFH', 'Fixed Mass', 'Free SFH']
    dustNames = ['No Dust', 'Calzetti', 'Cardelli', 'Power Law', 'K&C']
    dustSymbols = ['*', 's', '^', 'o', 'D']
    dustSymbolSizes = [600, 200, 300, 300, 200]
    for m in range(len(methods)):
        # UV-IR
        # plot chi2 of each band separate, averaged over galaxies
        plt.figure(figsize=(10,8))
        for d in range(len(dustList)):
            plt.scatter(None, None, linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='k', label=dustNames[d]) 
            plt.scatter(np.log10(wave_eff_DustPedia), np.mean(chi2_UV_IR[:,m,d,0,:], axis=0), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='blue') 
            plt.scatter(np.log10(wave_eff_DustPedia), np.mean(chi2_UV_IR[:,m,d,1,:], axis=0), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='red') 
        plt.xlabel(r'$\lambda_{\rm eff}\; [\AA]$', fontsize=28)
        plt.ylabel(r'$<\chi_{\rm UV-IR}^{2}>$', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-IR', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_IR_chi2_waveEff.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        # plot chi2 of each galaxy vs. true mass, summed over bands
        plt.figure(figsize=(10,8))
        for d in range(len(dustList)):
            plt.scatter(None, None, linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='k', label=dustNames[d]) 
            plt.scatter(np.log10(singleStellarMass), np.sum(chi2_UV_IR[:,m,d,0,:], axis=-1), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='blue') 
            plt.scatter(np.log10(singleStellarMass), np.sum(chi2_UV_IR[:,m,d,1,:], axis=-1), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='red') 
        plt.xlabel(r'$\log_{10}(True \, Mass \, [M_{\odot}])$', fontsize=28)
        plt.ylabel(r'$\chi_{\rm UV-IR}^{2}$', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-IR', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_IR_chi2_mass.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        # UV-Optical
        # plot chi2 of each band separate, averaged over galaxies
        plt.figure(figsize=(10,8))
        for d in range(len(dustList)):
            plt.scatter(None, None, linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='k', label=dustNames[d]) 
            plt.scatter(np.log10(wave_eff_GSWLC1), np.mean(chi2_UV_Optical[:,m,d,0,:,0], axis=0), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='cyan') 
            plt.scatter(np.log10(wave_eff_GSWLC1), np.mean(chi2_UV_Optical[:,m,d,1,:,0], axis=0), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='orange') 
            plt.scatter(np.log10(wave_eff_GSWLC1), np.mean(chi2_UV_Optical[:,m,d,0,:,1], axis=0), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='blue') 
            plt.scatter(np.log10(wave_eff_GSWLC1), np.mean(chi2_UV_Optical[:,m,d,1,:,1], axis=0), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='red')
        plt.xlabel(r'$\lambda_{\rm eff}\; [\AA]$', fontsize=28)
        plt.ylabel(r'$<\chi_{\rm UV-Optical}^{2}>$', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-Optical', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_Optical_chi2_waveEff.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        # plot chi2 of each galaxy vs. true mass, summed over bands
        plt.figure(figsize=(10,8))
        for d in range(len(dustList)):
            plt.scatter(None, None, linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='k', label=dustNames[d]) 
            plt.scatter(np.log10(singleStellarMass), np.sum(chi2_UV_Optical[:,m,d,0,:,0], axis=-1), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='cyan') 
            plt.scatter(np.log10(singleStellarMass), np.sum(chi2_UV_Optical[:,m,d,1,:,0], axis=-1), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='orange') 
            plt.scatter(np.log10(singleStellarMass), np.sum(chi2_UV_Optical[:,m,d,0,:,1], axis=-1), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='blue') 
            plt.scatter(np.log10(singleStellarMass), np.sum(chi2_UV_Optical[:,m,d,1,:,1], axis=-1), linewidth=0, 
                s=dustSymbolSizes[d], marker=dustSymbols[d], alpha=0.7, color='red') 
        plt.xlabel(r'$\log_{10}(True \, Mass \, [M_{\odot}])$', fontsize=28)
        plt.ylabel(r'$\chi_{\rm UV-Optical}^{2}$', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-Optical', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_Optical_chi2_mass.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        # for each method, plot chi2 summed over all bands,
        # averaged over low and high mass galaxies separately
        # UV-IR bands, UV-IR fits
        plt.figure(figsize=(10,8))
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_IR[lowMass,m,:,0,:], axis=-1), axis=0), linewidth=0, 
            s=100, marker='o', alpha=0.7, color='blue') 
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_IR[lowMass,m,:,1,:], axis=-1), axis=0), linewidth=0, 
            s=100, marker='s', alpha=0.7, color='red')
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_IR[highMass,m,:,0,:], axis=-1), axis=0), linewidth=0, 
            s=400, marker='o', alpha=0.7, color='blue', label='Face-on') 
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_IR[highMass,m,:,1,:], axis=-1), axis=0), linewidth=0, 
            s=400, marker='s', alpha=0.7, color='red', label='Edge-on')
        plt.ylabel(r'$\langle \chi_{\rm UV-IR}^{2} \rangle$', fontsize=28)
        plt.xticks(ticks=[0,1,2,3,4], labels=dustNames, fontsize=20)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-IR', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_IR_chi2_dustModels.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        # UV-Optical bands, UV-Optical Fits
        plt.figure(figsize=(10,8))
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[lowMass,m,:,0,:,0], axis=-1), axis=0), linewidth=0, 
            s=100, marker='o', alpha=0.7, color='blue') 
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[lowMass,m,:,1,:,0], axis=-1), axis=0), linewidth=0, 
            s=100, marker='s', alpha=0.7, color='red')
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[highMass,m,:,0,:,0], axis=-1), axis=0), linewidth=0, 
            s=400, marker='o', alpha=0.7, color='blue', label='Face-on') 
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[highMass,m,:,1,:,0], axis=-1), axis=0), linewidth=0, 
            s=400, marker='s', alpha=0.7, color='red', label='Edge-on')
        plt.ylabel(r'$\langle \chi_{\rm UV-Optical}^{2} \rangle$', fontsize=28)
        plt.xticks(ticks=[0,1,2,3,4], labels=dustNames, fontsize=20)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-Optical', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_Optical_chi2_dustModels.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        # UV-Optical bands, UV-IR Fits
        plt.figure(figsize=(10,8))
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[lowMass,m,:,0,:,1], axis=-1), axis=0), linewidth=0, 
            s=100, marker='o', alpha=0.7, color='blue') 
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[lowMass,m,:,1,:,1], axis=-1), axis=0), linewidth=0, 
            s=100, marker='s', alpha=0.7, color='red')
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[highMass,m,:,0,:,1], axis=-1), axis=0), linewidth=0, 
            s=400, marker='o', alpha=0.7, color='blue', label='Face-on') 
        plt.scatter([0,1,2,3,4], np.mean(np.sum(chi2_UV_Optical[highMass,m,:,1,:,1], axis=-1), axis=0), linewidth=0, 
            s=400, marker='s', alpha=0.7, color='red', label='Edge-on')
        plt.ylabel(r'$\langle \chi_{\rm UV-Optical}^{2} \rangle$', fontsize=28)
        plt.xticks(ticks=[0,1,2,3,4], labels=dustNames, fontsize=20)
        plt.yticks(fontsize=28)
        plt.locator_params(axis='y', nbins=4)
        plt.legend(fontsize=20)
        plt.title(methodLabels[m]+', UV-IR', fontsize=28)
        plt.savefig(plotPath+'chi2/'+methods[m]+'_'+'_UV_Optical_fitsIR_chi2_dustModels.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close()

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

# calculate distance from mass following DustPedia fits
logDist = -0.902 + (0.218 * np.log10(singleStellarMass))
dist = 10**(logDist) # in Mpcs

fitPath = '/scratch/ntf229/catalogFitTest/ProspectorFlexFit/'
codePath = '/home/ntf229/catalogFitTest/'
plotPath = '/scratch/ntf229/catalogFitTest/dynestyFlexAnalysis/'
dataPath = '/scratch/ntf229/catalogFitTest/dynestyFlexAnalysisData/'
os.system('mkdir -p '+dataPath)

storeData = False 
plotFits = False

dataTypes = ['GSWLC1', 'DustPedia']
methods = ['SFH', 'normSFH', 'totalMass', 'wild']
dustModels = ['Calzetti', 'Cardelli', 'PowerLaw', 'KriekAndConroy']
orientations = ['Face-on', 'Edge-on']
dust = ['dust', 'noDust']

fspsWave = np.load(codePath+'python/full_rf_wavelengths.npy')

sps = FastStepBasis()

alphaBinEdges = np.asarray([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
alpha_agelims = np.log10(alphaBinEdges) + 9
agebins = np.array([alpha_agelims[:-1], alpha_agelims[1:]])
agebins = agebins.T

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

# get effective wavelengths
res, obs, model = reader.results_from(fitPath+'wild/dynesty/dust/cardelli/GSWLC1/'+
                  singleNames[0]+'/faceFit.h5', dangerous=False)
wave_eff_GSWLC1 = [f.wave_effective for f in res['obs']['filters']]
res, obs, model = reader.results_from(fitPath+'wild/dynesty/dust/cardelli/DustPedia/'+
                  singleNames[0]+'/faceFit.h5', dangerous=False)
wave_eff_DustPedia = [f.wave_effective for f in res['obs']['filters']]

dustPaths = ['noDust/', 'dust/calzetti/', 'dust/cardelli/', 'dust/power_law/', 'dust/kriek_and_conroy/']

dustList = ['no-dust', 'Calzetti', 'Cardelli', 'PowerLaw', 'KriekAndConroy']
params = ['total_mass', 'SFR']

plotMass = np.array([False, True, False, True]) # don't plot mass for SFH or totalMass methods (fixed)
plotSFR = np.array([False, False, True, True]) # don't plot SFR for SFH or normSFH method (fixed)

numData = 2
numView = 2
numFlux = [len(filterlist_GSWLC1), len(filterlist_DustPedia)]
numMethod = [1, 2, 6, 7]
numDust = [0, 4, 8, 7, 7] # 3 from dust emission
if storeData:
    for m in range(len(methods)): # ['SFH', 'normSFH', 'totalMass', 'wild']
        #if m != 3:
        #    continue # only wild
        #if (m != 2) and (m != 3):
        #    continue # only totalMass and wild
        if m == 0:
            continue # skip SFH
        for d in range(len(dustPaths)): # ['noDust/', 'dust/calzetti/', 'dust/cardelli/', 'dust/power_law/', 'dust/kriek_and_conroy/']
            if (m == 0) and (d == 0):
                continue # SFH method needs dust turned on
            #if d != 2:
            #    continue # only cardelli
            for b in range(len(dataTypes)): # ['GSWLC1', 'DustPedia']
                if m == 1:
                    if d == 3:
                        if b == 0:
                            continue # skip normSFH, power law, GSWLC1 (still running)
                path = dataPath+methods[m]+'/'+dustPaths[d]+'/'+dataTypes[b]+'/'
                os.system('mkdir -p '+path)
                numParams = int(numMethod[m] + numDust[d])
                bestParams = np.zeros((len(singleNames), numView, numParams)) 
                bestParamsUnc = np.zeros((len(singleNames), numView, numParams, 2)) # [lower, upper]
                bestFluxes = np.zeros((len(singleNames), numView, numFlux[b]))
                bestSpec = np.zeros((len(singleNames), numView, len(fspsWave)))
                obsFluxes = np.zeros((len(singleNames), numView, numFlux[b]))
                obsUnc = np.zeros((len(singleNames), numView, numFlux[b]))
                for g in range(len(singleNames)):
                    (faceParams, faceParamsUnc, edgeParams, edgeParamsUnc, paramNames, faceFluxes, 
                        edgeFluxes, faceObs, faceUnc, edgeObs, edgeUnc, faceSpec, edgeSpec) = getFitData(
                        fitPath+methods[m]+'/dynesty/'+dustPaths[d]+dataTypes[b]+'/'+singleNames[g]+'/')
                    bestParams[g,0,:] = faceParams  
                    bestParams[g,1,:] = edgeParams 
                    bestParamsUnc[g,0,:,:] = faceParamsUnc  
                    bestParamsUnc[g,1,:,:] = edgeParamsUnc 
                    bestFluxes[g,0,:] = faceFluxes
                    bestFluxes[g,1,:] = edgeFluxes
                    bestSpec[g,0,:] = faceSpec 
                    bestSpec[g,1,:] = edgeSpec 
                    obsFluxes[g,0,:] = faceObs
                    obsFluxes[g,1,:] = edgeObs
                    obsUnc[g,0,:] = faceUnc
                    obsUnc[g,1,:] = edgeUnc
                np.save(path+'bestParams.npy', bestParams)
                np.save(path+'bestParamsUnc.npy', bestParamsUnc)
                np.save(path+'paramNames.npy', paramNames)
                np.save(path+'bestFluxes.npy', bestFluxes)
                np.save(path+'bestSpec.npy', bestSpec)
                np.save(path+'obsFluxes.npy', obsFluxes)
                np.save(path+'obsUnc.npy', obsUnc)

# [method, dust, data selection, orientation]
# for mass, first index is [normSFH, wild]
# for SFR, first index is [totalMass, wild]
# Mass
lowMassBiasMass = np.zeros((2, len(dustList), numData, numView))
highMassBiasMass = np.zeros((2, len(dustList), numData, numView))
lowMassStdMass = np.zeros((2, len(dustList), numData, numView))
highMassStdMass = np.zeros((2, len(dustList), numData, numView))
# SFR
lowMassBiasSFR = np.zeros((2, len(dustList), numData, numView))
highMassBiasSFR = np.zeros((2, len(dustList), numData, numView))
lowMassStdSFR = np.zeros((2, len(dustList), numData, numView))
highMassStdSFR = np.zeros((2, len(dustList), numData, numView))

lowMass = np.log10(singleStellarMass) < 9.5
highMass = np.log10(singleStellarMass) >= 9.5

chi2_UV_IR = np.zeros((len(singleStellarMass), len(methods), len(dustList), numView, len(filterlist_DustPedia)))
chi2_UV_Optical = np.zeros((len(singleStellarMass), len(methods), len(dustList), numView, len(filterlist_GSWLC1), numData)) # includes UV-Optical and UV-IR fits

# set to nan so unfinished runs don't get plotted
chi2_UV_IR[:,:,:,:,:] = np.nan
chi2_UV_Optical[:,:,:,:,:,:] = np.nan

for m in range(len(methods)): # ['SFH', 'normSFH', 'totalMass', 'wild']
    if m == 1:
        massPlotIndex = 0
    elif m == 2:
        SFRPlotIndex = 0
    elif m == 3:
        massPlotIndex = 1
        SFRPlotIndex = 1
    if m == 0:
        continue # skip SFH
    for d in range(len(dustPaths)): # ['noDust/', 'dust/calzetti/', 'dust/cardelli/', 'dust/power_law/', 'dust/kriek_and_conroy/']
        if (m == 0) and (d == 0):
            continue # SFH method needs dust turned on
        for b in range(len(dataTypes)): # ['GSWLC1', 'DustPedia']
            if m == 1:
                if d == 3:
                    if b == 0:
                        continue # skip normSFH, power law, GSWLC1 (still running)
            print('starting', methods[m]+', '+dustPaths[d]+', '+dataTypes[b])
            path = dataPath+methods[m]+'/'+dustPaths[d]+'/'+dataTypes[b]+'/'
            bestParams = np.load(path+'bestParams.npy')
            bestParamsUnc = np.load(path+'bestParamsUnc.npy') # len(singleNames), numView, numParams, 2
            paramNames = np.load(path+'paramNames.npy')
            bestFluxes = np.load(path+'bestFluxes.npy')
            bestSpec = np.load(path+'bestSpec.npy')
            obsFluxes = np.load(path+'obsFluxes.npy')
            obsUnc = np.load(path+'obsUnc.npy')
            if b == 0:
                waveEff = wave_eff_GSWLC1
            else:
                waveEff = wave_eff_DustPedia
            if plotMass[m]:
                print('plotting mass truth')
                fitMasses = bestParams[:,:, paramNames=='total_mass'][:,:,0]
                #plotTruth(methods[m], dustList[d], dataTypes[b], 'total_mass', singleStellarMass, 
                #          fitMasses, bestParamsUnc[:,:,paramNames=='total_mass',:][:,:,0,:])
                # with error bars
                #plotResiduals(methods[m], dustList[d], dataTypes[b], 'total_mass', singleStellarMass, 
                #          fitMasses, bestParamsUnc[:,:,paramNames=='total_mass',:][:,:,0,:])
                # without error bars
                plotResiduals(methods[m], dustList[d], dataTypes[b], 'total_mass', singleStellarMass, fitMasses)
                # low mass moments
                lowMassFaceLogResidual = np.log10(fitMasses[lowMass, 0]) - np.log10(singleStellarMass[lowMass])
                lowMassEdgeLogResidual = np.log10(fitMasses[lowMass, 1]) - np.log10(singleStellarMass[lowMass])
                lowMassBiasMass[massPlotIndex, d,b,0] = np.mean(lowMassFaceLogResidual) # face
                lowMassBiasMass[massPlotIndex, d,b,1] = np.mean(lowMassEdgeLogResidual) # edge
                lowMassStdMass[massPlotIndex, d,b,0] = np.sqrt(np.mean(lowMassFaceLogResidual**2) - np.mean(lowMassFaceLogResidual)**2) 
                lowMassStdMass[massPlotIndex, d,b,1] = np.sqrt(np.mean(lowMassEdgeLogResidual**2) - np.mean(lowMassEdgeLogResidual)**2)
                # high mass moments
                highMassFaceLogResidual = np.log10(fitMasses[highMass, 0]) - np.log10(singleStellarMass[highMass])
                highMassEdgeLogResidual = np.log10(fitMasses[highMass, 1]) - np.log10(singleStellarMass[highMass])
                highMassBiasMass[massPlotIndex, d,b,0] = np.mean(highMassFaceLogResidual) # face
                highMassBiasMass[massPlotIndex, d,b,1] = np.mean(highMassEdgeLogResidual) # edge
                highMassStdMass[massPlotIndex, d,b,0] = np.sqrt(np.mean(highMassFaceLogResidual**2) - np.mean(highMassFaceLogResidual)**2) 
                highMassStdMass[massPlotIndex, d,b,1] = np.sqrt(np.mean(highMassEdgeLogResidual**2) - np.mean(highMassEdgeLogResidual)**2)
            if plotSFR[m]:
                print('plotting SFR truth')
                fitSFRs = np.zeros((len(singleNames), numView))
                fitSFRUncs = np.zeros((len(singleNames), numView, 2)) # [lower, upper]
                if m == 2:
                    faceMass = singleStellarMass
                    edgeMass = singleStellarMass
                    faceMassUncLower = np.zeros(len(singleStellarMass))
                    faceMassUncUpper = np.zeros(len(singleStellarMass))
                    edgeMassUncLower = np.zeros(len(singleStellarMass))
                    edgeMassUncUpper = np.zeros(len(singleStellarMass))
                else:
                    faceMass = bestParams[:, 0, paramNames == 'total_mass'][:,0]
                    edgeMass = bestParams[:, 1, paramNames == 'total_mass'][:,0]
                    faceMassUncLower = bestParamsUnc[:,0,paramNames == 'total_mass',0][:,0]
                    faceMassUncUpper = bestParamsUnc[:,0,paramNames == 'total_mass',1][:,0]
                    edgeMassUncLower = bestParamsUnc[:,1,paramNames == 'total_mass',0][:,0]
                    edgeMassUncUpper = bestParamsUnc[:,1,paramNames == 'total_mass',1][:,0]
                for g in range(len(singleNames)):
                    fitSFRs[g, 0] = getFitSFR(bestParams[g,0,:], paramNames, faceMass[g])
                    fitSFRs[g, 1] = getFitSFR(bestParams[g,1,:], paramNames, edgeMass[g])
                    # with error bars
                    # important to remember that a lower bound in z_fraction 
                    # is an upper bound in SFR (need to flip the sign and order)
                    # lower SFR bound is given by z_fraction + z_fraction_upper
                    # face-on, lower
                    #fitSFRUncs[g, 0, 0] = fitSFRs[g, 0] - getFitSFR(bestParams[g,0,:]+bestParamsUnc[g,0,:,1], 
                    #                      paramNames, faceMass[g]-faceMassUncLower[g]) 
                    # face-on, upper
                    #fitSFRUncs[g, 0, 1] = getFitSFR(bestParams[g,0,:]-bestParamsUnc[g,0,:,0], paramNames, 
                    #                      faceMass[g]+faceMassUncUpper[g]) - fitSFRs[g, 0]
                    # edge-on, lower
                    #fitSFRUncs[g, 1, 0] = fitSFRs[g, 1] - getFitSFR(bestParams[g,1,:]+bestParamsUnc[g,1,:,1], 
                    #                      paramNames, edgeMass[g]-edgeMassUncLower[g]) 
                    # edge-on, upper
                    #fitSFRUncs[g, 1, 1] = getFitSFR(bestParams[g,1,:]-bestParamsUnc[g,1,:,0], paramNames, 
                    #                      edgeMass[g]+edgeMassUncUpper[g]) - fitSFRs[g, 1]
                #plotTruth(methods[m], dustList[d], dataTypes[b], 'SFR', singleSFR, fitSFRs, fitSFRUncs)
                # with error bars
                #plotResiduals(methods[m], dustList[d], dataTypes[b], 'SFR', singleSFR, fitSFRs, fitSFRUncs)
                # without error bars
                plotResiduals(methods[m], dustList[d], dataTypes[b], 'SFR', singleSFR, fitSFRs)
                # low mass moments
                lowMassFaceLogResidual = np.log10(fitSFRs[lowMass, 0]) - np.log10(singleSFR[lowMass])
                lowMassEdgeLogResidual = np.log10(fitSFRs[lowMass, 1]) - np.log10(singleSFR[lowMass])
                lowMassBiasSFR[SFRPlotIndex, d,b,0] = np.mean(lowMassFaceLogResidual) # face
                lowMassBiasSFR[SFRPlotIndex, d,b,1] = np.mean(lowMassEdgeLogResidual) # edge
                lowMassStdSFR[SFRPlotIndex, d,b,0] = np.sqrt(np.mean(lowMassFaceLogResidual**2) - np.mean(lowMassFaceLogResidual)**2) 
                lowMassStdSFR[SFRPlotIndex, d,b,1] = np.sqrt(np.mean(lowMassEdgeLogResidual**2) - np.mean(lowMassEdgeLogResidual)**2)
                # high mass moments
                highMassFaceLogResidual = np.log10(fitSFRs[highMass, 0]) - np.log10(singleSFR[highMass])
                highMassEdgeLogResidual = np.log10(fitSFRs[highMass, 1]) - np.log10(singleSFR[highMass])
                highMassBiasSFR[SFRPlotIndex, d,b,0] = np.mean(highMassFaceLogResidual) # face
                highMassBiasSFR[SFRPlotIndex, d,b,1] = np.mean(highMassEdgeLogResidual) # edge
                highMassStdSFR[SFRPlotIndex, d,b,0] = np.sqrt(np.mean(highMassFaceLogResidual**2) - np.mean(highMassFaceLogResidual)**2) 
                highMassStdSFR[SFRPlotIndex, d,b,1] = np.sqrt(np.mean(highMassEdgeLogResidual**2) - np.mean(highMassEdgeLogResidual)**2)
            # calculate chi square
            # fluxes: [len(singleNames), numView, numFlux[b]]
            for g in range(len(singleNames)):
                if b == 1: # UV-IR only
                    chi2_UV_IR[g,m,d,0,:] = ((obsFluxes[g,0,:] - bestFluxes[g,0,:])**2) / obsUnc[g,0,:]**2  # face-on
                    chi2_UV_IR[g,m,d,1,:] = ((obsFluxes[g,1,:] - bestFluxes[g,1,:])**2) / obsUnc[g,1,:]**2  # edge-on
                    chi2_UV_Optical[g,m,d,0,:,b] = ((obsFluxes[g,0,:len(filterlist_GSWLC1)] - 
                        bestFluxes[g,0,:len(filterlist_GSWLC1)])**2) / obsUnc[g,0,:len(filterlist_GSWLC1)]**2  # face-on
                    chi2_UV_Optical[g,m,d,1,:,b] = ((obsFluxes[g,1,:len(filterlist_GSWLC1)] - 
                        bestFluxes[g,1,:len(filterlist_GSWLC1)])**2) / obsUnc[g,1,:len(filterlist_GSWLC1)]**2  # edge-on
                else:
                    chi2_UV_Optical[g,m,d,0,:,b] = ((obsFluxes[g,0,:] - bestFluxes[g,0,:])**2) / obsUnc[g,0,:]**2  # face-on
                    chi2_UV_Optical[g,m,d,1,:,b] = ((obsFluxes[g,1,:] - bestFluxes[g,1,:])**2) / obsUnc[g,1,:]**2  # edge-on
            if plotFits:
                for g in range(len(singleNames)):
                    galaxy = singleNames[g]
                    nameMask = names == galaxy
                    edgeIndex = np.argmin(axisRatios[nameMask])
                    faceIndex = np.argmax(axisRatios[nameMask])
                    if d == 0:
                        shiftedCatalogFaceSpec = (100**2 * catalogSpectrum_nodust[nameMask][faceIndex]) / dist[g]**2
                        shiftedCatalogEdgeSpec = (100**2 * catalogSpectrum_nodust[nameMask][edgeIndex]) / dist[g]**2
                        catalogWave = wave_nodust
                    else:
                        shiftedCatalogFaceSpec = (100**2 * catalogSpectrum[nameMask][faceIndex]) / dist[g]**2
                        shiftedCatalogEdgeSpec = (100**2 * catalogSpectrum[nameMask][edgeIndex]) / dist[g]**2
                        catalogWave = wave
                    plotFit(dustList[d], methods[m], waveEff, obsFluxes[g,0,:], catalogWave, shiftedCatalogFaceSpec, 
                            bestFluxes[g,0,:], obsUnc[g,0,:], fspsWave, bestSpec[g,0,:], 'face')
                    plotFit(dustList[d], methods[m], waveEff, obsFluxes[g,1,:], catalogWave, shiftedCatalogEdgeSpec, 
                            bestFluxes[g,1,:], obsUnc[g,1,:], fspsWave, bestSpec[g,1,:], 'edge')
    #if plotMass[m]:
    #    plotMoments(methods[m], 'Mass', lowMassBiasMass, lowMassStdMass, highMassBiasMass, highMassStdMass)
    #if plotSFR[m]:
    #    plotMoments(methods[m], 'SFR', lowMassBiasSFR, lowMassStdSFR, highMassBiasSFR, highMassStdSFR)

plotAllMoments(methods, highMassBiasMass, highMassStdMass, highMassBiasSFR, highMassStdSFR)
plotChi2(chi2_UV_IR, chi2_UV_Optical, wave_eff_DustPedia, wave_eff_GSWLC1)

print('done')





