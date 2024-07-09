# Make plots for delayed-tau and prospector-alpha SFHs,
# calzetti, milky way, power law, and kriek and conroy dust models,
# with GSWLC1 and DustPedia bandpasses, with and without dust
# This code is finds the mcmc samples with the highest mass and SFR residuals

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

# data = [galaxy, SFH, dust model, data selection, parameters, orientation]
# parameters = [mass, SFR, attenuated energy, emitted energy (by dust)]
# noDustData = [galaxy, SFH, data selection, no-dust parameters, orientation]
# no-dust parameters = [mass, SFR]

def compareEnergyBalance():
    os.system('mkdir -p '+plotPath+'compareEnergyBalance/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceAttEnergy = data[massMask, h, d, s, 2, 0]
                faceEmitEnergy = data[massMask, h, d, s, 3, 0]
                edgeAttEnergy = data[massMask, h, d, s, 2, 1]
                edgeEmitEnergy = data[massMask, h, d, s, 3, 1]
                #faceAttEnergy = data[:, h, d, s, 2, 0]
                #faceEmitEnergy = data[:, h, d, s, 3, 0]
                #edgeAttEnergy = data[:, h, d, s, 2, 1]
                #edgeEmitEnergy = data[:, h, d, s, 3, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogFaceAttEnergy[massMask]/catalogFaceEmitEnergy[massMask]), 
                            np.log10(faceAttEnergy/faceEmitEnergy), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogEdgeAttEnergy[massMask]/catalogEdgeEmitEnergy[massMask]), 
                            np.log10(edgeAttEnergy/edgeEmitEnergy), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                boxLength = np.amax(np.absolute([xlim, ylim]))
                plt.plot([-boxLength, boxLength], [0,0], color='k', alpha=0.3)
                plt.plot([0,0], [-boxLength, boxLength], color='k', alpha=0.3)
                plt.xlim([-boxLength, boxLength])
                plt.ylim([-boxLength, boxLength])
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Attenuated \, / \, Emitted)$', fontsize=28)
                plt.ylabel('Inferred '+r'$\log_{10}(Attenuated \, / \, Emitted)$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'compareEnergyBalance/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close()

def MassMassPlots():
    os.system('mkdir -p '+plotPath+'MassMass/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                #faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                #edgeSFR = data[:, h, d, s, 1, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogMass), 
                            np.log10(faceMass) - np.log10(catalogMass), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogMass), 
                            np.log10(edgeMass) - np.log10(catalogMass), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Mass [M_{\odot}])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(Mass [M_{\odot}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'MassMass/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close()    

def SFRMassPlots():
    os.system('mkdir -p '+plotPath+'SFRMass/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                #faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                #edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogMass), 
                            np.log10(faceSFR) - np.log10(catalogSFR), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogMass), 
                            np.log10(edgeSFR) - np.log10(catalogSFR), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Mass [M_{\odot}])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'SFRMass/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close()  

def sSFRMassPlots():
    os.system('mkdir -p '+plotPath+'sSFRMass/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogMass), 
                            np.log10(faceSFR/faceMass) - np.log10(catalogSFR/catalogMass), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogMass), 
                            np.log10(edgeSFR/edgeMass) - np.log10(catalogSFR/catalogMass), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Mass [M_{\odot}])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(sSFR [yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'sSFRMass/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close()     

def MassBalancePlots():
    os.system('mkdir -p '+plotPath+'MassBalance/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                #faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                #edgeSFR = data[:, h, d, s, 1, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogFaceAttEnergy/catalogFaceEmitEnergy), 
                            np.log10(faceMass) - np.log10(catalogMass), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogEdgeAttEnergy/catalogEdgeEmitEnergy), 
                            np.log10(edgeMass) - np.log10(catalogMass), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Attenuated \, / \, Emitted)$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(Mass [M_{\odot}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'MassBalance/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close()  

def SFRBalancePlots():
    os.system('mkdir -p '+plotPath+'SFRBalance/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                #faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                #edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogFaceAttEnergy/catalogFaceEmitEnergy), 
                            np.log10(faceSFR) - np.log10(catalogSFR), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogEdgeAttEnergy/catalogEdgeEmitEnergy), 
                            np.log10(edgeSFR) - np.log10(catalogSFR), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Attenuated \, / \, Emitted)$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'SFRBalance/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def sSFRBalancePlots():
    os.system('mkdir -p '+plotPath+'sSFRBalance/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogFaceAttEnergy/catalogFaceEmitEnergy), 
                            np.log10(faceSFR/faceMass) - np.log10(catalogSFR/catalogMass), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(catalogEdgeAttEnergy/catalogEdgeEmitEnergy), 
                            np.log10(edgeSFR/edgeMass) - np.log10(catalogSFR/catalogMass), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Attenuated \, / \, Emitted)$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(sSFR [yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'sSFRBalance/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def SFRAttEnergyPlots():
    os.system('mkdir -p '+plotPath+'SFRAttEnergy/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                faceAttEnergy = data[:, h, d, s, 2, 0]
                faceEmitEnergy = data[:, h, d, s, 3, 0]
                edgeAttEnergy = data[:, h, d, s, 2, 1]
                edgeEmitEnergy = data[:, h, d, s, 3, 1]
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy), 
                            np.log10(faceSFR) - np.log10(catalogSFR), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy), 
                            np.log10(edgeSFR) - np.log10(catalogSFR), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('Residual '+r'$\log_{10}(Attenuated \, Energy)$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'SFRAttEnergy/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def SFRAttFUVPlots():
    os.system('mkdir -p '+plotPath+'SFRAttFUV/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                #faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                #edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                #faceAttEnergy = data[:, h, d, s, 2, 0]
                #faceEmitEnergy = data[:, h, d, s, 3, 0]
                #edgeAttEnergy = data[:, h, d, s, 2, 1]
                #edgeEmitEnergy = data[:, h, d, s, 3, 1]
                #faceAttFUV = data[:, h, d, s, 4, 0]
                #faceAttNUV = data[:, h, d, s, 5, 0]
                #edgeAttFUV = data[:, h, d, s, 4, 1]
                #edgeAttNUV = data[:, h, d, s, 5, 1]
                faceFUV = data[:, h, d, s, 4, 0]
                faceNUV = data[:, h, d, s, 5, 0]
                noDustFaceFUV = noDustData[:, h, s, 2, 0]
                noDustFaceNUV = noDustData[:, h, s, 3, 0]
                faceAttFUV = noDustFaceFUV - faceFUV
                faceAttNUV = noDustFaceNUV - faceNUV
                edgeFUV = data[:, h, d, s, 4, 1]
                edgeNUV = data[:, h, d, s, 5, 1]
                noDustEdgeFUV = noDustData[:, h, s, 2, 1]
                noDustEdgeNUV = noDustData[:, h, s, 3, 1]
                edgeAttFUV = noDustEdgeFUV - edgeFUV
                edgeAttNUV = noDustEdgeNUV - edgeNUV
                catalogFaceAttFUV = noDustCatalogFaceFUV - catalogFaceFUV
                catalogEdgeAttFUV = noDustCatalogEdgeFUV - catalogEdgeFUV
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(faceAttFUV) - np.log10(catalogFaceAttFUV), 
                            np.log10(faceSFR) - np.log10(catalogSFR), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(edgeAttFUV) - np.log10(catalogEdgeAttFUV), 
                            np.log10(edgeSFR) - np.log10(catalogSFR), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('Residual '+r'$\log_{10}(Attenuated \, FUV \, [Jy])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'SFRAttFUV/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def SFRAttNUVPlots():
    os.system('mkdir -p '+plotPath+'SFRAttNUV/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                #faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                #edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                #faceAttEnergy = data[:, h, d, s, 2, 0]
                #faceEmitEnergy = data[:, h, d, s, 3, 0]
                #edgeAttEnergy = data[:, h, d, s, 2, 1]
                #edgeEmitEnergy = data[:, h, d, s, 3, 1]
                #faceAttFUV = data[:, h, d, s, 4, 0]
                #faceAttNUV = data[:, h, d, s, 5, 0]
                #edgeAttFUV = data[:, h, d, s, 4, 1]
                #edgeAttNUV = data[:, h, d, s, 5, 1]
                faceFUV = data[:, h, d, s, 4, 0]
                faceNUV = data[:, h, d, s, 5, 0]
                noDustFaceFUV = noDustData[:, h, s, 2, 0]
                noDustFaceNUV = noDustData[:, h, s, 3, 0]
                faceAttFUV = noDustFaceFUV - faceFUV
                faceAttNUV = noDustFaceNUV - faceNUV
                edgeFUV = data[:, h, d, s, 4, 1]
                edgeNUV = data[:, h, d, s, 5, 1]
                noDustEdgeFUV = noDustData[:, h, s, 2, 1]
                noDustEdgeNUV = noDustData[:, h, s, 3, 1]
                edgeAttFUV = noDustEdgeFUV - edgeFUV
                edgeAttNUV = noDustEdgeNUV - edgeNUV
                catalogFaceAttNUV = noDustCatalogFaceNUV - catalogFaceNUV
                catalogEdgeAttNUV = noDustCatalogEdgeNUV - catalogEdgeNUV
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(faceAttNUV) - np.log10(catalogFaceAttNUV), 
                            np.log10(faceSFR) - np.log10(catalogSFR), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(np.log10(edgeAttNUV) - np.log10(catalogEdgeAttNUV), 
                            np.log10(edgeSFR) - np.log10(catalogSFR), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('Residual '+r'$\log_{10}(Attenuated \, NUV \, [Jy])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'SFRAttNUV/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def sSFRDeltaSlopePlots():
    os.system('mkdir -p '+plotPath+'sSFRDeltaSlope/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                faceFUV = data[:, h, d, s, 4, 0]
                faceNUV = data[:, h, d, s, 5, 0]
                noDustFaceFUV = noDustData[:, h, s, 2, 0]
                noDustFaceNUV = noDustData[:, h, s, 3, 0]
                faceDeltaSlope = np.log10(noDustFaceFUV/noDustFaceNUV) - np.log10(faceFUV/faceNUV)
                edgeFUV = data[:, h, d, s, 4, 1]
                edgeNUV = data[:, h, d, s, 5, 1]
                noDustEdgeFUV = noDustData[:, h, s, 2, 1]
                noDustEdgeNUV = noDustData[:, h, s, 3, 1]
                edgeDeltaSlope = np.log10(noDustEdgeFUV/noDustEdgeNUV) - np.log10(edgeFUV/edgeNUV)
                catalogFaceDeltaSlope = np.log10(noDustCatalogFaceFUV/noDustCatalogFaceNUV) - np.log10(catalogFaceFUV/catalogFaceNUV)
                catalogEdgeDeltaSlope = np.log10(noDustCatalogEdgeFUV/noDustCatalogEdgeNUV) - np.log10(catalogEdgeFUV/catalogEdgeNUV)
                plt.figure(figsize=(10, 8))
                plt.scatter(faceDeltaSlope - catalogFaceDeltaSlope, 
                            np.log10(faceSFR/faceMass) - np.log10(catalogSFR/catalogMass), 
                            label="Face-on", color='blue', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                plt.scatter(edgeDeltaSlope - catalogEdgeDeltaSlope, 
                            np.log10(edgeSFR/edgeMass) - np.log10(catalogSFR/catalogMass), 
                            label="Edge-on", color='red', 
                            alpha=0.5, linewidth=0, s=200, marker='o')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('Residual '+r'$\Delta\log_{10}[f_{\nu}$(FUV) / '+r'$f_{\nu}$(NUV)'+r'$]$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(sSFR \, [yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'sSFRDeltaSlope/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def MassMassAttEnergyPlots(dust = True):
    if dust:
        path = 'MassMassAttEnergy/'
    else:
        path = 'noDustMassMassAttEnergy/'
    os.system('mkdir -p '+plotPath+path)
    for h in range(len(SFHs)):
        for s in range(len(dataSelection)):
            for d in range(len(dustModels)):
                if (not dust) & (d > 0):
                    break
                plt.figure(figsize=(10, 8))
                if dust:
                    faceMass = data[:, h, d, s, 0, 0]
                    edgeMass = data[:, h, d, s, 0, 1]
                    faceAttEnergy = data[:, h, d, s, 2, 0]
                    edgeAttEnergy = data[:, h, d, s, 2, 1]
                    residualFaceAttEnergy = np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy)
                    residualEdgeAttEnergy = np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy)
                    plt.scatter(np.log10(catalogMass), 
                                np.log10(faceMass) - np.log10(catalogMass), 
                                label="Face-on", 
                                alpha=0.5, linewidth=0, s=200, marker='o', c=residualFaceAttEnergy, cmap='rainbow')
                    plt.scatter(np.log10(catalogMass), 
                                np.log10(edgeMass) - np.log10(catalogMass), 
                                label="Edge-on", 
                                alpha=0.5, linewidth=0, s=200, marker='s', c=residualEdgeAttEnergy, cmap='rainbow')
                    cbar = plt.colorbar()
                    cbar.set_label(label='Residual '+r'$\log_{10}(Attenuated \, Energy)$', size=28)
                    cbar.ax.tick_params(labelsize=28)
                else:
                    faceMass = noDustData[:, h, s, 0, 0]
                    edgeMass = noDustData[:, h, s, 0, 1]
                    plt.scatter(np.log10(catalogMass), 
                                np.log10(faceMass) - np.log10(catalogMass), 
                                label="Face-on", 
                                alpha=0.5, linewidth=0, s=200, marker='o', color='k')
                    plt.scatter(np.log10(catalogMass), 
                                np.log10(edgeMass) - np.log10(catalogMass), 
                                label="Edge-on", 
                                alpha=0.5, linewidth=0, s=200, marker='s', color='k')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                plt.xlabel('True '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                if dust:
                    plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                    plt.savefig(plotPath+path+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                                dpi=300, bbox_inches='tight', pad_inches=0.5)
                else:
                    plt.title(SFHsClean[h]+', '+dataSelection[s], fontsize=28)
                    plt.savefig(plotPath+path+SFHs[h]+'_'+dataSelection[s]+'.png', 
                                dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def SFRMassAttEnergyPlots():
    os.system('mkdir -p '+plotPath+'SFRMassAttEnergy/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                faceAttEnergy = data[:, h, d, s, 2, 0]
                faceEmitEnergy = data[:, h, d, s, 3, 0]
                edgeAttEnergy = data[:, h, d, s, 2, 1]
                edgeEmitEnergy = data[:, h, d, s, 3, 1]
                residualFaceAttEnergy = np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy)
                residualEdgeAttEnergy = np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy)
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogMass), 
                            np.log10(faceSFR) - np.log10(catalogSFR), 
                            label="Face-on", 
                            alpha=0.5, linewidth=0, s=200, marker='o', c=residualFaceAttEnergy, cmap='rainbow')
                plt.scatter(np.log10(catalogMass), 
                            np.log10(edgeSFR) - np.log10(catalogSFR), 
                            label="Edge-on", 
                            alpha=0.5, linewidth=0, s=200, marker='s', c=residualEdgeAttEnergy, cmap='rainbow')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                cbar = plt.colorbar()
                cbar.set_label(label='Residual '+r'$\log_{10}(Attenuated \, Energy)$', size=28)
                cbar.ax.tick_params(labelsize=28)
                plt.xlabel('True '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'SFRMassAttEnergy/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def sSFRMassAttEnergyPlots():
    os.system('mkdir -p '+plotPath+'sSFRMassAttEnergy/')
    for h in range(len(SFHs)):
        for d in range(len(dustModels)):
            for s in range(len(dataSelection)):
                faceMass = data[:, h, d, s, 0, 0]
                faceSFR = data[:, h, d, s, 1, 0]
                edgeMass = data[:, h, d, s, 0, 1]
                edgeSFR = data[:, h, d, s, 1, 1]
                faceAttEnergy = data[:, h, d, s, 2, 0]
                faceEmitEnergy = data[:, h, d, s, 3, 0]
                edgeAttEnergy = data[:, h, d, s, 2, 1]
                edgeEmitEnergy = data[:, h, d, s, 3, 1]
                residualFaceAttEnergy = np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy)
                residualEdgeAttEnergy = np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy)
                plt.figure(figsize=(10, 8))
                plt.scatter(np.log10(catalogMass), 
                            np.log10(faceSFR/faceMass) - np.log10(catalogSFR/catalogMass), 
                            label="Face-on", 
                            alpha=0.5, linewidth=0, s=200, marker='o', c=residualFaceAttEnergy, cmap='rainbow')
                plt.scatter(np.log10(catalogMass), 
                            np.log10(edgeSFR/edgeMass) - np.log10(catalogSFR/catalogMass), 
                            label="Edge-on", 
                            alpha=0.5, linewidth=0, s=200, marker='s', c=residualEdgeAttEnergy, cmap='rainbow')
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.plot(xlim, [0,0], color='k', alpha=0.3)
                plt.plot([0,0], ylim, color='k', alpha=0.3)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.legend(fontsize=20)
                cbar = plt.colorbar()
                cbar.set_label(label='Residual '+r'$\log_{10}(Attenuated \, Energy)$', size=28)
                cbar.ax.tick_params(labelsize=28)
                plt.xlabel('True '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
                plt.ylabel('Residual '+r'$\log_{10}(sSFR \, [yr^{-1}])$', fontsize=28)
                plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(plotPath+'sSFRMassAttEnergy/'+SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close() 

def SFHMassAttEnergyPlots(dust=True):
    bins = np.asarray(['0', '0.1', '0.3', '1', '3', '6', '13.6'])
    for i in range(len(bins) - 1):
        if dust:
            path = 'SFHMassAttEnergy/'+bins[i]+'-'+bins[i+1]+'Gyrs/'
        else:
            path = 'noDustSFHMassAttEnergy/'+bins[i]+'-'+bins[i+1]+'Gyrs/'
        os.system('mkdir -p '+plotPath+path)
        catalogSFR = catalogSFH[:, i]
        for h in range(len(SFHs)):
            for s in range(len(dataSelection)):
                for d in range(len(dustModels)):
                    if (not dust) & (d > 0):
                        break
                    if dust:
                        faceMass = data[:, h, d, s, 0, 0]
                        faceSFR = fitSFHs[:, h, d, s, i, 0]
                        edgeMass = data[:, h, d, s, 0, 1]
                        edgeSFR = fitSFHs[:, h, d, s, i, 1]
                        faceAttEnergy = data[:, h, d, s, 2, 0]
                        #faceEmitEnergy = data[:, h, d, s, 3, 0]
                        edgeAttEnergy = data[:, h, d, s, 2, 1]
                        #edgeEmitEnergy = data[:, h, d, s, 3, 1]
                        residualFaceAttEnergy = np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy)
                        residualEdgeAttEnergy = np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy)
                    else:
                        faceMass = noDustData[:, h, s, 0, 0]
                        faceSFR = noDustFitSFHs[:, h, s, i, 0]
                        edgeMass = noDustData[:, h, s, 0, 1]
                        edgeSFR = noDustFitSFHs[:, h, s, i, 1]
                        #faceAttEnergy = noDustData[:, h, s, 2, 0]
                        #faceEmitEnergy = noDustData[:, h, s, 3, 0]
                        #edgeAttEnergy = noDustData[:, h, s, 2, 1]
                        #edgeEmitEnergy = noDustData[:, h, s, 3, 1]
                    #residualFaceAttEnergy = np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy)
                    #residualEdgeAttEnergy = np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy)
                    plt.figure(figsize=(10, 8))
                    plt.scatter(None, None, label="Face-on", alpha=0.5, linewidth=0, s=200, marker='o', color='k')
                    plt.scatter(None, None, label="Edge-on", alpha=0.5, linewidth=0, s=200, marker='s', color='k')
                    if dust:
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(faceSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='o', c=residualFaceAttEnergy, cmap='rainbow')
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(edgeSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='s', c=residualEdgeAttEnergy, cmap='rainbow')
                        cbar = plt.colorbar()
                        cbar.set_label(label=r'$\Delta \log_{10}(Attenuated \, Energy)$', size=28)
                        cbar.ax.tick_params(labelsize=28)
                    else:
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(faceSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='o', color='k')
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(edgeSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='s', color='k')
                    xlim = plt.xlim()
                    ylim = plt.ylim()
                    plt.plot(xlim, [0,0], color='k', alpha=0.3)
                    plt.plot([0,0], ylim, color='k', alpha=0.3)
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.legend(fontsize=20)
                    plt.xlabel('True '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
                    plt.ylabel(r'$\Delta \log_{10}(SFR_{'+bins[i]+'-'+bins[i+1]+'\, Gyrs}[M_{\odot} \, yr^{-1}])$', fontsize=28)
                    plt.xticks(fontsize=28)
                    plt.yticks(fontsize=28)
                    if dust:
                        plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                        plt.savefig(plotPath+path+
                                    SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                                    dpi=300, bbox_inches='tight', pad_inches=0.5)
                    else:
                        plt.title(SFHsClean[h]+', '+dataSelection[s], fontsize=28)
                        plt.savefig(plotPath+path+
                                    SFHs[h]+'_'+dataSelection[s]+'.png', 
                                    dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close() 

def VariableSFRMassAttEnergyPlots(dust = True):
    # SFR averaged over varying number of years
    bins = np.asarray(['0', '0.1', '0.3', '1', '3', '6', '13.6'])
    for i in range(len(bins) - 1):
        if dust:
            path = 'VariableSFRMassAttEnergy/'+bins[0]+'-'+bins[i+1]+'Gyrs/''VariableSFRMassAttEnergy/'+bins[0]+'-'+bins[i+1]+'Gyrs/'
        else:
            path = 'noDustVariableSFRMassAttEnergy/'+bins[0]+'-'+bins[i+1]+'Gyrs/'
        os.system('mkdir -p '+plotPath+path)
        catalogSFR = 0.
        for j in range(i+1): # loop through age bins up to current bin (i)
            catalogSFR += catalogSFH[:, j]*(float(bins[j+1])-float(bins[j]))*1e9 # add mass from 0 to current age bin
        catalogSFR /= (float(bins[i+1])-float(bins[0]))*1e9 # convert from mass to mass/year
        for h in range(len(SFHs)):
            for s in range(len(dataSelection)):
                for d in range(len(dustModels)):
                    if (not dust) & (d > 0):
                        break
                    if dust:
                        faceMass = data[:, h, d, s, 0, 0]
                        faceSFR = 0
                        for j in range(i+1): # loop through age bins up to current bin (i)
                            faceSFR += fitSFHs[:, h, d, s, j, 0]*(float(bins[j+1])-float(bins[j]))*1e9 # add mass from 0 to current age bin
                        faceSFR /= (float(bins[i+1])-float(bins[0]))*1e9 # convert from mass to mass/year
                        edgeMass = data[:, h, d, s, 0, 1]
                        edgeSFR = 0
                        for j in range(i+1): # loop through age bins up to current bin (i)
                            edgeSFR += fitSFHs[:, h, d, s, j, 1]*(float(bins[j+1])-float(bins[j]))*1e9 # add mass from 0 to current age bin
                        edgeSFR /= (float(bins[i+1])-float(bins[0]))*1e9 # convert from mass to mass/year
                        faceAttEnergy = data[:, h, d, s, 2, 0]
                        #faceEmitEnergy = data[:, h, d, s, 3, 0]
                        edgeAttEnergy = data[:, h, d, s, 2, 1]
                        #edgeEmitEnergy = data[:, h, d, s, 3, 1]
                        residualFaceAttEnergy = np.log10(faceAttEnergy) - np.log10(catalogFaceAttEnergy)
                        residualEdgeAttEnergy = np.log10(edgeAttEnergy) - np.log10(catalogEdgeAttEnergy)
                    else:
                        faceMass = noDustData[:, h, s, 0, 0]
                        faceSFR = 0
                        for j in range(i+1): # loop through age bins up to current bin (i)
                            faceSFR += noDustFitSFHs[:, h, s, j, 0]*(float(bins[j+1])-float(bins[j]))*1e9 # add mass from 0 to current age bin
                        faceSFR /= (float(bins[i+1])-float(bins[0]))*1e9 # convert from mass to mass/year
                        edgeMass = noDustData[:, h, s, 0, 1]
                        edgeSFR = 0
                        for j in range(i+1): # loop through age bins up to current bin (i)
                            edgeSFR += noDustFitSFHs[:, h, s, j, 1]*(float(bins[j+1])-float(bins[j]))*1e9 # add mass from 0 to current age bin
                        edgeSFR /= (float(bins[i+1])-float(bins[0]))*1e9 # convert from mass to mass/year
                        #faceAttEnergy = noDustData[:, h, s, 2, 0]
                        #faceEmitEnergy = noDustData[:, h, s, 3, 0]
                        #edgeAttEnergy = noDustData[:, h, s, 2, 1]
                        #edgeEmitEnergy = noDustData[:, h, s, 3, 1]
                    plt.figure(figsize=(10, 8))
                    plt.scatter(None, None, label="Face-on", alpha=0.5, linewidth=0, s=200, marker='o', color='k')
                    plt.scatter(None, None, label="Edge-on", alpha=0.5, linewidth=0, s=200, marker='s', color='k')
                    if dust:
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(faceSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='o', c=residualFaceAttEnergy, cmap='rainbow')
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(edgeSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='s', c=residualEdgeAttEnergy, cmap='rainbow')
                        cbar = plt.colorbar()
                        cbar.set_label(label=r'$\Delta \log_{10}(Attenuated \, Energy)$', size=28)
                        cbar.ax.tick_params(labelsize=28)
                    else:
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(faceSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='o', color='k')
                        plt.scatter(np.log10(catalogMass), 
                                    np.log10(edgeSFR) - np.log10(catalogSFR), 
                                    alpha=0.5, linewidth=0, s=200, marker='s', color='k')
                    xlim = plt.xlim()
                    ylim = plt.ylim()
                    plt.plot(xlim, [0,0], color='k', alpha=0.3)
                    plt.plot([0,0], ylim, color='k', alpha=0.3)
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.legend(fontsize=20)
                    plt.xlabel('True '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
                    plt.ylabel(r'$\Delta \log_{10}(SFR_{'+bins[0]+'-'+bins[i+1]+'\, Gyrs}[M_{\odot} \, yr^{-1}])$', fontsize=28)
                    plt.xticks(fontsize=28)
                    plt.yticks(fontsize=28)
                    if dust:
                        plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
                        plt.savefig(plotPath+path+
                                    SFHs[h]+'_'+dustModelsClean[d]+'_'+dataSelection[s]+'.png', 
                                    dpi=300, bbox_inches='tight', pad_inches=0.5)
                    else:
                        plt.title(SFHsClean[h]+', '+dataSelection[s], fontsize=28)
                        plt.savefig(plotPath+path+
                                    SFHs[h]+'_'+dataSelection[s]+'.png', 
                                    dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close() 

def delayedTauSFH(tau, tage, mass): 
    ''' 
    tau = e-folding time of the SFH in Gyrs
    Delayed-tau SFH is given by t * exp(-t/tau) where t and tau are in Gyrs
    This is parametrized such that t=0 corresponds to the beginning of star-formation (tage)
    To calculate the SFR in a time bin [oldEdge, youngEdge] where oldEdge > youngEdge, 
    we integrate this function (after normalizing) from t=tage-oldEdge to t=tage-youngEdge
    we return the SFH of length 6 with the youngest bins first
    '''
    bins = np.asarray([0.0, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6]) # in Gyrs
    SFH = np.zeros((len(bins) - 1))
    norm = tau*(tau-np.exp(-tage/tau)*(tage+tau)) # normalization term
    for i in range(len(SFH)):
        if tage < bins[i+1]:  
            break
        oldEdge = tage - bins[i+1]
        youngEdge = tage - bins[i]
        SFH[i] = tau*(tau + oldEdge)*np.exp(-oldEdge/tau) - tau*(tau + youngEdge)*np.exp(-youngEdge/tau)
        SFH[i] *= mass / norm # full integratal equals total mass
        SFH[i] /= (youngEdge - oldEdge)*1e9 # convert from mass to mass/year
    return SFH
        
def calcCatalogSFH(sfh, ages):
    ages /= 1e9 # convert from years to Gyrs
    fullAges = np.append(ages, 10**(np.log10(ages[1]) - np.log10(ages[0]) + np.log10(ages[-1]))) # includes right-most edge
    bins = np.asarray([0.0, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6]) # in Gyrs
    SFH = np.zeros((len(bins) - 1))
    index = 0 # SFH index 
    for i in range(len(fullAges) - 1):
        if i == (len(fullAges) - 2): # last loop
            SFH[index] += sfh[i] # already in M_sun
            SFH[index] /= (bins[index+1] - bins[index])*1e9 # convert from M_sun to M_sun / year
            break
        if fullAges[i+1] < bins[index+1]: # +1 = right bin edge
            #SFH[index] += sfh[i]*(fullAges[i+1] - fullAges[i])*1e9 # in M_sun
            SFH[index] += sfh[i] # already in M_sun
        else:
            SFH[index] /= (bins[index+1] - bins[index])*1e9 # convert from M_sun to M_sun / year
            index += 1
    return SFH
        
def delayedTauSFR(tau, tage, mass, intTime=0.1):    
    ''' 
    intTime = integration time for calculating the SFR in Gyrs
    tau = e-folding time of the SFH in Gyrs
    Delayed-tau SFH is given by t * exp(-t/tau) where t and tau are in Gyrs
    This is parametrized such that t=0 corresponds to the beginning of star-formation (tage)
    To calculate the SFR, we integrate this function (after normalizing) 
    from t=tage-intTime to t=tage
    '''
    norm = tau*(tau-np.exp(-tage/tau)*(tage+tau)) # normalization term
    SFR = tau*np.exp(-(tage-intTime)/tau)*((tage-intTime)+tau) - tau*np.exp(-tage/tau)*(tage+tau)
    SFR *= mass / norm # full integral equals total mass
    SFR /= intTime*1e9 # convert from mass to mass/year
    return SFR

def og_energyBalance(wave, spec, noDustSpec):
    c = 2.998e18 # speed of light in Anstroms per second
    freq = c / wave # in Hz
    att_mask = wave <= 2e4
    emit_mask = wave > 2e4
    attenuation = noDustSpec[att_mask] - spec[att_mask] # in Janskys
    emission = spec[emit_mask] - noDustSpec[emit_mask]
    attEnergy = -1*np.trapz(attenuation, freq[att_mask]) # 10^(−23) erg * s^(−1) * cm^(−2)⋅
    emitEnergy = -1*np.trapz(emission, freq[emit_mask]) # 10^(−23) erg * s^(−1) * cm^(−2)⋅
    return attEnergy, emitEnergy

def energyBalance(wave, spec, noDustSpec):
    ''' attenuated energy only includes bins in which
    the attenuation curve is positive, emitted energy
    only includes bins in which the attenuation curve 
    is negative '''
    c = 2.998e18 # speed of light in Anstroms per second
    freq = c / wave # in Hz
    fullAttCurve = noDustSpec - spec # includes emission (negative values)
    attSpectrum = fullAttCurve.copy()
    emitSpectrum = fullAttCurve.copy()
    attSpectrum[attSpectrum < 0] = 0
    emitSpectrum[emitSpectrum > 0] = 0
    # factor of -1 below because freq is in decreasing order
    attEnergy = -1*np.trapz(attSpectrum, freq) # units 10^(−23) erg * s^(−1) * cm^(−2)⋅
    emitEnergy = np.trapz(emitSpectrum, freq) # units 10^(−23) erg * s^(−1) * cm^(−2)⋅
    return attEnergy, emitEnergy

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

def findBestResiduals(sfhType, flatchain, catalogMass, catalogSFR):
    mass = np.zeros(len(flatchain))
    SFR = np.zeros(len(flatchain))
    if sfhType == 'delayed-tau':
        for c in range(len(flatchain)):
            tau = flatchain[c][theta_labels == 'tau']
            tage = flatchain[c][theta_labels == 'tage']
            mass[c] = flatchain[c][theta_labels == 'mass']
            SFR[c] = delayedTauSFR(tau, tage, mass[c], intTime=0.1)
    elif sfhType == 'prospector-alpha':
        for c in range(len(flatchain)):
            mass[c] = flatchain[c][theta_labels == 'total_mass']
            zFracArray = np.zeros(5)
            zFracArray[0] = flatchain[c][theta_labels == 'z_fraction_1']
            zFracArray[1] = flatchain[c][theta_labels == 'z_fraction_2']
            zFracArray[2] = flatchain[c][theta_labels == 'z_fraction_3']
            zFracArray[3] = flatchain[c][theta_labels == 'z_fraction_4']
            zFracArray[4] = flatchain[c][theta_labels == 'z_fraction_5']
            SFR[c] = transforms.zfrac_to_masses(mass[c], z_fraction=zFracArray,
                     agebins = agebins)[0] / 1e8
    loss = ( (np.log10(mass) - np.log10(catalogMass))**2 + (np.log10(SFR) - np.log10(catalogSFR))**2 )**(1/2)
    ind = np.argmin(loss)
    return ind, mass[ind], SFR[ind]

def makeResidualPlot(dust, orientation, sfhType, flatchain, lnprobability, catalogMass, catalogSFR):
    mass = np.zeros(len(flatchain))
    SFR = np.zeros(len(flatchain))
    probLimit = np.sort(lnprobability)[int(len(lnprobability)*0.1)]
    lnprobMask = lnprobability > probLimit # don't plot the lowest 10% lnprobability iterations 
    #lnprobMask = lnprobability > 0.9*np.amax(lnprobability) 
    maxProbInd = np.argmax(lnprobability)
    if sfhType == 'delayed-tau':
        tau = flatchain[:,theta_labels == 'tau']
        tage = flatchain[:,theta_labels == 'tage']
        mass = flatchain[:,theta_labels == 'mass']
        SFR = delayedTauSFR(tau, tage, mass, intTime=0.1)
    elif sfhType == 'prospector-alpha':
        for c in range(len(flatchain)):
            mass[c] = flatchain[c][theta_labels == 'total_mass']
            zFracArray = np.zeros(5)
            zFracArray[0] = flatchain[c][theta_labels == 'z_fraction_1']
            zFracArray[1] = flatchain[c][theta_labels == 'z_fraction_2']
            zFracArray[2] = flatchain[c][theta_labels == 'z_fraction_3']
            zFracArray[3] = flatchain[c][theta_labels == 'z_fraction_4']
            zFracArray[4] = flatchain[c][theta_labels == 'z_fraction_5']
            SFR[c] = transforms.zfrac_to_masses(mass[c], z_fraction=zFracArray,
                     agebins = agebins)[0] / 1e8
        #mass = flatchain[:,theta_labels == 'total_mass']
        #zFracArray = np.zeros(5)
        #zFracArray[0] = flatchain[:,theta_labels == 'z_fraction_1']
        #zFracArray[1] = flatchain[:,theta_labels == 'z_fraction_2']
        #zFracArray[2] = flatchain[:,theta_labels == 'z_fraction_3']
        #zFracArray[3] = flatchain[:,theta_labels == 'z_fraction_4']
        #zFracArray[4] = flatchain[:,theta_labels == 'z_fraction_5']
        #SFR = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
        #         agebins = agebins)[0] / 1e8
    plt.figure(figsize=(10, 8))
    plt.scatter(0,0,alpha=0) # plot invisible point at [0,0] to keep center within plot range
    plt.scatter(np.log10(mass[lnprobMask]) - np.log10(catalogMass), 
                np.log10(SFR[lnprobMask]) - np.log10(catalogSFR), 
                alpha=0.1, linewidth=0, s=70, marker='o', c=lnprobability[lnprobMask], cmap='rainbow')
    cbar = plt.colorbar()
    cbar.solids.set(alpha=1)    
    cbar.set_label(label='ln(posterior)', size=28)
    cbar.ax.tick_params(labelsize=28)
    plt.scatter(np.log10(mass[maxProbInd]) - np.log10(catalogMass), 
                np.log10(SFR[maxProbInd]) - np.log10(catalogSFR), 
                alpha=1, linewidth=0, s=70, marker='s', color='k')
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(xlim, [0,0], color='k', alpha=0.3)
    plt.plot([0,0], ylim, color='k', alpha=0.3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.legend(fontsize=20)
    plt.xlabel('Residual '+r'$\log_{10}(Mass \, [M_{\odot}])$', fontsize=28)
    plt.ylabel('Residual '+r'$\log_{10}(SFR [M_{\odot} \, yr^{-1}])$', fontsize=28)
    plt.title(SFHsClean[h]+', '+dustModelsClean[d]+', '+dataSelection[s], fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    os.system('mkdir -p '+plotPath+'MassSFRPosterior/'+dust+'/'+SFHs[h]+'_'+dustModels[d]+'_'+dataSelection[s]+'/')
    plt.savefig(plotPath+'MassSFRPosterior/'+dust+'/'+SFHs[h]+'_'+dustModels[d]+'_'+dataSelection[s]+'/'+galaxies[g]+'_'+orientation+'.png', 
                dpi=100, bbox_inches='tight', pad_inches=0.5)
    plt.close() 


parser = argparse.ArgumentParser()
parser.add_argument("--sampling") # mcmc or dynesty
parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit
parser.add_argument("--niter") # number of steps taken by each walker
args = parser.parse_args()

codePath = "/home/ntf229/catalogFitTest/"
catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
plotPath = '/scratch/ntf229/catalogFitTest/ProspectorBestResidualsPlots/'
dataPath = '/scratch/ntf229/catalogFitTest/ProspectorBestResidualsData/'

if args.sampling == 'mcmc':
    plotPath += 'mcmc/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'
    dataPath += 'mcmc/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'
elif args.sampling == 'dynesty':
    plotPath += 'dynesty/'
    dataPath += 'dynesty/'

os.system('mkdir -p '+plotPath)
os.system('mkdir -p '+dataPath)

#dustModels = ['calzetti', 'power_law', 'milky_way', 'witt_and_gordon', 'kriek_and_conroy']
#dustModelsClean = ['Calzetti', 'Power Law', 'Milky Way', 'Witt and Gordon', 'Kriek and Conroy']
SFHs = ['delayed-tau', 'prospector-alpha']
SFHsClean = ['Delayed-Tau', 'Prospector-Alpha']
dustModels = ['calzetti', 'power_law', 'milky_way', 'kriek_and_conroy']
dustModelsClean = ['Calzetti', 'Power Law', 'Milky Way', 'Kriek and Conroy']
dataSelection = ['GSWLC1', 'DustPedia']
params = ['Mass', 'SFR', 'Attenuated Energy', 'Emitted Energy']
noDustParams = ['Mass', 'SFR']
orientations = ['Face', 'Edge']

fspsWave = np.load(codePath+'python/full_rf_wavelengths.npy')

# load catalog data
galaxies = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='GALAXIES') # one per galaxy
summary = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SUMMARY')
wave = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='WAVE')
spectrum = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPEC')
spectrum_nodust = fitsio.read(catalogPath+'nihao-integrated-seds.fits', ext='SPECNODUST')

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

galaxies = ['g1.88e10','g1.89e10','g1.90e10','g2.34e10','g2.63e10','g2.64e10','g2.80e10','g2.83e10','g2.94e10',
            'g3.44e10','g3.67e10','g3.93e10','g4.27e10','g4.48e10','g4.86e10','g4.94e10','g4.99e10',
            'g5.05e10','g6.12e10','g6.37e10','g6.77e10','g6.91e10','g7.12e10','g8.89e10','g9.59e10','g1.05e11',
            'g1.08e11','g1.37e11','g1.52e11','g1.57e11','g1.59e11','g1.64e11','g2.04e11','g2.19e11','g2.39e11',
            'g2.41e11','g2.42e11','g2.54e11','g3.06e11','g3.21e11','g3.23e11','g3.49e11','g3.55e11','g3.59e11',
            'g3.71e11','g4.90e11','g5.02e11','g5.31e11','g5.36e11','g5.38e11','g5.46e11','g5.55e11','g6.96e11',
            'g7.08e11','g7.44e11','g7.55e11','g7.66e11','g8.06e11','g8.13e11','g8.26e11','g8.28e11','g1.12e12',
            'g1.77e12','g1.92e12','g2.79e12']

massMask = np.log10(singleStellarMass) > 9.5

catalogMass = np.zeros(len(galaxies))
catalogSFR = np.zeros(len(galaxies))
catalogFaceAttEnergy = np.zeros(len(galaxies))
catalogFaceEmitEnergy = np.zeros(len(galaxies))
#catalogFaceAttFUV = np.zeros(len(galaxies))
#catalogFaceAttNUV = np.zeros(len(galaxies))
#catalogFaceFUV = np.zeros(len(galaxies))
#catalogFaceNUV = np.zeros(len(galaxies))
#noDustCatalogFaceFUV = np.zeros(len(galaxies))
#noDustCatalogFaceNUV = np.zeros(len(galaxies))
catalogEdgeAttEnergy = np.zeros(len(galaxies))
catalogEdgeEmitEnergy = np.zeros(len(galaxies))
#catalogEdgeAttFUV = np.zeros(len(galaxies))
#catalogEdgeAttNUV = np.zeros(len(galaxies))
#catalogEdgeFUV = np.zeros(len(galaxies))
#catalogEdgeNUV = np.zeros(len(galaxies))
#noDustCatalogEdgeFUV = np.zeros(len(galaxies))
#noDustCatalogEdgeNUV = np.zeros(len(galaxies))
catalogSFH = np.zeros((len(galaxies), 6)) # prospector-alpha age bins

# loop over galaxies fill catalog arrays
for g in range(len(galaxies)):
    galaxy = galaxies[g]
    nameMask = names == galaxy
    singleNameMask = singleNames == galaxy
    faceIndex = np.argmax(axisRatios[nameMask])
    edgeIndex = np.argmin(axisRatios[nameMask])
    catalogMass[g] = stellarMass[nameMask][0]
    catalogSFR[g] = SFR[nameMask][0]
    #catalogFaceAttFUV[g] = flux_noDust[nameMask][faceIndex][0] - flux[nameMask][faceIndex][0]
    #catalogFaceAttNUV[g] = flux_noDust[nameMask][faceIndex][1] - flux[nameMask][faceIndex][1]
    #catalogEdgeAttFUV[g] = flux_noDust[nameMask][edgeIndex][0] - flux[nameMask][edgeIndex][0]
    #catalogEdgeAttNUV[g] = flux_noDust[nameMask][edgeIndex][1] - flux[nameMask][edgeIndex][1]
    #catalogFaceFUV[g] = flux[nameMask][faceIndex][0]
    #catalogFaceNUV[g] = flux[nameMask][faceIndex][1]
    #catalogEdgeFUV[g] = flux[nameMask][edgeIndex][0]
    #catalogEdgeNUV[g] = flux[nameMask][edgeIndex][1]
    #noDustCatalogFaceFUV[g] = flux_noDust[nameMask][faceIndex][0]
    #noDustCatalogFaceNUV[g] = flux_noDust[nameMask][faceIndex][1]
    #noDustCatalogEdgeFUV[g] = flux_noDust[nameMask][edgeIndex][0]
    #noDustCatalogEdgeNUV[g] = flux_noDust[nameMask][edgeIndex][1]
    catalogFaceAttEnergy[g], catalogFaceEmitEnergy[g] = energyBalance(wave, 
                                                        spectrum[nameMask][faceIndex], 
                                                        spectrum_nodust[nameMask][faceIndex])
    catalogEdgeAttEnergy[g], catalogEdgeEmitEnergy[g] = energyBalance(wave, 
                                                        spectrum[nameMask][edgeIndex], 
                                                        spectrum_nodust[nameMask][edgeIndex])
    catalogSFH[g, :] = calcCatalogSFH(sfh[singleNameMask, :][0], ages[singleNameMask, :][0])

if os.path.exists(dataPath+'data.npy') & os.path.exists(dataPath+'noDustData.npy'):
    if os.path.exists(dataPath+'fitSFHs.npy') & os.path.exists(dataPath+'noDustFitSFHs.npy'):
        print('loading data')
        data = np.load(dataPath+'data.npy')
        noDustData = np.load(dataPath+'noDustData.npy')
        fitSFHs = np.load(dataPath+'fitSFHs.npy')
        noDustFitSFHs = np.load(dataPath+'noDustFitSFHs.npy')
        #compareEnergyBalance()
        #MassMassPlots()
        #SFRMassPlots()
        #sSFRMassPlots()
        #MassBalancePlots()
        #SFRBalancePlots()
        #sSFRBalancePlots()
        #SFRAttEnergyPlots()
        #SFRAttFUVPlots()
        #SFRAttNUVPlots()
        #sSFRDeltaSlopePlots()
        #MassMassAttEnergyPlots()
        MassMassAttEnergyPlots(dust = False)
        #SFRMassAttEnergyPlots()
        #sSFRMassAttEnergyPlots()
        #SFHMassAttEnergyPlots()
        #VariableSFRMassAttEnergyPlots()
        #SFHMassAttEnergyPlots(dust = False)
        #VariableSFRMassAttEnergyPlots(dust = False)
        exit()

alphaBinEdges = np.asarray([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
alpha_agelims = np.log10(alphaBinEdges) + 9
agebins = np.array([alpha_agelims[:-1], alpha_agelims[1:]])
agebins = agebins.T

data = np.zeros((len(galaxies), len(SFHs), len(dustModels), len(dataSelection), len(params), len(orientations)))
noDustData = np.zeros((len(galaxies), len(SFHs), len(dataSelection), len(noDustParams), len(orientations)))

fitSFHs = np.zeros((len(galaxies), len(SFHs), len(dustModels), len(dataSelection), 6, len(orientations)))
noDustFitSFHs = np.zeros((len(galaxies), len(SFHs), len(dataSelection), 6, len(orientations)))

for h in range(len(SFHs)):
    if SFHs[h] == 'delayed-tau':
        sps = CSPSpecBasis(zcontinuous=1, compute_vega_mags=False)
    else:
        sps = FastStepBasis()
    for d in range(len(dustModels)):
        for s in range(len(dataSelection)):
            faceMass = np.zeros(len(galaxies))
            faceSFR = np.zeros(len(galaxies))
            noDustFaceMass = np.zeros(len(galaxies))
            noDustFaceSFR = np.zeros(len(galaxies))
            faceAttEnergy = np.zeros(len(galaxies))
            faceEmitEnergy = np.zeros(len(galaxies))
            #faceAttFUV = np.zeros(len(galaxies))
            #faceAttNUV = np.zeros(len(galaxies))
            #faceFUV = np.zeros(len(galaxies))
            #faceNUV = np.zeros(len(galaxies))
            #noDustFaceFUV = np.zeros(len(galaxies))
            #noDustFaceNUV = np.zeros(len(galaxies))
            edgeMass = np.zeros(len(galaxies))
            edgeSFR = np.zeros(len(galaxies))
            noDustEdgeMass = np.zeros(len(galaxies))
            noDustEdgeSFR = np.zeros(len(galaxies))
            edgeAttEnergy = np.zeros(len(galaxies))
            edgeEmitEnergy = np.zeros(len(galaxies))
            #edgeAttFUV = np.zeros(len(galaxies))
            #edgeAttNUV = np.zeros(len(galaxies))
            #edgeFUV = np.zeros(len(galaxies))
            #edgeNUV = np.zeros(len(galaxies))
            #noDustEdgeFUV = np.zeros(len(galaxies))
            #noDustEdgeNUV = np.zeros(len(galaxies))
            for g in range(len(galaxies)):
                galaxy = galaxies[g]
                fitPath = '/scratch/ntf229/catalogFitTest/ProspectorFits/'
                noDustFitPath = '/scratch/ntf229/catalogFitTest/ProspectorFits/'
                # set paths and select catalog spectra
                if args.sampling == 'mcmc':
                    fitPath += 'mcmc/dust/'+SFHs[h]+'/'+dustModels[d]+'/'+dataSelection[s]+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+galaxy+'/'
                    noDustFitPath += 'mcmc/noDust/'+SFHs[h]+'/'+dataSelection[s]+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+galaxy+'/'
                elif args.sampling == 'dynesty':
                    fitPath += 'dynesty/dust/'+SFHs[h]+'/'+dustModels[d]+'/'+dataSelection[s]+'/'+galaxy+'/'
                    noDustFitPath += 'dynesty/noDust/'+SFHs[h]+'/'+dataSelection[s]+'/'+galaxy+'/'
                else:
                    print('invalid sampling type')
                    exit()
                if os.path.exists(plotPath+'MassSFRPosterior/no-dust/'+SFHs[h]+'_'+dustModels[d]+'_'+dataSelection[s]+'/'+galaxies[g]+'_edge.png'):
                    print('skipping '+SFHs[h]+'_'+dustModels[d]+'_'+dataSelection[s]+'/'+galaxies[g]+'_edge')
                    continue
                # face-on, dust 
                res, obs, model = reader.results_from(fitPath+'faceFit.h5', dangerous=False)
                model = reader.read_model(fitPath+'model')[0]
                theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
                csz = res["chain"].shape
                flatchain = res["chain"].reshape(csz[0] * csz[1], csz[2])
                ind, bestMass, bestSFR = findBestResiduals(SFHs[h], flatchain, catalogMass[g], catalogSFR[g])
                makeResidualPlot('dust', 'face', SFHs[h], flatchain, res["lnprobability"].flatten(), catalogMass[g], catalogSFR[g])
                # edge-on dust
                res, obs, model = reader.results_from(fitPath+'edgeFit.h5', dangerous=False)
                model = reader.read_model(fitPath+'model')[0]
                theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
                csz = res["chain"].shape
                flatchain = res["chain"].reshape(csz[0] * csz[1], csz[2])
                ind, bestMass, bestSFR = findBestResiduals(SFHs[h], flatchain, catalogMass[g], catalogSFR[g])
                makeResidualPlot('dust', 'edge', SFHs[h], flatchain, res["lnprobability"].flatten(), catalogMass[g], catalogSFR[g])
                # face-on, no-dust 
                res, obs, model = reader.results_from(noDustFitPath+'faceFit.h5', dangerous=False)
                model = reader.read_model(noDustFitPath+'model')[0]
                theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
                csz = res["chain"].shape
                flatchain = res["chain"].reshape(csz[0] * csz[1], csz[2])
                ind, bestMass, bestSFR = findBestResiduals(SFHs[h], flatchain, catalogMass[g], catalogSFR[g])
                makeResidualPlot('no-dust', 'face', SFHs[h], flatchain, res["lnprobability"].flatten(), catalogMass[g], catalogSFR[g])
                # edge-on no-dust
                res, obs, model = reader.results_from(noDustFitPath+'edgeFit.h5', dangerous=False)
                model = reader.read_model(noDustFitPath+'model')[0]
                theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
                csz = res["chain"].shape
                flatchain = res["chain"].reshape(csz[0] * csz[1], csz[2])
                ind, bestMass, bestSFR = findBestResiduals(SFHs[h], flatchain, catalogMass[g], catalogSFR[g])
                makeResidualPlot('no-dust', 'edge', SFHs[h], flatchain, res["lnprobability"].flatten(), catalogMass[g], catalogSFR[g])
                continue
                #best = res["bestfit"]
                #bestParams = np.asarray(best["parameter"])
                #spec, phot = getSpectrum(bestParams, obs, sps, model) # in Jy, catalog wave 
                #faceFUV[g] = phot[0]
                #faceNUV[g] = phot[1]
                if SFHs[h] == 'delayed-tau':
                    tau = bestParams[theta_labels == 'tau']
                    tage = bestParams[theta_labels == 'tage']
                    mass = bestParams[theta_labels == 'mass']
                    faceMass[g] = mass
                    faceSFR[g] = delayedTauSFR(tau, tage, mass, intTime=0.1)
                    fitSFHs[g, h, d, s, :, 0] = delayedTauSFH(tau, tage, mass)
                elif SFHs[h] == 'prospector-alpha':
                    mass = bestParams[theta_labels == 'total_mass']
                    faceMass[g] = mass
                    zFracArray = np.zeros(5)
                    zFracArray[0] = bestParams[theta_labels == 'z_fraction_1']
                    zFracArray[1] = bestParams[theta_labels == 'z_fraction_2']
                    zFracArray[2] = bestParams[theta_labels == 'z_fraction_3']
                    zFracArray[3] = bestParams[theta_labels == 'z_fraction_4']
                    zFracArray[4] = bestParams[theta_labels == 'z_fraction_5']
                    faceSFR[g] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
                                 agebins = agebins)[0] / 1e8
                    fitSFHs[g, h, d, s, :, 0] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
                                                agebins = agebins)[:] 
                    fitSFHs[g, h, d, s, 0, 0] /= (alphaBinEdges[1] - alphaBinEdges[0])*1e9
                    fitSFHs[g, h, d, s, 1, 0] /= (alphaBinEdges[2] - alphaBinEdges[1])*1e9
                    fitSFHs[g, h, d, s, 2, 0] /= (alphaBinEdges[3] - alphaBinEdges[2])*1e9
                    fitSFHs[g, h, d, s, 3, 0] /= (alphaBinEdges[4] - alphaBinEdges[3])*1e9
                    fitSFHs[g, h, d, s, 4, 0] /= (alphaBinEdges[5] - alphaBinEdges[4])*1e9
                    fitSFHs[g, h, d, s, 5, 0] /= (alphaBinEdges[6] - alphaBinEdges[5])*1e9
                # face-on, no dust
                noDustRes, noDustObs, noDustModel = reader.results_from(noDustFitPath+'faceFit.h5', dangerous=False)
                noDustModel = reader.read_model(noDustFitPath+'model')[0]
                noDustTheta_labels = np.asarray(noDustRes["theta_labels"]) # List of strings describing free parameters
                noDustBest = noDustRes["bestfit"]
                noDustBestParams = np.asarray(noDustBest["parameter"]) 
                if SFHs[h] == 'delayed-tau':
                    noDustTau = noDustBestParams[noDustTheta_labels == 'tau']
                    noDustTage = noDustBestParams[noDustTheta_labels == 'tage']
                    noDustMass = noDustBestParams[noDustTheta_labels == 'mass']
                    noDustFaceMass[g] = noDustMass
                    noDustFaceSFR[g] = delayedTauSFR(noDustTau, noDustTage, noDustMass, intTime=0.1)
                    noDustFitSFHs[g, h, s, :, 0] = delayedTauSFH(noDustTau, noDustTage, noDustMass)
                elif SFHs[h] == 'prospector-alpha':
                    noDustMass = noDustBestParams[noDustTheta_labels == 'total_mass']
                    noDustFaceMass[g] = noDustMass
                    zFracArray = np.zeros(5)
                    zFracArray[0] = noDustBestParams[noDustTheta_labels == 'z_fraction_1']
                    zFracArray[1] = noDustBestParams[noDustTheta_labels == 'z_fraction_2']
                    zFracArray[2] = noDustBestParams[noDustTheta_labels == 'z_fraction_3']
                    zFracArray[3] = noDustBestParams[noDustTheta_labels == 'z_fraction_4']
                    zFracArray[4] = noDustBestParams[noDustTheta_labels == 'z_fraction_5']
                    noDustFaceSFR[g] = transforms.zfrac_to_masses(noDustMass, z_fraction=zFracArray,
                                       agebins = agebins)[0] / 1e8
                    if d == 0: # does not depend on dust model
                        noDustFitSFHs[g, h, s, :, 0] = transforms.zfrac_to_masses(noDustMass, z_fraction=zFracArray,
                                                       agebins = agebins)[:] 
                        noDustFitSFHs[g, h, s, 0, 0] /= (alphaBinEdges[1] - alphaBinEdges[0])*1e9
                        noDustFitSFHs[g, h, s, 1, 0] /= (alphaBinEdges[2] - alphaBinEdges[1])*1e9
                        noDustFitSFHs[g, h, s, 2, 0] /= (alphaBinEdges[3] - alphaBinEdges[2])*1e9
                        noDustFitSFHs[g, h, s, 3, 0] /= (alphaBinEdges[4] - alphaBinEdges[3])*1e9
                        noDustFitSFHs[g, h, s, 4, 0] /= (alphaBinEdges[5] - alphaBinEdges[4])*1e9
                        noDustFitSFHs[g, h, s, 5, 0] /= (alphaBinEdges[6] - alphaBinEdges[5])*1e9
                # now we use the best dusty parameters in the no dust model to calculate attenuation curves
                matchedParams = orderParams(bestParams, theta_labels, noDustTheta_labels)
                noDustSpec, noDustPhot = getSpectrum(matchedParams, obs, sps, noDustModel) # in Jy, catalog wave
                #faceAttEnergy[g], faceEmitEnergy[g] = energyBalance(wave, spec, noDustSpec)
                faceAttEnergy[g], faceEmitEnergy[g] = energyBalance(fspsWave, spec, noDustSpec)
                #faceAttFUV[g] = noDustPhot[0] - phot[0]
                #faceAttNUV[g] = noDustPhot[1] - phot[1]
                noDustFaceFUV[g] = noDustPhot[0]
                noDustFaceNUV[g] = noDustPhot[1]
                # edge-on, dust
                res, obs, model = reader.results_from(fitPath+'edgeFit.h5', dangerous=False)
                model = reader.read_model(fitPath+'model')[0]
                theta_labels = np.asarray(res["theta_labels"]) # List of strings describing free parameters
                best = res["bestfit"]
                bestParams = np.asarray(best["parameter"])
                spec, phot = getSpectrum(bestParams, obs, sps, model) # in Jy, catalog wave 
                edgeFUV[g] = phot[0]
                edgeNUV[g] = phot[1]
                if SFHs[h] == 'delayed-tau':
                    tau = bestParams[theta_labels == 'tau']
                    tage = bestParams[theta_labels == 'tage']
                    mass = bestParams[theta_labels == 'mass']
                    edgeMass[g] = mass
                    edgeSFR[g] = delayedTauSFR(tau, tage, mass, intTime=0.1)
                    fitSFHs[g, h, d, s, :, 1] = delayedTauSFH(tau, tage, mass)
                elif SFHs[h] == 'prospector-alpha':
                    mass = bestParams[theta_labels == 'total_mass']
                    edgeMass[g] = mass
                    zFracArray = np.zeros(5)
                    zFracArray[0] = bestParams[theta_labels == 'z_fraction_1']
                    zFracArray[1] = bestParams[theta_labels == 'z_fraction_2']
                    zFracArray[2] = bestParams[theta_labels == 'z_fraction_3']
                    zFracArray[3] = bestParams[theta_labels == 'z_fraction_4']
                    zFracArray[4] = bestParams[theta_labels == 'z_fraction_5']
                    edgeSFR[g] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
                                 agebins = agebins)[0] / 1e8
                    fitSFHs[g, h, d, s, :, 1] = transforms.zfrac_to_masses(mass, z_fraction=zFracArray,
                                                agebins = agebins)[:] 
                    fitSFHs[g, h, d, s, 0, 1] /= (alphaBinEdges[1] - alphaBinEdges[0])*1e9
                    fitSFHs[g, h, d, s, 1, 1] /= (alphaBinEdges[2] - alphaBinEdges[1])*1e9
                    fitSFHs[g, h, d, s, 2, 1] /= (alphaBinEdges[3] - alphaBinEdges[2])*1e9
                    fitSFHs[g, h, d, s, 3, 1] /= (alphaBinEdges[4] - alphaBinEdges[3])*1e9
                    fitSFHs[g, h, d, s, 4, 1] /= (alphaBinEdges[5] - alphaBinEdges[4])*1e9
                    fitSFHs[g, h, d, s, 5, 1] /= (alphaBinEdges[6] - alphaBinEdges[5])*1e9
                # edge-on, no dust
                noDustRes, noDustObs, noDustModel = reader.results_from(noDustFitPath+'edgeFit.h5', dangerous=False)
                noDustModel = reader.read_model(noDustFitPath+'model')[0]
                noDustTheta_labels = np.asarray(noDustRes["theta_labels"]) # List of strings describing free parameters
                noDustBest = noDustRes["bestfit"]
                noDustBestParams = np.asarray(noDustBest["parameter"])
                if SFHs[h] == 'delayed-tau':
                    noDustTau = noDustBestParams[noDustTheta_labels == 'tau']
                    noDustTage = noDustBestParams[noDustTheta_labels == 'tage']
                    noDustMass = noDustBestParams[noDustTheta_labels == 'mass']
                    noDustEdgeMass[g] = noDustMass
                    noDustEdgeSFR[g] = delayedTauSFR(noDustTau, noDustTage, noDustMass, intTime=0.1)
                    noDustFitSFHs[g, h, s, :, 1] = delayedTauSFH(noDustTau, noDustTage, noDustMass)
                elif SFHs[h] == 'prospector-alpha':
                    noDustMass = noDustBestParams[noDustTheta_labels == 'total_mass']
                    noDustEdgeMass[g] = noDustMass
                    zFracArray = np.zeros(5)
                    zFracArray[0] = noDustBestParams[noDustTheta_labels == 'z_fraction_1']
                    zFracArray[1] = noDustBestParams[noDustTheta_labels == 'z_fraction_2']
                    zFracArray[2] = noDustBestParams[noDustTheta_labels == 'z_fraction_3']
                    zFracArray[3] = noDustBestParams[noDustTheta_labels == 'z_fraction_4']
                    zFracArray[4] = noDustBestParams[noDustTheta_labels == 'z_fraction_5']
                    noDustEdgeSFR[g] = transforms.zfrac_to_masses(noDustMass, z_fraction=zFracArray,
                                 agebins = agebins)[0] / 1e8
                    if d == 0: # does not depend on dust model
                        noDustFitSFHs[g, h, s, :, 1] = transforms.zfrac_to_masses(noDustMass, z_fraction=zFracArray,
                                                       agebins = agebins)[:] 
                        noDustFitSFHs[g, h, s, 0, 1] /= (alphaBinEdges[1] - alphaBinEdges[0])*1e9
                        noDustFitSFHs[g, h, s, 1, 1] /= (alphaBinEdges[2] - alphaBinEdges[1])*1e9
                        noDustFitSFHs[g, h, s, 2, 1] /= (alphaBinEdges[3] - alphaBinEdges[2])*1e9
                        noDustFitSFHs[g, h, s, 3, 1] /= (alphaBinEdges[4] - alphaBinEdges[3])*1e9
                        noDustFitSFHs[g, h, s, 4, 1] /= (alphaBinEdges[5] - alphaBinEdges[4])*1e9
                        noDustFitSFHs[g, h, s, 5, 1] /= (alphaBinEdges[6] - alphaBinEdges[5])*1e9
                # now we use the best dusty parameters in the no dust model to calculate attenuation curves
                matchedParams = orderParams(bestParams, theta_labels, noDustTheta_labels)
                noDustSpec, noDustPhot = getSpectrum(matchedParams, obs, sps, noDustModel) # in Jy, catalog wave
                #edgeAttEnergy[g], edgeEmitEnergy[g] = energyBalance(wave, spec, noDustSpec)
                edgeAttEnergy[g], edgeEmitEnergy[g] = energyBalance(fspsWave, spec, noDustSpec)
                #edgeAttFUV[g] = noDustPhot[0] - phot[0]
                #edgeAttNUV[g] = noDustPhot[1] - phot[1]
                noDustEdgeFUV[g] = noDustPhot[0]
                noDustEdgeNUV[g] = noDustPhot[1]
            continue
            data[:, h, d, s, 0, 0] = faceMass
            data[:, h, d, s, 1, 0] = faceSFR
            data[:, h, d, s, 2, 0] = faceAttEnergy
            data[:, h, d, s, 3, 0] = faceEmitEnergy
            #data[:, h, d, s, 4, 0] = faceAttFUV
            #data[:, h, d, s, 5, 0] = faceAttNUV
            data[:, h, d, s, 4, 0] = faceFUV
            data[:, h, d, s, 5, 0] = faceNUV
            data[:, h, d, s, 0, 1] = edgeMass
            data[:, h, d, s, 1, 1] = edgeSFR
            data[:, h, d, s, 2, 1] = edgeAttEnergy
            data[:, h, d, s, 3, 1] = edgeEmitEnergy
            #data[:, h, d, s, 4, 1] = edgeAttFUV
            #data[:, h, d, s, 5, 1] = edgeAttNUV
            data[:, h, d, s, 4, 1] = edgeFUV
            data[:, h, d, s, 5, 1] = edgeNUV
            if d == 0: # does not depend on dust model
                noDustData[:, h, s, 0, 0] = noDustFaceMass
                noDustData[:, h, s, 1, 0] = noDustFaceSFR
                noDustData[:, h, s, 2, 0] = noDustFaceFUV
                noDustData[:, h, s, 3, 0] = noDustFaceNUV
                noDustData[:, h, s, 0, 1] = noDustEdgeMass
                noDustData[:, h, s, 1, 1] = noDustEdgeSFR
                noDustData[:, h, s, 2, 1] = noDustEdgeFUV
                noDustData[:, h, s, 3, 1] = noDustEdgeNUV

exit()

np.save(dataPath+'data.npy', data)
np.save(dataPath+'noDustData.npy', noDustData)
np.save(dataPath+'fitSFHs.npy', fitSFHs)
np.save(dataPath+'noDustFitSFHs.npy', noDustFitSFHs)

#compareEnergyBalance()
#MassMassPlots()
#SFRMassPlots()
#sSFRMassPlots()
#MassBalancePlots()
#SFRBalancePlots()
#sSFRBalancePlots()
#SFRAttEnergyPlots()
#SFRAttFUVPlots()
#SFRAttNUVPlots()
#sSFRDeltaSlopePlots()
#MassMassAttEnergyPlots()
MassMassAttEnergyPlots(dust = False)
#SFRMassAttEnergyPlots()
#sSFRMassAttEnergyPlots()
#SFHMassAttEnergyPlots()
#VariableSFRMassAttEnergyPlots()
#SFHMassAttEnergyPlots(dust = False)
#VariableSFRMassAttEnergyPlots(dust = False)

print('Done')


