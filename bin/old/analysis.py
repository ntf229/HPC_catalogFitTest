import prospect.io.read_results as reader
import os
import numpy as np
import argparse
import fitsio
import matplotlib.pyplot as plt
from sedpy.observate import load_filters, getSED

def makePlots(orientation, orientationIndex, res, obs, model, catalogSpectrum):
    # calculate catalog broadband photometry from spectrum
    f_lambda_cgs = (1/33333) * (1/(wave**2)) * catalogSpectrum
    mags = getSED(wave, f_lambda_cgs, filterlist=filterlist) # AB magnitudes
    maggies = 10**(-0.4*mags) # convert to maggies
    catalogFlux = maggies * 3631
    #chain = res["chain"] # Samples from the posterior probability (ndarray)
    #lnprobability = res["lnprobability"] # The posterior probability of each sample
    theta_labels = res["theta_labels"] # List of strings describing free parameters
    #run_params = res["run_params"] # A dictionary of arguments supplied to prospector at the time of the fit
    best = res["bestfit"] # The prediction of the data for the posterior sample with the highest `"lnprobability"`, as a dictionary
    #sps = reader.get_sps(res)
    wave_eff = [f.wave_effective for f in res['obs']['filters']]
    spec, phot, x = model.predict(best['parameter'], obs=obs, sps=sps)
    spec *= 3631 # convert from Maggies to Jy
    phot *= 3631 # convert from Maggies to Jy
    # generate fake obs to get full resolution spectra
    fake_obs = obs.copy()
    fake_obs['spectrum'] = None
    fake_obs['wavelength'] = wave
    full_spec = model.predict(best['parameter'], obs=fake_obs, sps=sps)[0]
    full_spec *= 3631 # convert from Maggies to Jy
    # plot spectra
    plt.figure(figsize=(10,8))
    plt.plot(np.log10(wave_eff), np.log10(phot), label="Bestfit Photometry", marker='x', linewidth=0, markersize=15, color='orange', alpha=0.5, markeredgewidth=5)
    plt.plot(np.log10(wave[waveMask]), np.log10(full_spec[waveMask]), label="Bestfit Spectrum", color='orange', linewidth=1.5, alpha=0.5)
    plt.plot(np.log10(wave_eff), np.log10(catalogFlux), label="True Photometry", marker='+', linewidth=0, markersize=20, color='blue', alpha=0.5, markeredgewidth=5)
    plt.plot(np.log10(wave[waveMask]), np.log10(catalogSpectrum[waveMask]), color='blue', label='True Spectrum', linewidth=1.5, alpha=0.5)
    plt.legend(fontsize=28)
    plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
    plt.ylabel(r'$\log_{10}(f_{\nu} \, / \, Jy)$',fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(plotPath+orientation+'SEDs.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    # plot parameter traces
    #if args.sampling == 'mcmc':
    #    chosen = np.random.choice(res["run_params"]["nwalkers"], size=int(args.nwalkers), replace=False)
    #    tracefig = reader.traceplot(res, figsize=(20,10), chains=chosen)
    #else:
    #    tracefig = reader.traceplot(res, figsize=(20,10))
    #plt.savefig(plotPath+orientation+'Traces.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    #plt.close()
    if args.sfh == 'delayed-tau':
        truths = len(theta_labels)*[-1e50]
        truths[0] = stellarMass[nameMask][orientationIndex]
        if eval(args.dust):
            truths[4] = 1e-50 # gets log scaled in corner plot (tau)  
        else:
            truths[3] = 1e-50 # gets log scaled in corner plot (tau)  
    else:
        truths = None
    # corner plot
    #imax = np.argmax(res['lnprobability'])
    #i, j = np.unravel_index(imax, res['lnprobability'].shape)
    #theta_max = res['chain'][i, j, :].copy()
    cornerfig = reader.subcorner(res, start=0, thin=1, truths=truths, fig=plt.subplots(len(theta_labels),len(theta_labels),figsize=(27,27))[0])
    plt.savefig(plotPath+orientation+'Corner.png')
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--sampling") # mcmc or dynesty
parser.add_argument("--dust") # include dust; True or False
parser.add_argument("--dust_type") # power_law, milky_way, calzetti, witt_and_gordon, kriek_and_conroy
parser.add_argument("--sfh") # delayed-tau, prospector-alpha
parser.add_argument("--fitType") # bands/spectra to use in Prospector fit (eg. GSWLC1)
parser.add_argument("--nwalkers") # number of walkers to use in Prospector fit
parser.add_argument("--niter") # number of steps taken by each walker
parser.add_argument("--galaxy") # name of galaxy
args = parser.parse_args()

codePath = "/home/ntf229/catalogFitTest/"
catalogPath = '/scratch/ntf229/NIHAO-SKIRT-Catalog/'
fitPath = '/scratch/ntf229/catalogFitTest/ProspectorFits/'
plotPath = '/scratch/ntf229/catalogFitTest/ProspectorPlots/'

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

nameMask = names == args.galaxy
faceIndex = np.argmax(axisRatios[nameMask])
edgeIndex = np.argmin(axisRatios[nameMask])

# wavelength mask for plots
waveMask = (wave >= 1e3) & (wave <= 1e7)

# set paths and select catalog spectra
if args.sampling == 'mcmc':
    if eval(args.dust):
        fitPath += 'mcmc/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
        plotPath += 'mcmc/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
        faceSpectrum = spectrum[nameMask][faceIndex]
        edgeSpectrum = spectrum[nameMask][edgeIndex]
    else:
        fitPath += 'mcmc/noDust/'+args.sfh+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
        plotPath += 'mcmc/noDust/'+args.sfh+'/'+args.fitType+'/numWalkers'+args.nwalkers+'/numIter'+args.niter+'/'+args.galaxy+'/'
        faceSpectrum = spectrum_nodust[nameMask][faceIndex]
        edgeSpectrum = spectrum_nodust[nameMask][edgeIndex]
elif args.sampling == 'dynesty':
    if eval(args.dust):
        fitPath += 'dynesty/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/'+args.galaxy+'/'
        plotPath += 'dynesty/dust/'+args.sfh+'/'+args.dust_type+'/'+args.fitType+'/'+args.galaxy+'/'
        faceSpectrum = spectrum[nameMask][faceIndex]
        edgeSpectrum = spectrum[nameMask][edgeIndex]
    else:
        fitPath += 'dynesty/noDust/'+args.sfh+'/'+args.fitType+'/'+args.galaxy+'/'
        plotPath += 'dynesty/noDust/'+args.sfh+'/'+args.fitType+'/'+args.galaxy+'/'
        faceSpectrum = spectrum_nodust[nameMask][faceIndex]
        edgeSpectrum = spectrum_nodust[nameMask][edgeIndex]
else:
    print('invalid sampling type')
    exit()

os.system('mkdir -p '+plotPath)

# load sps 
if args.sfh == 'delayed-tau':
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=1,
                       compute_vega_mags=False)
elif args.sfh == 'prospector-alpha':
        from prospect.sources import FastStepBasis
        sps = FastStepBasis()

# face-on plots
res, obs, model = reader.results_from(fitPath+'faceFit.h5', dangerous=False)
model = reader.read_model(fitPath+'model')[0]
makePlots('face', faceIndex, res, obs, model, faceSpectrum)

# edge-on plots
res, obs, model = reader.results_from(fitPath+'edgeFit.h5', dangerous=False)
model = reader.read_model(fitPath+'model')[0]
makePlots('edge', edgeIndex, res, obs, model, edgeSpectrum)

print('Done')


