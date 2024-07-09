
# to run: 
# python params.py --emcee --outfile=fit --nwalkers=64

import time, sys
import math
import numpy as np
from sedpy.observate import load_filters, getSED
from os.path import expanduser
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.models.templates import TemplateLibrary
from prospect.models import priors, sedmodel

home = expanduser("~")

# --------------
# Model Definition
# --------------

def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
               add_neb=False, luminosity_distance=0.0, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    models.sedmodel.SedModel object.
    """
    # prospector-alpha
    model_params = TemplateLibrary["alpha"]
    model_params["total_mass"] = {"N": 1, "isfree": True, "init": 1e11, "init_disp": 1e10, 
                                  "prior": priors.LogUniform(mini=1e6, maxi=1e12)}
    if args.method == 'SFH': # only works with dust (need at least 2 free parameters)
        model_params["z_fraction"] = {"N": 5, "isfree": False, "init": z_fraction}
        model_params["total_mass"] = {"N": 1, "isfree": False, "init": float(args.total_mass)}
        print('fitting with full SFH fixed')
    elif args.method == 'normSFH':
        model_params["z_fraction"] = {"N": 5, "isfree": False, "init": z_fraction}
        print('fitting with normalized SFH fixed (total mass free)')
    elif args.method == 'totalMass':
        model_params["total_mass"] = {"N": 1, "isfree": False, "init": float(args.total_mass)}
        print('fitting with total mass fixed')
    elif args.method == 'wild':
        print('fitting with all parameters free')
    ## set luminosity distance to 100 Mpcs to match SKIRT
    #model_params["lumdist"] = {"N": 1, "isfree": False,
    #                            "init": 100, "units":"Mpc"}
    # set luminosity distance to luminosity_distance
    print('setting luminosity distance to', luminosity_distance)
    model_params["lumdist"] = {"N": 1, "isfree": False,
                                "init": luminosity_distance, "units":"Mpc"}
    model_params["zred"]['init'] = object_redshift
    # set dust type based on parser args
    model_params["dust_type"]['init'] = dust_num
    if dust_num != 2:
        print('turning off dust_index parameter')
        model_params["dust_index"] = {"N": 1, "isfree": False, "init": 0.0}
    # turn off agn parameters
    model_params["fagn"] = {"N": 1, "isfree": False, "init": 0.0}
    model_params["agn_tau"] = {"N": 1, "isfree": False, "init": 0.0}
    model_params["add_agn_dust"] = {"N": 1, "isfree": False, "init": False}
    model_params["dust_ratio"] = {"N": 1, "isfree": False, "init": 0.0}
    # set dust type to Chabrier (to match NIHAO-SKIRT-Catalog)
    model_params["imf_type"]['init'] = 1
    model_params["logzsol"] = {"N": 1, "isfree": True, "init": 0.0,
                               "prior": priors.TopHat(mini=-2.0, maxi=1.0)}
    if eval(args.dust):
        print('applying dust parameters')
        # Add dust emission (with fixed dust SED parameters)
        print('adding dust emission')
        model_params.update(TemplateLibrary["dust_emission"])
        model_params["duste_umin"]["isfree"] = True
        model_params["duste_qpah"]["isfree"] = True
        model_params["duste_gamma"]["isfree"] = True
        model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=25.0)
        model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)
        model_params["duste_gamma"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
        model_params["duste_umin"]['init'] = 12.5
        model_params["duste_qpah"]['init'] = 5
        model_params["duste_gamma"]['init'] = 0.5
        model_params["duste_umin"]['init_disp'] = 6.25
        model_params["duste_qpah"]['init_disp'] = 2.5
        model_params["duste_gamma"]['init_disp'] = 0.25  
        # ------------- DUST MODELS -------------- #
        # power law 
        if model_params["dust_type"]['init'] == 0:
            print('in the dust_type=0 if statement')
            model_params["dust1"] = {"N": 1, "isfree": True, "init": 1.0, "units":"optical depth for young population"}
            model_params["dust1"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["dust2"] = {"N": 1, "isfree": True, "init": 1.0}
            model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["dust_index"] = {"N": 1, "isfree": True, "init": -0.7, "units":"Power law index of the attenuation curve"}
            model_params["dust_index"]["prior"] = priors.TopHat(mini=-3, maxi=0.5) 
            model_params["dust1_index"] = {"N": 1, "isfree": True, "init": -1, "units":"Power law index of the attenuation curve for young stars"}
            model_params["dust1_index"]["prior"] = priors.TopHat(mini=-3, maxi=0.5) 
            #model_params["frac_nodust"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_nodust"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
            #model_params["frac_obrun"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_obrun"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
            #model_params["dust_tesc"] = {"N": 1, "isfree": True, "init": 7.0} # in log10(years)
            #model_params["dust_tesc"]["prior"] = priors.TopHat(mini=6.0, maxi=7.7)
        # Milky Way extinction law (with the R=AV/E(B−V) value given by mwr) parameterized by Cardelli et al. (1989), with variable UV bump strength
        if model_params["dust_type"]['init'] == 1:
            print('in the dust_type=1 if statement')
            model_params["dust1"] = {"N": 1, "isfree": True, "init": 1.0, "units":"optical depth for young population"}
            model_params["dust1"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["dust2"] = {"N": 1, "isfree": True, "init": 1.0}
            model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["mwr"] = {"N": 1, "isfree": True, "init": 3.1}
            model_params["mwr"]["prior"] = priors.TopHat(mini=0.1, maxi=10.0)
            model_params["uvb"] = {"N": 1, "isfree": True, "init": 1.0}
            model_params["uvb"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)
            model_params["dust1_index"] = {"N": 1, "isfree": True, "init": -1, "units":"Power law index of the attenuation curve for young stars"}
            model_params["dust1_index"]["prior"] = priors.TopHat(mini=-3, maxi=0.5) 
            #model_params["frac_nodust"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_nodust"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
            #model_params["frac_obrun"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_obrun"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
            #model_params["dust_tesc"] = {"N": 1, "isfree": True, "init": 7.0} # in log10(years)
            #model_params["dust_tesc"]["prior"] = priors.TopHat(mini=6.0, maxi=7.7)
        # Calzetti et al. (2000) attenuation curve. Note that if this value is set then the dust attenuation is applied to all starlight equally (not split by age), and therefore the only relevant parameter is dust2, which sets the overall normalization
        if model_params["dust_type"]['init'] == 2:
            print('in the dust_type=2 if statement')
            model_params["dust2"] = {"N": 1, "isfree": True, "init": 1.0}
            model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["dust_index"] = {"N": 1, "isfree": False, "init": -0.7}
            #model_params["frac_nodust"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_nodust"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
        # Witt & Gordon (2000) using the parameters wgp1 and wgp2. In this case the parameters dust2 has no effect because the WG00 models specify the full attenuation curve.
        if model_params["dust_type"]['init'] == 3:
            print('in the dust_type=3 if statement')
            model_params["dust1"] = {"N": 1, "isfree": False, "init": 0.0}
            model_params["dust2"] = {"N": 1, "isfree": False, "init": 0.0}
            model_params["dust_index"] = {"N": 1, "isfree": False, "init": -0.7}
            model_params["wgp1"] = {"N": 1, "isfree": True, "init": 1}
            model_params["wgp1"]["prior"] = priors.TopHat(mini=1, maxi=18)
            model_params["wgp2"] = {"N": 1, "isfree": True, "init": 1}
            model_params["wgp2"]["prior"] = priors.TopHat(mini=1, maxi=6)
        # Kriek & Conroy (2013) attenuation curve. In this model the slope of the curve, set by the parameter dust_index, is linked to the strength of the UV bump
        if model_params["dust_type"]['init'] == 4:
            print('in the dust_type=4 if statement')
            model_params["dust1"] = {"N": 1, "isfree": True, "init": 1.0, "units":"optical depth for young population"}
            model_params["dust1"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["dust2"] = {"N": 1, "isfree": True, "init": 1.0}
            model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=5.0)
            model_params["dust_index"] = {"N": 1, "isfree": True, "init": -0.7}
            model_params["dust_index"]["prior"] = priors.TopHat(mini=-3, maxi=0.5)
            model_params["dust1_index"] = {"N": 1, "isfree": True, "init": -1, "units":"Power law index of the attenuation curve for young stars"}
            model_params["dust1_index"]["prior"] = priors.TopHat(mini=-3, maxi=0.5) 
            #model_params["frac_nodust"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_nodust"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
            #model_params["frac_obrun"] = {"N": 1, "isfree": True, "init": 0.0}
            #model_params["frac_obrun"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
            #model_params["dust_tesc"] = {"N": 1, "isfree": True, "init": 7.0} # in log10(years)
            #model_params["dust_tesc"]["prior"] = priors.TopHat(mini=6.0, maxi=7.7)
        # ---------------------------------------- #
    else:
        print('getting rid of all dust parameters')
        model_params["add_dust_emission"] = {"N": 1, "isfree": False, "init": False}
        model_params["dust1"] = {"N": 1, "isfree": False, "init": 0.}
        model_params["dust2"] = {"N": 1, "isfree": False, "init": 0.}
        model_params["dust_index"] = {"N": 1, "isfree": False, "init": -0.7}
        model_params["duste_umin"] = {"N": 1, "isfree": False, "init": 12.5}
        model_params["duste_qpah"] = {"N": 1, "isfree": False, "init": 5}
        model_params["duste_gamma"] = {"N": 1, "isfree": False, "init": 0.5}
        model_params["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
        model_params["add_neb_continuum"] = {"N": 1, "isfree": False, "init": False}
        model_params["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    model = sedmodel.SedModel(model_params)
    return model

# --------------
# Observational Data
# --------------

def build_obs(objid=0, luminosity_distance=None, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.
    
    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """
    from prospect.utils.obsutils import fix_obs
    filterlist = load_filters(args.filters)
    #filter_errors = np.asarray(args.filter_errors.split(',')).astype(np.float32)
    angstroms = np.load('{0}/wave.npy'.format(args.path))
    spec = np.load('{0}/spec.npy'.format(args.path)) # units of Jy 
    f_lambda_cgs = (1/33333) * (1/(angstroms**2)) * spec 
    mags = getSED(angstroms, f_lambda_cgs, filterlist=filterlist) # AB magnitudes
    maggies = 10**(-0.4*mags) # convert to maggies
    # errors given by:
    # sigma = sqrt(c + alpha*flux + (calib*flux)**2) with fluxes in Jy
    # c and alpha from fits of DustPedia
    # calib from Table 1 of Clark et al
    if len(filterlist) == 7:
        # GSWLC1
        # u,g,r,i,z,FUV,NUV
        c = [1.0962e-8, 1.6421e-8, 2.5926e-8, 1.2963e-7, 9.0480e-7, 1.4566e-10, 4.588e-10]
        alpha = [9.6788e-5, 2.2165e-5, 2.8862e-5, 6.7169e-5, 2.2377e-4, 4.3922e-6, 2.4238e-6] 
        calib = [0.013, 0.008, 0.008, 0.007, 0.008, 0.045, 0.027]
    elif len(filterlist) == 27:
        # DustPedia
        # u,g,r,i,z,FUV,NUV,
        # J,H,K,W1,W2,W3,W4,
        # irac1,irac2,irac3,irac4,
        # mips24,mips70,mips160,pacs70
        # pacs100,pacs160,spire250
        # spire350,spire500
        c = [1.0962e-8, 1.6421e-8, 2.5926e-8, 1.2963e-7, 9.0480e-7, 1.4566e-10, 4.588e-10,
             1.0103e-5, 1.1989e-5, 2.2551e-5, 3.0633e-7, 4.3207e-7, 3.0377e-6, 3.5885e-5,
             6.4139e-7, 4.2164e-7, 6.3356e-6, 3.1927e-6, 
             2.561e-5, 7.9693e-3, 1.5751e-2, 1.927e-1, 
             1.0821e-1, 1.3729e-1, 1.0465e-2, 
             6.0181e-3, 1.9612e-3]
        alpha = [9.6788e-5, 2.2165e-5, 2.8862e-5, 6.7169e-5, 2.2377e-4, 4.3922e-6, 2.4238e-6,
                 1.4023e-3, 8.485e-3, 3.6033e-3, 5.4390e-5, 1.2953e-4, 7.0860e-4, 2.4209e-3,
                 4.1288e-5, 5.8425e-5, 4.8108e-4, 1.4007e-4, 
                 9.4577e-4, 2.0658e-2, 1.1444e-1, 1.2964e-1, 
                 1.7251e-1, 1.1827e-1, 1.2453e-2, 
                 1.1519e-2, 8.4129e-3] 
        calib = [0.013, 0.008, 0.008, 0.007, 0.008, 0.045, 0.027,
                 0.017, 0.019, 0.019, 0.029, 0.034, 0.046, 0.056,
                 0.03, 0.03, 0.03, 0.03, 
                 0.05, 0.1, 0.12, 0.07, 
                 0.07, 0.07, 0.055,
                 0.055, 0.055]          
    else:
        print('invalid number of filters')
        exit()
    uncertainties = np.sqrt(c + alpha*(maggies*3631) + (calib*maggies*3631)**2) / 3631
    wave_eff = np.zeros(len(filterlist))
    for i in range(len(filterlist)):
        filterlist[i].get_properties()
        wave_eff[i] = filterlist[i].wave_effective
    print('maggies', maggies)
    print('effective wavelengths', wave_eff)
    # Build output dictionary.
    obs = {}
    obs['filters'] = filterlist
    obs['maggies'] = maggies
    #obs['maggies_unc'] = obs['maggies'] * 0.07
    #obs['maggies_unc'] = obs['maggies'] * filter_errors
    obs['maggies_unc'] = uncertainties
    obs['phot_mask'] = np.isfinite(np.squeeze(maggies))
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['objid'] = None
    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)
    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    # prospector-alpha
    from prospect.sources import FastStepBasis
    sps = FastStepBasis()
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true", default=True,
                        help="If set, add dust emission to the model.")
    parser.add_argument('--luminosity_distance', type=float, default=100,
                        help=("Luminosity distance in Mpc. Defaults to 10pc "
                              "(for case of absolute mags)"))
    parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")
    parser.add_argument("--path")
    parser.add_argument("--filters")
    #parser.add_argument("--filter_errors")
    #parser.add_argument("--nwalkers")
    #parser.add_argument("--niter")
    parser.add_argument("--dust")
    parser.add_argument("--dust_type")
    parser.add_argument("--total_mass")
    parser.add_argument("--z_frac0")
    parser.add_argument("--z_frac1")
    parser.add_argument("--z_frac2")
    parser.add_argument("--z_frac3")
    parser.add_argument("--z_frac4")
    parser.add_argument("--method")
    parser.add_argument("--numParams")
    args = parser.parse_args()

    z_fraction = np.array([args.z_frac0, args.z_frac1, args.z_frac2, args.z_frac3, 
                           args.z_frac4], dtype = np.float32)

    args.filters = args.filters.split(',')

    if args.dust_type == 'power_law':
        dust_num = 0
    elif args.dust_type == 'cardelli':
        dust_num = 1
    elif args.dust_type == 'calzetti':
        dust_num = 2
    elif args.dust_type == 'witt_and_gordon':
        dust_num = 3
    elif args.dust_type == 'kriek_and_conroy':
        dust_num = 4
    else:
        print('invalid dust model')
        exit()
    
    run_params = vars(args)
    # mcmc
    run_params['nwalkers'] = int(args.nwalkers) 
    run_params['niter'] = int(args.niter) 
    # dynesty
    run_params['nested_bound'] = 'multi'
    run_params['nested_sample'] = 'auto' 
    run_params['nested_walks'] = 30 # minimum number of steps before proposing new live point
    run_params['nested_bootstrap'] = 0 
    #run_params['nested_nlive_init'] = int(50*int(args.numParams))
    #run_params['nested_nlive_batch'] = int(50*int(args.numParams))
    # doing a special run with 4x live points for normSFH, power law, GSWLC1:
    run_params['nested_nlive_init'] = int(200*int(args.numParams))
    run_params['nested_nlive_batch'] = int(200*int(args.numParams))
    run_params['nested_dlogz_init'] = 0.02
    # default dynesty behavior for update_interval
    if int(args.numParams) < 10:
        run_params['nested_update_interval'] = 1.5  
    else:
        run_params['nested_update_interval'] = 0.15 * run_params['nested_walks']
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print("model:", model)

    if args.debug:
      sys.exit()

    hfile = "{0}/fit.h5".format(args.path)
    mfile = "{0}/model".format(args.path)
    output = fit_model(obs, model, sps, noise, **run_params)

    print('done fitting')

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    print('fit file written')

    writer.write_model_pickle(mfile, model)

    try:
        hfile.close()
    except(AttributeError):
        pass

