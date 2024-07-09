# to run: 
# python params.py --emcee --outfile=fit --nwalkers=64

import time, sys
import math

import numpy as np
from sedpy.observate import load_filters, getSED

from os.path import expanduser
home = expanduser("~")

sys.path.insert(1, home+'/prospector') # import from home instead of anaconda 

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol':0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee fitting parameters
              'nwalkers':128,
              'nburn': [16, 32, 64],
              'niter': 512,
              'interval': 0.25,
              'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'unif', # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_stop_kwargs': {"post_thresh": 0.1},
              # Obs data parameters
              'objid':0,
              'phottable': 'demo_photometry.dat',
              'luminosity_distance': 1e-5,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# Model Definition
# --------------

def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
               add_neb=False, luminosity_distance=0.0, **extras):

    no_dust = True # if True, get rid of all dust related parameters

    # set dust model type by hand
    dust_type = 2

    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """

    # ----------------------------------------------------------- #
    # get dust model from config file

    # HAVING THIS MAKES THE MODEL NONETYPE
    # store config.txt inputs as dictionary 
    #new_dict = {}
    #with open(args.path+"/config.txt") as config_file:
    #  for line in config_file:
    #    (key, val) = line.split()
    #    new_dict[key] = val
    #config_file.close()
    #
    #for name, value in new_dict.items():
    #  split_name = name.split('/')
    #
    #  if split_name[0] == 'dust_model':
    #    config_dust_model = int(value)
    #    print('dust_model from config file:', config_dust_model)

    # ----------------------------------------------------------- #

    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.

    model_params = TemplateLibrary["parametric_sfh"]
    print('redshift is ', object_redshift)
    model_params["zred"]['init'] = object_redshift
    #print('template dust type:', model_params["dust_type"]['init'])

    # set dust type based on value set above
    model_params["dust_type"]['init'] = dust_type

    # set dust type to Chabrier (to match SKIRT)
    model_params["imf_type"]['init'] = 1

    #model_params["dust_type"] = {"N": 1, "isfree": False, "init": config_dust_model}
    #model_params["zred"] = {"N": 1, "isfree": False, "init": object_redshift}
    #print('dust type set by config file:', model_params["dust_type"]['init'])
    #print('redshift is ', model_params["zred"]['init'])




    # Add lumdist parameter.  If this is not added then the distance is
    # controlled by the "zred" parameter and a WMAP9 cosmology.
    if luminosity_distance > 0:
        print('applying luminosity distance', luminosity_distance)
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units":"Mpc"}

    # mass logzsol dust2 tage tau duste_umin duste_qpah duste_gamma

    # Adjust model initial values (only important for optimization or emcee)
    model_params["mass"]["init"] = 1e11
    model_params["logzsol"]["init"] = -1
    model_params["dust2"]["init"] = 5
    model_params["tage"]["init"] = 11
    model_params["tau"]["init"] = 25

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["mass"]["init_disp"] = 1e10
    model_params["logzsol"]["init_disp"] = 0.5
    model_params["dust2"]["init_disp"] = 2.5
    model_params["tage"]["init_disp"] = 2
    model_params["tau"]["init_disp"] = 12.5
    #model_params["tage"]["disp_floor"] = 2.0
    #model_params["dust2"]["disp_floor"] = 0.1

    # adjust priors
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e8, maxi=1e13) 
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.5)
    model_params["tage"]["prior"] = priors.LogUniform(mini=9, maxi=14)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=50) # can go to 100

    if no_dust:
      print('getting rid of all dust parameters')
      model_params["dust2"]["init"] = 0
      model_params["dust2"]["isfree"] = False
    else:
      print('applying dust parameters')
      if add_duste:
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
  
          #model_params["duste_qpah"]["init"] = 9.0282 # calculated from https://skirt.ugent.be/skirt9/class_draine_li_dust_mix.html#a397683515561f08b8aae8f6107900337 under detailed description Mdust/MH values
  
      # ------------- DUST MODELS -------------- #
  
      # Set dust type from config file
      #model_params["dust_type"]['init'] = config_dust_model
      print('dust type set by config file:', model_params["dust_type"]['init'])
  
      # power law with index dust index set by dust_index
      if model_params["dust_type"]['init'] == 0:
  
      	print('in the dust_type=0 if statement')
  
      	model_params["dust1"] = {"N": 1, "isfree": True, "init": 0.0, "units":"optical depth for young population"}
      	model_params["dust1"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)
      	
      	model_params["dust_index"] = {"N": 1, "isfree": True, "init": -0.7, "units":"Power law index of the attenuation curve"}
      	model_params["dust_index"]["prior"] = priors.TopHat(mini=-5, maxi=5) # just guessing the range
      	
      	model_params["dust1_index"] = {"N": 1, "isfree": True, "init": -1, "units":"Power law index of the attenuation curve for young stars"}
      	model_params["dust1_index"]["prior"] = priors.TopHat(mini=-5, maxi=5) # just guessing the range
  
      # Milky Way extinction law (with the R=AV/E(B−V) value given by mwr) parameterized by Cardelli et al. (1989), with variable UV bump strength
      if model_params["dust_type"]['init'] == 1:
  
        print('in the dust_type=1 if statement')
  
        model_params["dust1"] = {"N": 1, "isfree": True, "init": 0.0, "units":"optical depth for young population"}
        model_params["dust1"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)
  
        # mwr: The ratio of total to selective absorption which characterizes the MW extinction curve: R=AV/E(B−V)
        model_params["mwr"] = {"N": 1, "isfree": True, "init": 3.1}
        model_params["mwr"]["prior"] = priors.TopHat(mini=0.1, maxi=10.0)
  
        # uvb: Parameter characterizing the strength of the 2175A extinction feature with respect to the standard Cardelli et al. determination for the MW.
        model_params["uvb"] = {"N": 1, "isfree": True, "init": 1}
        model_params["uvb"]["prior"] = priors.TopHat(mini=0.1, maxi=10.0)
  
      # Calzetti et al. (2000) attenuation curve. Note that if this value is set then the dust attenuation is applied to all starlight equally (not split by age), and therefore the only relevant parameter is dust2, which sets the overall normalization
      if model_params["dust_type"]['init'] == 2:
        print('in the dust_type=2 if statement')
  
      # allows the user to access a variety of attenuation curve models from Witt & Gordon (2000) using the parameters wgp1 and wgp2. In this case the parameters dust1 and dust2 have no effect because the WG00 models specify the full attenuation curve.
      if model_params["dust_type"]['init'] == 3:
        print('in the dust_type=3 if statement')
        model_params["dust2"]["isfree"] = False
  
        # wgp1: Integer specifying the optical depth in the Witt & Gordon (2000) models
        model_params["wgp1"] = {"N": 1, "isfree": True, "init": 1}
        model_params["wgp1"]["prior"] = priors.TopHat(mini=1, maxi=18)
  
        # wgp2: Integer specifying the type of large-scale geometry and extinction curve
        model_params["wgp2"] = {"N": 1, "isfree": True, "init": 1}
        model_params["wgp2"]["prior"] = priors.TopHat(mini=1, maxi=6)
  
      # Kriek & Conroy (2013) attenuation curve. In this model the slope of the curve, set by the parameter dust_index, is linked to the strength of the UV bump
      if model_params["dust_type"]['init'] == 4:
        print('in the dust_type=4 if statement')
  
        # dust_index: Power law index of the attenuation curve.
        model_params["dust_index"] = {"N": 1, "isfree": True, "init": -0.7}
        model_params["dust_index"]["prior"] = priors.TopHat(mini=-5, maxi=5)
  
      # ---------------------------------------- #

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        print('fixing metallicity')
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if object_redshift != 0.0:
        print('applying redshift')
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_neb:
        # Add nebular emission (with fixed parameters)
        print('adding nebular emission')
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
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
    
    #filterlist = load_filters(['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 
    #               'galex_FUV', 'galex_NUV', 'wise_w1', 'wise_w2', 
    #               'wise_w3', 'wise_w4'])

    filterlist = load_filters(args.filters)

    #microns = np.load('{0}/wave.npy'.format(args.path))
    #angstroms = microns * 1e4
    angstroms = np.load('{0}/wave.npy'.format(args.path))

    spec = np.load('{0}/spec.npy'.format(args.path)) # units of Jy 
    f_lambda_cgs = (1/33333) * (1/(angstroms**2)) * spec 

    mags = getSED(angstroms, f_lambda_cgs, filterlist=filterlist) # AB magnitudes

    #print('AB mags', mags)
    #np.save('ab_mags.npy', mags)

    maggies = 10**(-0.4*mags) # convert to maggies

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
    obs['maggies_unc'] = obs['maggies'] * 0.07
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
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
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
    args = parser.parse_args()

    args.filters = args.filters.split(',')

    ## store config.txt inputs as dictionary 
    #d = {}
    #with open(args.path+"/config.txt") as f:
    #  for line in f:
    #    (key, val) = line.split()
    #    d[key] = val
    ##f.close()
#
    #for name, value in d.items():
    #  s = name.split('/')
#
    #  if s[0] == 'nwalkers':
    #    args.nwalkers = int(value)
    #    print('number of walkers set by config file:', args.nwalkers)
#
    #  if s[0] == 'filters':
    #    args.filters = value.split(',')
    #    print('filters from config file:', args.filters)
#
    #  #if s[0] == 'dust_model':
    #  #  args.dust_model = int(value)
    #  #  print('dust_model from config file:', args.dust_model)




    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print("model:", model)

    if args.debug:
      sys.exit()

    hfile = "{0}/fit.h5".format(args.path)
    output = fit_model(obs, model, sps, noise, **run_params)

    print('done fitting')

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    print('fit file written')

    try:
        hfile.close()
    except(AttributeError):
        pass

