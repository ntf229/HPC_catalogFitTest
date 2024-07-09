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
    model_params = TemplateLibrary["alpha"]
    # set luminosity distance to 100 Mpcs and z=0 to match NIHAO-SKIRT-Catalog
    model_params["lumdist"] = {"N": 1, "isfree": False,
                                "init": 100, "units":"Mpc"}
    model_params["zred"] = {"N":1, "isfree": False, "init": 0.0}
    # change total mass prior mini from 1e8 to 1e5 and maxi from 1e12 to 1e14 (paper)
    model_params["total_mass"]["prior"] = priors.LogUniform(mini=1e5, maxi=1e14)
    model_params["total_mass"]["init"] = 1e11
    model_params["total_mass"]["init_disp"] = 1e10
    # change dust_index prior mini from -2 to -2.2 and maxi from 0.5 to 0.4 (paper)
    model_params["dust_index"]["prior"] = priors.TopHat(mini=-2.2, maxi=0.4)
    # change IMF from Kroupa to Chabrier (paper, also matches NIHAO-SKIRT-Catalog)
    model_params["imf_type"] = {"N": 1, "isfree": False, "init": 1}
    # ---------------------------------------- #
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

    angstroms = np.load('{0}/wave.npy'.format(args.path))

    spec = np.load('{0}/spec.npy'.format(args.path)) # units of Jy 
    f_lambda_cgs = (1/33333) * (1/(angstroms**2)) * spec 

    mags = getSED(angstroms, f_lambda_cgs, filterlist=filterlist) # AB magnitudes

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
    #parser.add_argument("--nwalkers")
    #parser.add_argument("--niter")
    #parser.add_argument("--dust")
    args = parser.parse_args()

    args.filters = args.filters.split(',')
    
    run_params = vars(args)
    run_params['nwalkers'] = int(args.nwalkers)
    run_params['niter'] = int(args.niter)
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

