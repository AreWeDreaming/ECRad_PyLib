'''
Created on Jun 18, 2020

@author: denk
'''

from plotting_configuration import *
from W7X_ECRad_Forward_Model import W7XECRadForwardModel
from Data_Set import DataSet
from Profile_Parametrization import UnivariateLSQSpline
from ECRad_Scenario import Use3DScenario
from scipy.interpolate import InterpolatedUnivariateSpline
from Objective_Functions import MaximumPosterior
import numpy as np
import os
import time
from ECRad_Results import ECRadResults
from Fit_Methods import ScipyMinimize

def test_scipy_minimize_optimizer(working_dir, data_file, ne_prof_data, equilibrium_file, equilibrium_type, \
                                 wall_file, ecrad_config_file, ecrad_scenario_file, B_scale=1.0):
    time  = 4.45
    ECE_data = np.loadtxt(os.path.join(working_dir,data_file), skiprows=12)
    Result = ECRadResults()
    ecrad_scenario_file =  os.path.join(working_dir,ecrad_scenario_file)
    ecrad_config_file = os.path.join(working_dir,ecrad_config_file)
    Result.from_mat_file(ecrad_scenario_file)
    # *1.e3, because  the ECRad ascii output file is in keV but the result file used in the forward model uses eV
    # Te only loaded for convenience here, but it is not used
    rhot, ne, Te, Zeff = np.loadtxt(os.path.join(working_dir,ne_prof_data), skiprows=3, unpack=True)
    Te *= 1.e3
    Use3DScen = Use3DScenario()
    Use3DScen.used = True
    Use3DScen.equilibrium_file = os.path.join(working_dir,equilibrium_file)
    Use3DScen.equilibrium_type = equilibrium_type
    Use3DScen.vessel_filename = os.path.join(working_dir,wall_file)
    Use3DScen.B_ref = B_scale
    forward_models = [W7XECRadForwardModel(ecrad_scenario_file, ecrad_config_file)]
    # Change some settings used keywords
    forward_models[0].config_model(working_dir=working_dir, \
                                   scratch_dir = working_dir, \
                                   extra_output = False, \
                                   parallel = True, \
                                   parallel_cores = 8, \
                                   debug = False)
    # ECRad cannot directly use the profile_parametrization
    # Therefore, a profile is used as an intermediary
    rho = np.linspace(0.0, 1.0, 200)
    # Create spline here since ECRad wants Te and ne to have the same rho axis
    ne_spl = InterpolatedUnivariateSpline(rhot, np.log(ne))
    Te_spl = InterpolatedUnivariateSpline(rhot, np.log(Te))
    forward_models[0].set_static_parameters(time, rho, np.exp(Te_spl(rho)), np.exp(ne_spl(rho)), Use3DScen, ecrad_scenario_file)
    # the set static parpameters methods computes resonances for the ECE assuming straight lines of sight
    # This could in principle be used in the next step for the inital guess via the spline fit
    # Currently there is a bug in ECRad which causes the resonances in straight ray mode to be all zero
    # Hence the workaround below
    # TODO -> Fix
    data = [DataSet("ECE", "radiometer", [2.149, 2.151], \
                    measurements=ECE_data.T[3] * 1.e3, uncertainties=ECE_data.T[-3] * 1.e3, \
                    positions=Result.resonance["rhop_cold"][0])]
    profile_parametrization = UnivariateLSQSpline()
    # Use maximum of data to guess highest Te
    profile_parametrization.make_initital_guess(data=data[0].measurements, unc=data[0].uncertainties, \
                                                rho=data[0].positions)
    Te = profile_parametrization.eval(rho)
    # Make rays
    forward_models[0].pre_optimization_tuneup(profile_parametrization)
    scipy_minimize = ScipyMinimize()
    plt.plot(rho, Te, "-", label=r"Initial $T_\mathrm{e}$")
    scipy_minimize.set_data(data)
    scipy_minimize.set_forward_models(forward_models)
    scipy_minimize.set_profile_parametrization(profile_parametrization)
    obj_fun = MaximumPosterior()
    obj_fun.post_init(forward_models, data, profile_parametrization)
    scipy_minimize.set_obj_fun(obj_fun)
    if(scipy_minimize.check_if_ready()):
        final_params, success = scipy_minimize.optimize()
        #final_params = profile_parametrization.parameters
        profile_parametrization.set_parameters(final_params)
        Te = profile_parametrization.eval(rho)
        plt.plot(rho, Te, "--", label=r"Final $T_\mathrm{e}$")
        ecrad_results = forward_models[0].get_full_results(profile_parametrization)
        plt.errorbar(forward_models[0].rho, data[0].measurements, data[0].uncertainties ,fmt="o", label="Artifical ECE data")
        plt.plot(forward_models[0].rho, forward_models[0].eval(profile_parametrization), "+", label=r"ECRad $T_\mathrm{rad}$")
        plt.gca().set_xlabel(r"$\rho_\mathrm{tor}$")
        plt.gca().set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\electronvolt}]$")
        plt.gca().set_xlim(0,1)
    else:
        print("Not ready")
        return
    plt.legend()
    plt.show()
    
if(__name__=="__main__"):
    test_scipy_minimize_optimizer("/gss_efgw_work/work/g2sdenk/ECRad_runs/", \
                                  "ECE_res.txt", "plasma_profiles.txt", \
                                  "VMEC.txt", "VMEC", "W7X_wall_SI.dat" , \
                                  "ECRad_20180823016002_EXT_ed20.mat", \
                                  "ECRad_20180823016002_EXT_ed20.mat", \
                                  B_scale = 0.9209027777777778)