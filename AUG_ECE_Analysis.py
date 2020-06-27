'''
Created on Jun 18, 2020

@author: denk
'''

from plotting_configuration import *
from ECRad_Forward_Model import ECRadForwardModel
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
from shotfile_handling_AUG import get_data_calib, load_IDA_data
from equilibrium_utils_AUG import EQData
from Generic_ECE_Diag import BasicDiag
from zmq.ssh import forward

def analyze_ECE_measurements_AUG(working_dir, shot, time, t_smooth, ECE_diag, eq_diag, ida_diag, \
                                 wall_file, ecrad_config_file, ecrad_scenario_file, B_scale=1.0):
    ECE_diag.t_smooth = t_smooth
    ECE_unc, ECE_data = get_data_calib(ECE_diag, shot, time, eq_exp=eq_diag.exp, eq_diag=eq_diag.diag,eq_ed=eq_diag.ed)
    plasma_dict = load_IDA_data(shot, [time], exp=ida_diag.exp, ed = ida_diag.ed)
    EQ_obj = EQData(shot, EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
    eq_slice_obj = EQ_obj.GetSlice(time)
    ecrad_scenario_file =  os.path.join(working_dir,ecrad_scenario_file)
    ecrad_config_file = os.path.join(working_dir,ecrad_config_file)
    # *1.e3, because  the ECRad ascii output file is in keV but the result file used in the forward model uses eV
    # Te only loaded for convenience here, but it is not used
    forward_models = [ECRadForwardModel(ecrad_scenario_file, ecrad_config_file)]
    # Change some settings used keywords
    forward_models[0].config_model(working_dir=working_dir, \
                                   scratch_dir = working_dir, \
                                   extra_output = False, \
                                   parallel = True, \
                                   parallel_cores = 8, \
                                   debug = False)
    # ECRad cannot directly use the profile_parametrization
    # Therefore, a profile is used as an intermediary
    IDA_ne_spl = InterpolatedUnivariateSpline(plasma_dict["rhop_prof"][0], np.log(plasma_dict["ne"][0]))
    data = [DataSet("ECE", "radiometer", [time - t_smooth*0.5, time + t_smooth*0.5], \
                    measurements=ECE_data[1] * 1.e3, uncertainties=np.sqrt(ECE_unc[0]**2 + ECE_unc[1]**2) * 1.e3, \
                    positions=ECE_data[0])]
    forward_models[0].setup_data_mask(data[0], None)
    profile_parametrization = UnivariateLSQSpline()
    profile_parametrization.set_axis(200, [0.0, 1.2])
    rho = profile_parametrization.get_axis()
    # Use maximum of data to guess highest Te
    initial_guess_dataset = forward_models[0].initial_guess_data(data[0])
    profile_parametrization.make_initital_guess(data=initial_guess_dataset.measurements, \
                                                unc=initial_guess_dataset.uncertainties, \
                                                rho=initial_guess_dataset.positions, \
                                                mask=forward_models[0].mask)
    Te = profile_parametrization.eval(rho)
    forward_models[0].set_static_parameters(time, rho, Te, np.exp(IDA_ne_spl(rho)), ecrad_scenario_file, \
                                            rho_ref = "rhop_prof", eq_slice=eq_slice_obj)
    # Make rays
    forward_models[0].pre_optimization_tuneup(profile_parametrization)
    scipy_minimize = ScipyMinimize()
    plt.plot(rho, Te / 1.e3, "-", label=r"Initial $T_\mathrm{e}$")
    plt.errorbar(forward_models[0].rho, data[0].measurements / 1.e3, data[0].uncertainties/1.e3 ,fmt="o", label="ECE data")
    plt.errorbar(forward_models[0].rho, initial_guess_dataset.measurements / 1.e3, initial_guess_dataset.uncertainties/1.e3 ,\
                 fmt="o", label="ECE data for initial guess")
    plt.legend()
    plt.show()
    plt.plot(rho, Te / 1.e3, "-", label=r"Initial $T_\mathrm{e}$")
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
        plt.plot(plasma_dict["rhop_prof"][0], plasma_dict["Te"][0] / 1.e3, "-", label=r"IDA $T_\mathrm{e}$")
        plt.plot(rho, Te / 1.e3, "--", label=r"Final $T_\mathrm{e}$")
#         ecrad_results = forward_models[0].get_full_results(profile_parametrization)
        plt.errorbar(forward_models[0].rho, data[0].measurements / 1.e3, data[0].uncertainties/1.e3 ,fmt="o", label="ECE data")
        plt.plot(forward_models[0].rho, forward_models[0].eval(profile_parametrization) / 1.e3, "+", label=r"ECRad $T_\mathrm{rad}$")
        plt.gca().set_xlabel(r"$\rho_\mathrm{tor}$")
        plt.gca().set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\electronvolt}]$")
        plt.gca().set_xlim(0,1.2)
        plt.gca().set_ylim(0,8)
    else:
        print("Not ready")
        return
    plt.legend()
    plt.show()
    
if(__name__=="__main__"):
    ECE_diag = BasicDiag("ECE", "AUGD", "RMD", 0)
    eq_diag = BasicDiag("EQ", "AUGD", "EQH", 0)
    ida_diag = BasicDiag("IDA", "AUGD", "IDA", 0)
    shot = 31539
    time = 2.814
    working_dir = "/gss_efgw_work/work/g2sdenk/ECRad_runs"
    t_smooth = 1.e-3
    wall_file = "/afs/ipp/u/sdenk/ECRad_testing/augd_ecrad_pylib/vessel.dat"
    ecrad_config_file = "/afs/ipp/u/sdenk/ECRad_testing/ECRad_31539_ECE_ed4.mat"
    ecrad_scenario_file = ecrad_config_file
    analyze_ECE_measurements_AUG(working_dir, shot, time, t_smooth, ECE_diag, eq_diag, ida_diag, \
                                 wall_file, ecrad_config_file, ecrad_scenario_file, B_scale=1.0)
