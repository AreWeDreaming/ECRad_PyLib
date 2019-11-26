'''
Created on Nov 25, 2019

@author: g2sdenk
'''
from Forward_Model import ForwardModel
from ECRad_F2PY_Interface import ECRadF2PYInterface
import os
import numpy as np

class W7XECRadForwardModel(ForwardModel):
    # Provides simple python interface to run ECRad
    # For the moment ECRad is run as an application and not a library
    def __init__(self, Scenario, Config):
        # Override the initializer to load forward model specific modules
        ForwardModel.__init__(self, Scenario, Config)
    
    def reset(self, default_Scenario_filename, default_Config_filename):
        ForwardModel.reset(self, None, None)
        from ECRad_Config import ECRadConfig
        from ECRad_Scenario import ECRadScenario, Use3DScenario
        from ECRad_Results import ECRadResults
        self.Scenario = ECRadScenario(noLoad=True)
        self.Config = ECRadConfig(noLoad=True)
        self.Scenario.from_mat(path_in=default_Scenario_filename, load_plasma_dict=False)
        self.Config.from_mat_file(path=default_Config_filename)
        self.ecrad_f2py_interface = ECRadF2PYInterface()
        self.Results = ECRadResults()
        self.ready = False
        self.name = "ECRad"
        self.type = "ECE"
        self.norm = 1.e3 # -> eV to keV
        self.rho = None
        self.Trad = None
        self.tau = None
        
    def setup_data_mask(self, data, user_mask):
        # Takes instance of the data_set argument and sets up the mask
        self.mask = np.zeros(data.measurements.shape)
        
    def is_ready(self):
        return self.ready
    
    def set_static_parameters(self, time, rho, Te, ne, equilibrium, diag_configuration):
        from Diags import EXT_diag
        # Here very specific types of rho Te ne, equilibrium, diag_configuration
        self.Scenario.plasma_dict = {}
        self.Scenario.plasma_dict["time"] = np.atleast_1d(time)
        self.Scenario.plasma_dict["rhot_prof"] = np.atleast_2d(rho) #
        self.Scenario.plasma_dict["Te"] = np.atleast_2d(Te)
        self.Scenario.plasma_dict["ne"] = np.atleast_2d(ne)
        self.Scenario.plasma_dict["prof_reference"] = "rhot_prof"
        self.Scenario.use3Dscen = equilibrium # Has be an instance of Use3DScenario
        ext_diag = EXT_diag("EXT")
        ext_diag.set_from_mat(diag_configuration) # Has to point to a .mat file containing an ECRad Scenario configuration
        self.Scenario.avail_diags_dict.update({"EXT":  ext_diag})
        self.Scenario.used_diags_dict.update({"EXT":  ext_diag})
        # Currently only single time point analysis supported, hence itime has to be zero
        # ECrad logs the used geometry in this folder
        # This unnecessary here, so ECRad needs to be changed to not write this file -> TODO
        if(not os.path.isdir(os.path.join(self.Config.working_dir, "ecfm_data"))):
            os.mkdir(os.path.join(self.Config.scratch_dir, "ecfm_data"))
        self.ecrad_f2py_interface.set_config_and_diag(self.Config, self.Scenario, 0)
        self.rho = self.ecrad_f2py_interface.set_equilibrium(self.Scenario, 0)
        self.ready = True
        
    def pre_optimization_tuneup(self, model):
        # model is an instance of profile_parametrization
        # Here the Forward is fine tuned
        # Rays are computed and the radiation transport grid it is checked that the step size in the radiation transport is sufficiently small
        # This routine could also be used to determine which channels should be forward modeled via the fm_flag
        rhop_out = self.ecrad_f2py_interface.make_rays(self.Scenario, 0)
        fm_flag = np.zeros(self.ecrad_f2py_interface.N_ch, dtype=np.bool)
        fm_flag[:] = True
        self.ecrad_f2py_interface.set_fm_flag(fm_flag)
        self.ecrad_f2py_interface.set_grid_update(True)
        self.Scenario.plasma_dict["Te"][0] = model.eval(self.Scenario.plasma_dict["rhot_prof"][0])
        self.rho = self.ecrad_f2py_interface.make_rays(self.Scenario, 0)
        Trad_fm, tau_fm = self.ecrad_f2py_interface.eval_Trad(self.Scenario, self.Config, 0)
        fm_flag[:] = False
        self.ecrad_f2py_interface.set_fm_flag(fm_flag)
        self.Trad, self.tau = self.ecrad_f2py_interface.eval_Trad(self.Scenario, self.Config, 0)
        # Set channels to be evaluated with radiation transport
        # if classical and forward model have more 20 eV difference
        # and more than 3% relative deviation
        fm_flag[np.logical_and(np.abs(Trad_fm - self.Trad) > 20.0, \
                               np.abs((Trad_fm - self.Trad)) > 0.015* (Trad_fm + self.Trad))] = True
        # Also forward model channels with low optical depth
        fm_flag[tau_fm < 4] = True
        # Do not forward channels with super low optical depth
        fm_flag[tau_fm < 0.5] = False     
        self.ecrad_f2py_interface.set_fm_flag(fm_flag)
        self.ecrad_f2py_interface.set_grid_update(False)
        
        
    def config_model(self, *args, **kwargs):
        for key in kwargs:
            if(hasattr(self.Config, key)):
                setattr(self.Config, key, kwargs[key])
            if(hasattr(self.Scenario, key)):
                setattr(self.Scenario, key, kwargs[key])
    
    def eval(self, model):
        if(not self.ready):
            raise AttributeError("W7XECRadInterface Instance was not properly initialized before first call to eval")
        # Again zero hardcoded here atm
        self.Scenario.plasma_dict["Te"][0] = model.eval(self.Scenario.plasma_dict["rhot_prof"][0])
        self.Trad, self.tau = self.ecrad_f2py_interface.eval_Trad(self.Scenario, self.Config, 0)
        return self.Trad
        
    def get_full_results(self, model):
        #Currently not supported
        pass
#         self.prepare_results(model)
#         self.Results.reset()
#         self.Results.Scenario = self.Scenario
#         self.Results.Config = self.Config
#         try:
#             # Append times twice to track which time points really do have results in case of crashes
#             self.Results.append_new_results(self.Results.Scenario.plasma_dict["time"][0])
#         except IOError as e:
#             raise IOError("Failed to find ECRad results")
#         self.Results.tidy_up(False)
#         return self.Results
#     

def test_ECRad_fm(working_dir, data_file, ne_prof_data, equilibrium_file, equilibrium_type, \
                  wall_file, ecrad_config_file, ecrad_scenario_file, B_scale=1.0):
    from Data_Set import DataSet
    from Profile_Parametrization import UnivariateLSQSpline
    from ECRad_Scenario import Use3DScenario
    from plotting_configuration import plt
    from ECRad_Results import ECRadResults
    from scipy.interpolate import InterpolatedUnivariateSpline
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
    forward_model = W7XECRadForwardModel(ecrad_scenario_file, ecrad_config_file)
    # Change some settings used keywords
    forward_model.config_model(working_dir=working_dir, \
                               scratch_dir = working_dir, \
                               extra_output = False, \
                               parallel = True, \
                               parallel_cores = 1, \
                               debug = False)
    # ECRad cannot directly use the profile_parametrization
    # Therefore, a profile is used as an intermediary
    rho = np.linspace(0.0, 1.0, 200)
    # Create spline here since ECRad wants Te and ne to have the same rho axis
    ne_spl = InterpolatedUnivariateSpline(rhot, np.log(ne))
    Te_spl = InterpolatedUnivariateSpline(rhot, np.log(Te))
    forward_model.set_static_parameters(time, rho, np.exp(Te_spl(rho)), np.exp(ne_spl(rho)), Use3DScen, ecrad_scenario_file)
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
    forward_model.pre_optimization_tuneup(profile_parametrization)
    Trad = forward_model.eval(profile_parametrization)
    plt.plot(forward_model.rho, Trad * 1.e-3, "+")
    plt.errorbar(forward_model.rho, data[0].measurements * 1.e-3, yerr=data[0].uncertainties / 1.e3, marker= "o", linestyle="")
    plt.show()
    

if(__name__ =="__main__"):
    test_ECRad_fm("/gss_efgw_work/work/g2sdenk/ECRad_runs/", \
                  "ECE_res.txt", "plasma_profiles.txt", \
                  "VMEC.txt", "VMEC", "W7X_wall_SI.dat" , \
                  "ECRad_20180823016002_EXT_ed20.mat", \
                  "ECRad_20180823016002_EXT_ed20.mat", \
                  B_scale = 0.9209027777777778)
    