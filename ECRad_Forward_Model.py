'''
Created on Nov 25, 2019

@author: g2sdenk
'''
from Forward_Model import ForwardModel
from ECRad_F2PY_Interface import ECRadF2PYInterface
import os
import numpy as np
from Generic_ECE_Diag import ECEDiag

class ECRadForwardModel(ForwardModel):
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
    
    def set_static_parameters(self, time, rho, Te, ne, diag_configuration, \
                              rho_ref = "rhop_prof", eq_slice=None, eq3D=None):
        # Here very specific types of rho Te ne, equilibrium, diag_configuration
        self.Scenario.plasma_dict = {}
        self.Scenario.plasma_dict["time"] = np.atleast_1d(time)
        self.Scenario.plasma_dict["Te"] = np.atleast_2d(Te)
        self.Scenario.plasma_dict["ne"] = np.atleast_2d(ne)
        self.Scenario.plasma_dict["prof_reference"] = rho_ref
        self.Scenario.plasma_dict[self.Scenario.plasma_dict["prof_reference"]] = np.atleast_2d(rho)
        if(eq_slice is not None):
            self.Scenario.plasma_dict["eq_data"] = [eq_slice] # Has be an instance of EQDataSlice
        elif(eq_slice is not None):
            self.Scenario.use3Dscen = eq3D
        else:
            raise ValueError("ECRad forward model needs to be initialized with either a 2D or 3D equilibrium.")
        ext_diag = ECEDiag("EXT")
        ext_diag.set_from_mat(diag_configuration) # Has to point to a .mat file containing an ECRad Scenario configuration
        self.Scenario.avail_diags_dict.update({"EXT":  ext_diag})
        self.Scenario.used_diags_dict.update({"EXT":  ext_diag})
        # Currently only single time point analysis supported, hence itime has to be zero
        # ECrad logs the used geometry in this folder
        # This unnecessary here, so ECRad needs to be changed to not write this file -> TODO
        if(not os.path.isdir(os.path.join(self.Config.working_dir, "ECRad_data"))):
            os.mkdir(os.path.join(self.Config.scratch_dir, "ECRad_data"))
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

\