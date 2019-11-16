'''
Created on Nov 6, 2019

@author: sdenk
'''
import numpy as np
import os
class ForwardModel():
    # Very basic at the moment
    def __init__(self, Scenario, Config):
        self.reset(Scenario, Config)
        
    def reset(self, Scenario, Config):
        # Initializes the model using model specific Scenario and Config objects
        # These have to be generalized later so that each ForardModel uses the same type of object
        # The first initialization should be independent of the current discharge
        self.name = "Base Class"
        self.diag_type = "None"
        # Forward models can provide a normalization which is inteded to bring the likelihood function closer to unity
        # Example would be 1.e19 for ne and 1.e3 for Te
        self.norm = 1.0
        pass
    
    def is_ready(self):
        # Allows custom check to see if forward model is ready
        return True
    
    def set_static_parameters(self, rho, Te = None, ne= None, equilibrium=None, diag_configuration=None):
        # Used before an optimization starts
        # Sets parameters that are needed for the optimization, but not optimized themselves
        # Ideally the tyoe of the rho, Te, ne and equilibrium and diagnostic:configuration object is indepedent of which
        # child of FrrwardModel is used
        pass
        
    def __call__(self, Te_model):
        return self.eval(Te_model)

    def eval(self, Te_model):
        # Currently only Te_model but can be extended
        return 0
    
    def config_model(self, *args, **kwargs):
        # Allows the user to change model specific settings
        pass


class W7XECRadForwardModel(ForwardModel):
    # Provides simple python interface to run ECRad
    # For the moment ECRad is run as an application and not a library
    def __init__(self, Scenario, Config):
        # Override the initializer to load forward model specific modules
        ForwardModel.__init__(self, Scenario, Config)
    
    def reset(self, default_Scenario_filename, default_Config_filename):
        from ECRad_Config import ECRadConfig
        from ECRad_Scenario import ECRadScenario, Use3DScenario
        from ECRad_Results import ECRadResults
        self.Scenario = ECRadScenario(noLoad=True)
        self.Config = ECRadConfig(noLoad=True)
        self.Scenario.from_mat(path_in=default_Scenario_filename, load_plasma_dict=False)
        self.Config.from_mat_file(path=default_Config_filename)
        self.Results = ECRadResults()
        self.ready = False
        self.name = "ECRad"
        self.type = "ECE"
        self.norm = 1.e3 # -> eV to keV
        
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
        self.ready = True
        
    def config_model(self, *args, **kwargs):
        for key in kwargs:
            if(hasattr(self.Config, key)):
                setattr(self.Config, key, kwargs[key])
            if(hasattr(self.Scenario, key)):
                setattr(self.Scenario, key, kwargs[key])
    
    def prepare_results(self, model):
        from ECRad_Interface import prepare_input_files, GetECRadExec
        from time import sleep
        import subprocess
        if(not self.ready):
            raise AttributeError("W7XECRadInterface Instance was not properly initialized before first call to eval")
        self.Scenario.plasma_dict["Te"][0] = model.eval(self.Scenario.plasma_dict["rhot_prof"][0])
        call_ECRad =  GetECRadExec(self.Config, self.Scenario, self.Scenario.plasma_dict["time"][0])
        if(not prepare_input_files(self.Config, self.Scenario, 0, False)):
            print("Failed to prepare input data")
            return -1
        
        if(self.Results.Config.extra_output):
            print("-------- Launching ECRad -----------\n")
            print("-------- INVOKE COMMAND------------\n")
            print(call_ECRad)
            print("-------- Current working directory ------------\n")
            print(os.getcwd())
            print("-----------------------------------\n")
        args = call_ECRad.split(" ")[-1]
        invoke = call_ECRad.split(" ")[0]
        ECRad_process = subprocess.Popen([invoke, args])
        if(self.Results.Config.parallel and not self.Results.Config.batch):
            sleep(0.25)
            os.system("renice -n 10 -p " + "{0:d}".format(ECRad_process.pid) + " >/dev/null 2>&1")
        ECRad_process.wait()
    
    def eval(self, model):
        self.prepare_results(model)
        rho, Trad, tau = np.loadtxt(os.path.join(self.Config.scratch_dir,"ECRad_data","TRadM_therm.dat"), unpack=True)
        return Trad * 1.e3
        
    def get_full_results(self, model):
        self.prepare_results(model)
        self.Results.reset()
        self.Results.Scenario = self.Scenario
        self.Results.Config = self.Config
        try:
            # Append times twice to track which time points really do have results in case of crashes
            self.Results.append_new_results(self.Results.Scenario.plasma_dict["time"][0])
        except IOError as e:
            raise IOError("Failed to find ECRad results")
        self.Results.tidy_up(False)
        return self.Results
    
        