'''
Created on Nov 6, 2019

@author: sdenk
'''
import numpy as np
import time
from ECRad_Results import ECRadResults
from dask.diagnostics.tests.test_profiler import prof

class OptimizerNotReadyException(Exception):
    pass

class FitMethod:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.reset()
        
    def reset(self):
        self.name = "Base Class" # Identifier
        self.use = False #
        self.tooltip = "Do not use - only provides base class for different profile reconstruction methods"
        self.forward_models = None
        self.data = None
        self.profile_parametrization = None
        self.obj_fun = None
        self.ready = False
        
    def check_if_ready(self):
        ready = True
        if(self.forward_models is None):
            print("Forward models have not been set")
        else:
            for forward_model in self.forward_models:
                if(not forward_model.is_ready()):
                    print(forward_model.type + " forward model with name " + forward_model.name + " is not ready.")
                    # One could alternatively return here, but if we keep the loop going the user gets a list of everything that is not ready
                    ready = False
        if(self.data is None):
            print("Data has not been set")
        else:
            for data_set in self.data:
                if(not data_set.is_ready()):
                    print(data_set.type + " data set with name " + data_set.name + " is not ready.")
                    # One could alternatively return here, but if we keep the loop going the user gets a list of everything that is not ready
                    ready = False
        if(self.profile_parametrization is None):
            print("Profile parametrization has not been set")
        else:
            if(not self.profile_parametrization.is_ready()):
                print(self.profile_parametrization.type + " profile parameterization with name " + self.profile_parametrization.name + " is not ready.")
                # One could alternatively return here, but if we keep the loop going the user gets a list of everything that is not ready
                ready = False
        return ready
        
    def set_forward_models(self, forward_models):
        # List of forward model Objects
        # Each entry belongs to an one diagnostic
        self.forward_models = forward_models
        
    def set_profile_parametrization(self, profile_parametrization):
        self.profile_parametrization = profile_parametrization
        
    def set_data(self, data):
        # list of data_set objects
        # Each entry respresents one diagnostic
        # The ForwardModel instances has to be set up such that it uses the same indixing
        self.data = data
        
    def optimize(self):
        # Runs optimization
        if(not self.check_if_ready()):
            raise OptimizerNotReadyException("Cannot optimize yet - some components are not ready")
    
    def set_obj_fun(self, obj_fun):
        # Sets the instance objective function
        self.obj_fun = obj_fun
    
    def set_all(self, data, profile_parametrization, forward_models, obj_fun):
        # Convenience function to up everything at once
        self.data = data
        self.profile_parametrization = profile_parametrization
        self.forward_models = forward_models
        self.obj_fun = obj_fun


class ScipyMinimize(FitMethod):
    def reset(self):
        FitMethod.reset(self)
        self.name = "Scipy Opimizer" # Identifier
        self.use = True #
        self.tooltip = "Provides wrapper to a variety of optimizers"
    
    def optimize(self, method = "BFGS"):
        # First call parent optimize - all it does is check if everything is ready and
        # raise an Exception if not
        FitMethod.optimize(self)
        from scipy.optimize import minimize
        if(method == "BFGS"):
            res = minimize(self.obj_fun.__call__, self.profile_parametrization.parameters, jac=False, method="BFGS", \
                           options={'gtol': 1e-04, 'eps': 1.e-03, 'disp': True})
        else:
            raise ValueError("Only BFGS method supported at the moment!")
        print(res.message)
        print("Used a total of {0:d} objective function calls.".format(self.obj_fun.obj_func_eval_cnt))
        if(res.success):
            print("Success")
            return res.x, True
        else:
            print("optimization failed, returning final result anyways")
            return res.x, False
    
# To be removevd later, necessary now because import * only allowed at module level and this module sets up matplotlib nicely
from plotting_configuration import *
def test_scipy_minimize_optimizer(data_file, ne_prof_data, equilibrium_file, equilibrium_type, \
                                 wall_file, ecrad_config_file, ecrad_scenario_file, B_scale=1.0):
    from Forward_Model import W7XECRadForwardModel
    from Data_Set import DataSet
    from Profile_Parametrization import UnivariateLSQSpline
    from ECRad_Scenario import Use3DScenario
    from scipy.interpolate import InterpolatedUnivariateSpline
    from Objective_Functions import LeastSquares
    import os 
    import numpy as np
    time  = 2.15
    scipy_minimize = ScipyMinimize()
    ECE_data = np.loadtxt(data_file, skiprows=12)
    Result = ECRadResults()
    Result.from_mat_file(ecrad_scenario_file)
    # *1.e3, because  the ECRad ascii output file is in keV but the result file used in the forward model uses eV
    data = [DataSet("ECE", "radiometer", [2.149, 2.151], \
                    measurements=ECE_data.T[3] * 1.e3, uncertainties=ECE_data.T[-3] * 1.e3, \
                    positions=Result.resonance["rhop_cold"][0])]
    profile_parametrization = UnivariateLSQSpline()
    # Use maximum of data to guess highest Te
    profile_parametrization.make_initital_guess(data=data[0].measurements, unc=data[0].uncertainties, \
                                                rho=data[0].positions)
    forward_models = [W7XECRadForwardModel(ecrad_scenario_file, ecrad_config_file)]
    # Te only loaded for convenience here, but it is not used
    rhot, Te, ne = np.loadtxt(ne_prof_data, skiprows=3, unpack=True)
    ne *= 1.e20
    Use3DScen = Use3DScenario()
    Use3DScen.used = True
    Use3DScen.equilibrium_file = equilibrium_file
    Use3DScen.equilibrium_type = equilibrium_type
    Use3DScen.vessel_filename = wall_file
    Use3DScen.B_ref = B_scale
    # ECRad cannot directly use the profile_parametrization
    # Therefore, a profile is used as an intermediary
    rho = np.linspace(0.0, 1.0, 200)
    # Create spline here since ECRad wants Te and ne to have the same rho axis
    ne_spl = InterpolatedUnivariateSpline(rhot, np.log(ne))
    Te = profile_parametrization.eval(rho)
    forward_models[0].set_static_parameters(time, rho, Te, np.exp(ne_spl(rho)), Use3DScen, ecrad_scenario_file)
    # Fine tune the model
    forward_models[0].config_model(scratch_dir = "/tokp/scratch/sdenk/ECRad/", \
                                   extra_output = False, \
                                   parallel = True, \
                                   parallel_cores = 32, \
                                   debug = False)
    plt.plot(rho, Te, "-", label=r"Initial $T_\mathrm{e}$")
    scipy_minimize.set_data(data)
    scipy_minimize.set_forward_models(forward_models)
    scipy_minimize.set_profile_parametrization(profile_parametrization)
    obj_fun = LeastSquares()
    obj_fun.post_init(forward_models, data, profile_parametrization)
    scipy_minimize.set_obj_fun(obj_fun)
    ecrad_results = forward_models[0].get_full_results(profile_parametrization)
    if(scipy_minimize.check_if_ready()):
        final_params, success = scipy_minimize.optimize()
        #final_params = profile_parametrization.parameters
        profile_parametrization.set_parameters(final_params)
        Te = profile_parametrization.eval(rho)
        plt.plot(rho, Te, "--", label=r"Final $T_\mathrm{e}$")
        ecrad_results = forward_models[0].get_full_results(profile_parametrization)
        plt.plot(ecrad_results.resonance["rhop_cold"][0], data[0].measurements, "o", label="Artifical ECE data")
        plt.plot(ecrad_results.resonance["rhop_cold"][0], forward_models[0].eval(profile_parametrization), "+", label=r"ECRad $T_\mathrm{rad}$")
        plt.gca().set_xlabel(r"$\rho_\mathrm{tor}$")
        plt.gca().set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\electronvolt}]$")
        plt.gca().set_xlim(0,1)
    else:
        print("Not ready")
        return
    plt.legend()
    plt.show()
    
if(__name__=="__main__"):
    test_scipy_minimize_optimizer("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/20180823016002/ECE_res.txt", \
                                 "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/20180823016002/profiles_res.txt", \
                                 "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/20180823016002/wout.txt", \
                                 "VMEC", "/tokp/work/sdenk/ECRad/W7X_wall_SI.dat" , \
                                 "/tokp/work/sdenk/ECRad/ECRad_20180823016002_EXT_ed4.mat", \
                                 "/tokp/work/sdenk/ECRad/ECRad_20180823016002_EXT_ed4.mat", \
                                 B_scale = 0.9209027777777778)

        