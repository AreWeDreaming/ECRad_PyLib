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
