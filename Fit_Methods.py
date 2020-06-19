'''
Created on Nov 6, 2019

@author: sdenk
'''

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
        # Should be list for simultaneous ne Te determination
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
            res = minimize(self.obj_fun, self.profile_parametrization.parameters, jac=False, method="BFGS", \
                           options={'gtol': 1e-04, 'eps': 1.e-06, 'disp': True})
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


        