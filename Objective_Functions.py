'''
Created on Nov 9, 2019

@author: sdenk
'''
import numpy as np
import time
from plotting_configuration import plt
class ObjectiveFunction(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.name = "Base Class"
        self.tool_tip = "Do not use"
        self.use = False
        self.reset()
        
    def reset(self):
        self.obj_func_eval_cnt = 0
        self.time = 0
        
    def post_init(self, forward_models, data, profile_parametrization):
        # Sets forward models, data and profile_parametrization
        self.forward_models = forward_models
        self.data = data
        self.profile_parametrization = profile_parametrization
        
    def __call__(self, parameters):
        return self.eval(parameters)
    
    def eval(self, parameters):
        return np.inf
    
class LeastSquares(ObjectiveFunction):
    def __init__(self):
        '''
        Constructor
        '''
        ObjectiveFunction.__init__(self)
        self.name = "Least squares fit"
        self.tool_tip = "returns Sum((y-fwd_mdl(x))**2/ y_err**2)"
        self.use = True
        
        
    def eval(self,parameters):
        obj_f = 0.e0
        if(self.obj_func_eval_cnt == 0):
            self.time = time.time()
        self.profile_parametrization.set_parameters(parameters)
        for data_set, forward_model in zip(self.data, self.forward_models):
            obj_f += np.sum(((data_set.measurements - forward_model.eval(self.profile_parametrization))/data_set.uncertainties)**2)
        self.obj_func_eval_cnt += 1
        if(self.obj_func_eval_cnt > 0 and self.obj_func_eval_cnt % 10 == 0):
            temp_time = time.time()
            print("Time between ten obj_fun calls: {0:4.6f}".format(temp_time - self.time))
            self.time = temp_time
            print("Objective function call number {0:d}.".format(self.obj_func_eval_cnt))
            print("Current value of objective function  {0:2.6e}".format(obj_f))
#         if(self.obj_func_eval_cnt > 0 and self.obj_func_eval_cnt % 200 == 0):
#             x = np.linspace(0.0, 1.0, 100)
#             plt.plot(x, self.profile_parametrization.eval(x))
#             for data_set, forward_model in zip(self.data, self.forward_models):
#                 plt.errorbar(data_set.positions, data_set.measurements, yerr=data_set.uncertainties, marker="*", linestyle=" ")
#                 plt.plot(data_set.positions, forward_model.eval(self.profile_parametrization), "+", linestyle=" ")
#                 plt.show()
        return obj_f
    
    
class MaximumPosterior(ObjectiveFunction):
    def __init__(self):
        '''
        Constructor
        '''
        # Currently includes curvature constraint and constraint for first derivative at axis
        ObjectiveFunction.__init__(self)
        self.name = "Maximum posterior"
        self.tool_tip = "returns loglikelihood + priors. Assumes Cauchy distributed errors with the measurements uncertainties being the median"
        self.use = True
        self.curv_constr_weight = 10.0
        self.deriv_zero_on_axis_weight = 1.0
        self.curv_pnts = 200
        self.cauchy_a0 = 0.5
        self.SOL_constraint = 1
        self.SOL_max_Te = 100.e0 # Everything in eV at the moment 
        
        
        
    def eval(self,parameters):
        obj_f = 0.e0
        if(self.obj_func_eval_cnt == 0):
            self.time = time.time()
        self.profile_parametrization.set_parameters(parameters)
        # Likelihood (logarithmic)
        for data_set, forward_model in zip(self.data, self.forward_models):
            if(np.any(forward_model.mask)):
                obj_f +=  forward_model.norm * (self.cauchy_a0 + 0.5) * np.sum(np.log( 2.0 * self.cauchy_a0 + ((data_set.measurements[forward_model.mask] - forward_model.eval(self.profile_parametrization)[forward_model.mask]) / data_set.uncertainties[forward_model.mask])**2))
        # Prior -> only curvature atm
        if(self.obj_func_eval_cnt > 0 and (self.obj_func_eval_cnt + 1) % 10 == 0):
            print("likelihood_only", obj_f)
        # Curvature constraint
        rho = self.profile_parametrization.get_axis()
        obj_f += np.sum((self.profile_parametrization.eval(rho, dn=2)**2 / \
                        self.profile_parametrization.eval(rho)**2)) / self.curv_constr_weight / self.curv_pnts**2
        Te_SOL = self.profile_parametrization.eval(rho[rho>1.0])
        # Maximum SOL Te prior
        obj_f += self.SOL_constraint*np.sum(Te_SOL[Te_SOL>self.SOL_max_Te])    
        # dval/drho|_(rho=0)=0 constraint
        obj_f += (self.profile_parametrization.eval(0.0, dn=1)/self.profile_parametrization.eval(0.0))**2 / self.deriv_zero_on_axis_weight                    
        self.obj_func_eval_cnt += 1
        if(self.obj_func_eval_cnt > 0 and self.obj_func_eval_cnt % 10 == 0):
            temp_time = time.time()
            print("Time between ten obj_fun calls: {0:4.6f}".format(temp_time - self.time))
            self.time = temp_time
            print("Objective function call number {0:d}.".format(self.obj_func_eval_cnt))
            print("Current value of objective function  {0:2.6e}".format(obj_f))
#         if(self.obj_func_eval_cnt > 0 and self.obj_func_eval_cnt % 200 == 0):
#             x = np.linspace(0.0, 1.0, 100)
#             plt.plot(x, self.profile_parametrization.eval(x))
#             for data_set, forward_model in zip(self.data, self.forward_models):
#                 plt.errorbar(data_set.positions, data_set.measurements, yerr=data_set.uncertainties, marker="*", linestyle=" ")
#                 plt.plot(data_set.positions, forward_model.eval(self.profile_parametrization), "+", linestyle=" ")
#                 plt.show()
        return obj_f
    