'''
Created on Nov 6, 2019

@author: sdenk
'''
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline
#from plotting_configuration import plt
class ProfileParametrization:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.reset()
        
    def reset(self):
        # Should set default initial parameters
        self.name = "Base Class"
        self.type = "None"
        self.parameters = None
        self.bounds = None
        self.fixed = None
        self.n_param = 0
        self.axis_bounds = None
        self.points = 0
        # In case the Profile_Parametrization needs some set up
        pass
    
    def is_ready(self):
    # Model depedent test function to see if everything is set up correctly
    # Allows to check if bounds and fixed is set up correctly
        return True
    
    def __call__(self, rho):
        return self.eval(rho)
    
    def eval(self, rho, dn=0):
        # Returns profile value for rho
        # Should support numpy arrays of arbitrary shape and scalars
        # dn sets the order of the derivative to be taken
        return 0.0
    
    
    def make_initital_guess(self, data=None, unc=None, rho=None):
        # Estimate parameters using the data directly
        # Also used for initialization
        return self.parameters
    
    def set_bounds(self, bounds):
        # 2D Numpy array of shape (n_param, 2) contains upper and lower bounds of optimization
        if(bounds.shape[0] != self.n_param):
            raise ValueError("Bounds must have the shape ({0:d}, 2).".format(self.n_param) + " Supplied shape: " + str(bounds.shape))
        self.bounds = bounds
    
    def set_fixed(self, fixed):
        # Requires splitting of the parameter variable
        # 1D boolean numpy array - True means that the parameter will not be optimized
        if(len(fixed) != self.n_param):
            raise ValueError("The number of parameters must not be changed")
        self.fixed = fixed
    
    def set_parameters(self, parameters):
        # 1D Numpy array containing the initial guess
        if(len(parameters) != self.n_param):
            raise ValueError("The number of parameters must not be changed")
        self.parameters = parameters
        
    def set_axis(self, points, axis_bounds):
        # Number of points in target profile
        self.points = points
        # Bounds covered by profile
        self.axis_bounds = axis_bounds
        
    def get_axis(self):
        return np.linspace(self.axis_bounds[0], self.axis_bounds[1], self.points)
        
        
        
    

# class TravisTeProfile(ProfileParametrization):
# # Do not use - easily produces negative Te
#         
#     
#     def reset(self):
#         ProfileParametrization.reset(self)
#         #                             0,   g,  hole, p,   q,  width
#         self.parameters = np.array([0.97, 0.0, -0.5, 3.0, 2.0, 0.5])
#         self.name = "Travis Profile"
#         self.type = "Te"
#         self.n_param = 6
#         # Here True directly after initialization
# 
#     def is_ready(self):
#         return True
#         
#     def make_initital_guess(self, data=None, unc=None, rho=None):
#         if(data is not None):
#             self.parameters[0] = np.max(data)
#         return self.parameters
#     
#     def eval(self, rho):
#         if(np.isscalar(rho) and ( rho > 1.0 or rho < 0.0)):
#             return 0.0
#         prof = self.parameters[0] * (self.parameters[1] - self.parameters[2] + \
#                                      (1.e0 + self.parameters[2] - self.parameters[1]) * \
#                                      ((1.e0 - rho**self.parameters[3])**self.parameters[4]) + \
#                                      self.parameters[2] * (1.e0 - np.exp(- (rho / self.parameters[5]**2))))
#         if(not np.isscalar(rho)):
#             prof[np.logical_or(rho > 1.0, rho < 0.0)] = 0.0
#         return prof

class UnivariateLSQSpline(ProfileParametrization):
    def __init__(self):
        # Uses exp(BiSpline)
        self.name = "Univariate Least-Square Spline"
        self.type = "Te"
        self.parameters = None
        self.c = None
        self.knot_pos = np.array([0.15,0.30, 0.45, 0.55, 0.70, 0.80, 0.86, 0.91, \
                                  0.95, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.05, 1.09, 1.15])
        
    def make_initital_guess(self, data=None, unc=None, rho=None, mask=None):
        if(data is None or rho is None):
            raise ValueError("data and rho are necessary for the UnivariateLSQSpline inteprolation")
        x = np.copy(rho)
        y = np.copy(data)
        if(mask is not None):
            x=x[mask]
            y=y[mask]
        if(unc is not None):
            # Use gaussian error propagation to calcuate error of log(y)
            w = np.copy(unc)
            if(mask is not None):
                w = w[mask]
            w = w / y
            w = 1.0 / w #  turn error into weights
        else:
            w = 1.0
        y = np.log(y)
        # To construct the spline we must have data for rho = 0 and rho=1
        # Use extrapoliation of InterpolatedUnivariateSpline to get these points
        sort = np.argsort(x)
        x = np.concatenate([[self.axis_bounds[0]], x[sort], [self.axis_bounds[1]]])
        y = np.concatenate([[np.max(y)], y[sort], [np.min(y)]])
        w = np.concatenate([[np.mean(w)], w[sort], [np.mean(w)]])
        self.order = 3
        self.t = self.knot_pos[np.logical_and(self.knot_pos > self.axis_bounds[0], \
                                              self.knot_pos < self.axis_bounds[1])]
        self.t = np.r_[(self.axis_bounds[0],)*(self.order+1), self.t, (self.axis_bounds[1],)*(self.order+1)]
        self.spl = make_lsq_spline(x, y, t=self.t, k=self.order, w=w)
        self.c = np.copy(self.spl.c)
        self.parameters = self.spl.c[1:]
        
#         print(self.t)
#         print(self.parameters)
        self.n_param = len(self.parameters)
#         plt.plot(x2, self.spl(x2))
#         plt.plot(x2, np.log(self.eval(x2)), "--")
#         plt.show()
        return self.parameters
    
    def is_ready(self):
        if(self.parameters is not None):
            return True
        else:
            return False
        
    
    def eval(self, rho, dn=0):
        self.c[1:] = self.parameters
        self.c[:2] = self.parameters[0]
        spl = BSpline(self.t, self.c, k=self.order)
        #self.spl.k = self.parameters
        if(dn == 0):
            return(np.exp(spl(rho)))
        elif(dn == 1):
            return spl(rho, nu=1) * np.exp(spl(rho))
        elif(dn == 2):
            return spl(rho, nu=1)**2 * np.exp(spl(rho)) + \
                   spl(rho, nu=2) * np.exp(spl(rho))
        else:
            raise(ValueError("dn in UnivariateLSQSpline is " + str(dn) + " only 0, 1,2 are supported"))



if(__name__== "__main__"):
    from plotting_configuration import plt
    x_test_data = np.linspace(0.05, 1.15, 50)
    test_data = (10*np.exp(-x_test_data**2/0.5**2) * (1.0 - 0.1*(0.5 - np.random.rand(50))))
    unc = np.ones(50)
#     plt.errorbar(x_test_data, test_data,unc, linestyle="none", marker="+")
#     plt.show()
    profile_parametrization = UnivariateLSQSpline()
    profile_parametrization.set_axis(200, [0.0, 1.2])
    profile_parametrization.make_initital_guess(test_data, unc, x_test_data)
    plt.errorbar(x_test_data, test_data,unc)
    plt.plot(profile_parametrization.get_axis(), \
             profile_parametrization.eval(profile_parametrization.get_axis()))
    plt.show()
#     profile_parametrization = TravisTeProfile()
#     from plotting_configuration import plt
#     rho = np.linspace(0.0, 1.0, 100)
#     dummy, Trad, dummy2 = np.loadtxt("/tokp/work/sdenk/ECRad/TRadM_therm.dat", unpack=True)
#     # *1.e3, because  the ECRad ascii output file is in keV but the result file used in the forward model uses eV
#     Trad *= 1.e3
#     art_unc = np.zeros(len(Trad))
#     art_unc[:] = 1.0 # unweighted least squares
#     # Use maximum of data to guess highest Te
#     parameters = profile_parametrization.make_initital_guess(Trad)
#     plt.plot(rho, profile_parametrization.eval(rho, profile_parametrization.parameters), "+")
#     rhot, ne, Te, Zeff = np.loadtxt("/tokp/work/sdenk/ECRad/201810090432150_plasma_profiles.txt", \
#                                     skiprows=3, unpack=True)
#     profile_parametrization.set_parameters(np.array([1.28377419e+03, -8.64675130e-03, -4.53211484e-01,\
#                                    2.99424827e+00, 2.02938658e+00,  5.02846527e-01]))
#     plt.plot(rho, profile_parametrization.eval(rho, profile_parametrization.parameters), "-")
#     plt.plot(rhot, Te * 1.e3, "--")
#     plt.show()
#     
    
