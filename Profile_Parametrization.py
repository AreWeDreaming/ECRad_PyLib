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
        # In case the Profile_Parametrization needs some set up
        pass
    
    def is_ready(self):
    # Model depedent test function to see if everything is set up correctly
    # Allows to check if bounds and fixed is set up correctly
        return True
    
    def __call__(self, rho):
        return self.eval(rho)
    
    def eval(self, rho):
        # Returns profile value for rho
        # Should support numpy arrays of arbitrary shape and scalars
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
        # 1D boolean numpy array - True means that the parameter will not be optimized
        if(len(fixed) != self.n_param):
            raise ValueError("The number of parameters must not be changed")
        self.fixed = fixed
    
    def set_parameters(self, parameters):
        # 1D Numpy array containing the initial guess
        if(len(parameters) != self.n_param):
            raise ValueError("The number of parameters must not be changed")
        self.parameters = parameters
        
        
        
    

class TravisTeProfile(ProfileParametrization):
# Do not use - easily produces negative Te
        
    
    def reset(self):
        ProfileParametrization.reset(self)
        #                             0,   g,  hole, p,   q,  width
        self.parameters = np.array([0.97, 0.0, -0.5, 3.0, 2.0, 0.5])
        self.name = "Travis Profile"
        self.type = "Te"
        self.n_param = 6
        # Here True directly after initialization

    def is_ready(self):
        return True
        
    def make_initital_guess(self, data=None, unc=None, rho=None):
        if(data is not None):
            self.parameters[0] = np.max(data)
        return self.parameters
    
    def eval(self, rho):
        if(np.isscalar(rho) and ( rho > 1.0 or rho < 0.0)):
            return 0.0
        prof = self.parameters[0] * (self.parameters[1] - self.parameters[2] + \
                                     (1.e0 + self.parameters[2] - self.parameters[1]) * \
                                     ((1.e0 - rho**self.parameters[3])**self.parameters[4]) + \
                                     self.parameters[2] * (1.e0 - np.exp(- (rho / self.parameters[5]**2))))
        if(not np.isscalar(rho)):
            prof[np.logical_or(rho > 1.0, rho < 0.0)] = 0.0
        return prof

class UnivariateLSQSpline(ProfileParametrization):
    def __init__(self):
        # Uses exp(BiSpline)
        self.name = "Univariate Least-Square Spline"
        self.type = "Te"
        self.parameters = None
        
    def make_initital_guess(self, data=None, unc=None, rho=None):
        if(data is None or rho is None):
            raise ValueError("data and rho are necessary for the UnivariateLSQSpline inteprolation")
        x = np.copy(rho)
        y = np.copy(data)
        x = x[y > 0.0]
        if(unc is not None):
            # Use gaussian error propagation to calcuate error of log(y)
            w = np.copy(unc)
            w = w[y > 0.0]
            w = w / y[y > 0.0]
            w = 1.0 / w #  turn error into weights
        else:
            w = 1.0
        y = y[y > 0.0]
        y = np.log(y)
        y = y[np.logical_and(x>0.0, x<1.0)]
        w = w[np.logical_and(x>0.0, x<1.0)]
        x = x[np.logical_and(x>0.0, x<1.0)]
        # To construct the spline we must have data for rho = 0 and rho=1
        # Use extrapoliation of InterpolatedUnivariateSpline to get these points
        sort = np.argsort(x)
#         plt.plot(x,y, "+")
        self.order = 3
        self.t = np.linspace(0.1, 0.9, 7)
        self.t=np.r_[(0.0,)*(self.order+1),self.t, (1.0,)*(self.order+1)]
        self.spl = make_lsq_spline(x[sort], y[sort], t=self.t, k=self.order, w=w[sort])
        self.parameters = self.spl.c
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
        
    
    def eval(self, rho):
        spl =BSpline(self.t, self.parameters, k=self.order)
        #self.spl.k = self.parameters
        return(np.exp(spl(rho)))

if(__name__== "__main__"):
    profile_parametrization = TravisTeProfile()
    from plotting_configuration import plt
    rho = np.linspace(0.0, 1.0, 100)
    dummy, Trad, dummy2 = np.loadtxt("/tokp/work/sdenk/ECRad/TRadM_therm.dat", unpack=True)
    # *1.e3, because  the ECRad ascii output file is in keV but the result file used in the forward model uses eV
    Trad *= 1.e3
    art_unc = np.zeros(len(Trad))
    art_unc[:] = 1.0 # unweighted least squares
    # Use maximum of data to guess highest Te
    parameters = profile_parametrization.make_initital_guess(Trad)
    plt.plot(rho, profile_parametrization.eval(rho, profile_parametrization.parameters), "+")
    rhot, ne, Te, Zeff = np.loadtxt("/tokp/work/sdenk/ECRad/201810090432150_plasma_profiles.txt", \
                                    skiprows=3, unpack=True)
    profile_parametrization.set_parameters(np.array([1.28377419e+03, -8.64675130e-03, -4.53211484e-01,\
                                   2.99424827e+00, 2.02938658e+00,  5.02846527e-01]))
    plt.plot(rho, profile_parametrization.eval(rho, profile_parametrization.parameters), "-")
    plt.plot(rhot, Te * 1.e3, "--")
    plt.show()
    
    
