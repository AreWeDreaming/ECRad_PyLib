


# Import statement
from plotting_configuration import *
import numpy as np
from scipy.optimize import curve_fit
# A message
"""
Deformation of the distribution function tabulated by 
Spitzer and Haerm [L. Spitzer and R. Haerm, Physical Review 89 (1953), 977].

Create object with, e.g., 
    >>> <name> = SpitzerHaerm()
Obtain some help,
    >>> <name>.help()
"""

# Definition of the class
class SpitzerHaerm:
    #
    # --- Documentation string ---
    #
    """
    This provides the functions tabulated by Spitzer and Haerm
    [L. Spitzer and R. Haerm, Physical Review 89 (1953), 977]
    together with a linear interpolation routine.
    """
    #
    # Help method
    def help(self):
        """
        List of mathods.
        """
        text = \
            """
            Available data:"
              1. npt = 51   - number of data points; 
              2. x[:]       - tabulated abscissae;    
              3. G[label,:] - tabulated value; the label runs in the 
                 the range [0,1,2,3,4] for the four values of 
                 Z=1,2,4,16,infinity.     
              4. q[label]   - coefficients defined such that
                       D(x) = (vD/vth) q(Z) G(Z,x),
                 where D id the deformation function, vD the drift velocity
                 which can be written in terms of the current density j,
                       vD = j/ne,
                 n being the electron density and vth the electron thermal
                 speed, cf. Segre, Plasma Physics 20, 288 (1978).

            Available methods:"
              1. interp_G(Z,x) - Returns the linearly interpolated          
                 value of the function ZD(x)/A of Spitzer and Haerm  
                 for the value of Z specified by the argument Z:     
                 Z=1,2,4,16, while Z=-1 means Z=infinity.         
              2. plot_G(Z,fmt='b-',fig=1) - plot the tabulated function
                 corresponding to the given value of Z and using format
                 fmt and figure fig.
              3. deform_D(R,Z,x) - Spitzer deformation function
                       D(x) = (vD/vth) q(Z) G(Z,x),
                 with R=vD/vth.
              4. coef_q(Z) - Return the coefficient q(Z)
            """
        print(text)
    #
    # --- Define data ---
    #
    # Number of data points
    npt = 51
    # Coefficients
    q = np.empty([5])
    q[0], q[1], q[2], q[3], q[4] = 0.7619, 0.6485, 0.5645, 0.4803, 0.4431
    # Data
    x = np.empty([npt])
    G = np.empty([5, npt])
    #
    x[0], G[0, 0], G[1, 0], G[2, 0], G[3, 0], G[4, 0] = 0.10, 0.0008093, 0.0001340, 0.0, 0.0, 0.0001
    x[1], G[0, 1], G[1, 1], G[2, 1], G[3, 1], G[4, 1] = 0.11, 0.0013, 0.0002262, 0.0, 0.0, 0.0001464
    x[2], G[0, 2], G[1, 2], G[2, 2], G[3, 2], G[4, 2] = 0.12, 0.001970, 0.0003630, 0.0, 0.0, 0.0002074
    x[3], G[0, 3], G[1, 3], G[2, 3], G[3, 3], G[4, 3] = 0.13, 0.002847, 0.0005582, 0.0, 0.0, 0.0002856
    x[4], G[0, 4], G[1, 4], G[2, 4], G[3, 4], G[4, 4] = 0.14, 0.003955, 0.0008262, 0.0, 0.0, 0.0003842
    x[5], G[0, 5], G[1, 5], G[2, 5], G[3, 5], G[4, 5] = 0.15, 0.005317, 0.001183, 0.0, 0.0, 0.0005062
    x[6], G[0, 6], G[1, 6], G[2, 6], G[3, 6], G[4, 6] = 0.16, 0.006955, 0.001645, 0.0, 0.0, 0.0006554
    x[7], G[0, 7], G[1, 7], G[2, 7], G[3, 7], G[4, 7] = 0.17, 0.008886, 0.002228, 0.0, 0.0, 0.0008352
    x[8], G[0, 8], G[1, 8], G[2, 8], G[3, 8], G[4, 8] = 0.18, 0.01113, 0.002952, 0.0, 0.0, 0.001050
    x[9], G[0, 9], G[1, 9], G[2, 9], G[3, 9], G[4, 9] = 0.19, 0.01370, 0.003832, 0.0, 0.0, 0.001303
    x[10], G[0, 10], G[1, 10], G[2, 10], G[3, 10], G[4, 10] = 0.20, 0.01660, 0.004884, 0.002163, 0.001645, 0.0016
    x[11], G[0, 11], G[1, 11], G[2, 11], G[3, 11], G[4, 11] = 0.22, 0.02347, 0.007576, 0.003373, 0.002432, 0.002343
    x[12], G[0, 12], G[1, 12], G[2, 12], G[3, 12], G[4, 12] = 0.24, 0.03180, 0.01116, 0.005044, 0.003477, 0.003318
    x[13], G[0, 13], G[1, 13], G[2, 13], G[3, 13], G[4, 13] = 0.26, 0.04165, 0.01575, 0.007280, 0.004833, 0.004570
    x[14], G[0, 14], G[1, 14], G[2, 14], G[3, 14], G[4, 14] = 0.28, 0.05304, 0.02146, 0.01018, 0.006560, 0.006147
    x[15], G[0, 15], G[1, 15], G[2, 15], G[3, 15], G[4, 15] = 0.30, 0.06601, 0.02840, 0.01386, 0.008721, 0.0081
    x[16], G[0, 16], G[1, 16], G[2, 16], G[3, 16], G[4, 16] = 0.32, 0.08057, 0.03666, 0.01842, 0.01139, 0.01049
    x[17], G[0, 17], G[1, 17], G[2, 17], G[3, 17], G[4, 17] = 0.34, 0.09672, 0.04632, 0.02398, 0.01463, 0.01336
    x[18], G[0, 18], G[1, 18], G[2, 18], G[3, 18], G[4, 18] = 0.36, 0.1145, 0.05746, 0.03063, 0.01853, 0.01680
    x[19], G[0, 19], G[1, 19], G[2, 19], G[3, 19], G[4, 19] = 0.38, 0.1338, 0.07012, 0.03849, 0.02317, 0.02085
    x[20], G[0, 20], G[1, 20], G[2, 20], G[3, 20], G[4, 20] = 0.40, 0.1548, 0.08440, 0.04764, 0.02866, 0.02560
    x[21], G[0, 21], G[1, 21], G[2, 21], G[3, 21], G[4, 21] = 0.44, 0.2015, 0.1180, 0.07028, 0.04249, 0.03748
    x[22], G[0, 22], G[1, 22], G[2, 22], G[3, 22], G[4, 22] = 0.48, 0.2545, 0.1586, 0.09924, 0.06082, 0.05308
    x[23], G[0, 23], G[1, 23], G[2, 23], G[3, 23], G[4, 23] = 0.52, 0.3137, 0.2066, 0.1352, 0.08443, 0.07312
    x[24], G[0, 24], G[1, 24], G[2, 24], G[3, 24], G[4, 24] = 0.56, 0.3792, 0.2620, 0.1789, 0.1142, 0.09834
    x[25], G[0, 25], G[1, 25], G[2, 25], G[3, 25], G[4, 25] = 0.60, 0.4508, 0.3254, 0.2309, 0.1511, 0.1296
    x[26], G[0, 26], G[1, 26], G[2, 26], G[3, 26], G[4, 26] = 0.64, 0.5285, 0.3968, 0.2917, 0.1958, 0.1678
    x[27], G[0, 27], G[1, 27], G[2, 27], G[3, 27], G[4, 27] = 0.68, 0.6123, 0.4764, 0.3618, 0.2494, 0.2138
    x[28], G[0, 28], G[1, 28], G[2, 28], G[3, 28], G[4, 28] = 0.72, 0.7023, 0.5646, 0.4416, 0.3127, 0.2687
    x[29], G[0, 29], G[1, 29], G[2, 29], G[3, 29], G[4, 29] = 0.76, 0.7983, 0.6612, 0.5324, 0.3868, 0.3336
    x[30], G[0, 30], G[1, 30], G[2, 30], G[3, 30], G[4, 30] = 0.80, 0.9005, 0.7668, 0.6336, 0.4722, 0.4096
    x[31], G[0, 31], G[1, 31], G[2, 31], G[3, 31], G[4, 31] = 0.88, 1.123, 1.005, 0.8704, 0.6819, 0.5997
    x[32], G[0, 32], G[1, 32], G[2, 32], G[3, 32], G[4, 32] = 0.96, 1.371, 1.281, 1.156, 0.9499, 0.8493
    x[33], G[0, 33], G[1, 33], G[2, 33], G[3, 33], G[4, 33] = 1.04, 1.645, 1.596, 1.494, 1.287, 1.170
    x[34], G[0, 34], G[1, 34], G[2, 34], G[3, 34], G[4, 34] = 1.12, 1.945, 1.952, 1.889, 1.702, 1.574
    x[35], G[0, 35], G[1, 35], G[2, 35], G[3, 35], G[4, 35] = 1.20, 2.273, 2.352, 2.344, 2.204, 2.074
    x[36], G[0, 36], G[1, 36], G[2, 36], G[3, 36], G[4, 36] = 1.28, 2.630, 2.796, 2.864, 2.804, 2.684
    x[37], G[0, 37], G[1, 37], G[2, 37], G[3, 37], G[4, 37] = 1.36, 3.017, 3.290, 3.455, 3.510, 3.421
    x[38], G[0, 38], G[1, 38], G[2, 38], G[3, 38], G[4, 38] = 1.44, 3.435, 3.836, 4.120, 4.331, 4.300
    x[39], G[0, 39], G[1, 39], G[2, 39], G[3, 39], G[4, 39] = 1.52, 3.887, 4.440, 4.868, 5.281, 5.338
    x[40], G[0, 40], G[1, 40], G[2, 40], G[3, 40], G[4, 40] = 1.60, 4.375, 5.096, 5.700, 6.366, 6.554
    x[41], G[0, 41], G[1, 41], G[2, 41], G[3, 41], G[4, 41] = 1.76, 5.465, 6.604, 7.660, 8.996, 9.595
    x[42], G[0, 42], G[1, 42], G[2, 42], G[3, 42], G[4, 42] = 1.92, 6.728, 8.406, 10.06, 12.34, 13.59
    x[43], G[0, 43], G[1, 43], G[2, 43], G[3, 43], G[4, 43] = 2.08, 8.190, 10.54, 12.96, 16.54, 18.72
    x[44], G[0, 44], G[1, 44], G[2, 44], G[3, 44], G[4, 44] = 2.24, 9.880, 13.05, 16.46, 21.74, 25.18
    x[45], G[0, 45], G[1, 45], G[2, 45], G[3, 45], G[4, 45] = 2.40, 11.83, 16.00, 20.64, 28.10, 33.18
    x[46], G[0, 46], G[1, 46], G[2, 46], G[3, 46], G[4, 46] = 2.56, 14.06, 19.40, 25.60, 35.80, 42.95
    x[47], G[0, 47], G[1, 47], G[2, 47], G[3, 47], G[4, 47] = 2.72, 16.62, 23.3, 31.4, 45.0, 54.74
    x[48], G[0, 48], G[1, 48], G[2, 48], G[3, 48], G[4, 48] = 2.88, 19.53, 27.7, 38.2, 56.0, 68.80
    x[49], G[0, 49], G[1, 49], G[2, 49], G[3, 49], G[4, 49] = 3.04, 22.74, 32.7, 46.0, 68.8, 85.41
    x[50], G[0, 50], G[1, 50], G[2, 50], G[3, 50], G[4, 50] = 3.20, 26.00, 38.5, 54.6, 83.9, 104.9
    #
    # --- Auxiliary private methods ---
    #
    def __find_label__(self, Z):
        """
        Given the value of Z, this finds the corresponding label.
        """
        # Define label
        if Z == 1:
            label = 0
        elif Z == 2:
            label = 1
        elif Z == 4:
            label = 2
        elif Z == 16:
            label = 3
        elif Z == -1:
            label = 4
        else:
            print(" ")
            print("ERROR: wrong value of Z! (Z=1,2,4,16, or -1)")
            print(" ")
        return label
    #
    # --- Interpolation function ---
    #
    def interp_G(self, Z, x):
        try:
            # find label corresponding to Z
            label = self.__find_label__(Z)
            # Determine the nearest data point
            if x < self.x[0]:
                x1 = 0; x2 = self.x[0]
                G1 = 0; G2 = self.G[label, 0]
            else:
                for i in xrange(0, self.npt):
                    if x <= self.x[i]:
                        x1 = self.x[i - 1];       x2 = self.x[i]
                        G1 = self.G[label, i - 1]; G2 = self.G[label, i]
                        break
            # Compute the linear interpolation and return
            interp = G1 + (x - x1) * (G2 - G1) / (x2 - x1)
        except(UnboundLocalError):
            return 0.0
        return interp
    #
    # --- Coefficient q(z)
    def coef_q(self, Z):
        # Find label
        label = self.__find_label__(Z)
        # Return
        return self.q[label]
    #
    # --- Spitzer deformation function ---
    def deform_D(self, R, Z, x):
        """
        Return the Spitzer deformation function corresponding to
        the driving
              R = vC/vth,
        where vD is the electron drift speed and vth the electron 
        thermal speed.
        """
        # Find label corresponding to Z
        label = self.__find_label__(Z)
        # compute and return
        deform = R * self.q[label] * self.interp_G(Z, x)
        return deform
    #
    # --- Graphics methods ---
    #

    def plot_G(self, Z, fmt='b-', fig=1):
        """
        Plot the tabulated funtion for the given values of Z
        using the format fmt on the figure fig.
        """
        # Find the label
        label = self.__find_label__(Z)
        figure(fig)
        plt.plot(self.x[:], self.G[label, :], fmt)
        xlabel(r'$x$')
        ylabel(r'$Z D(x)/A$')

    def __init__(self, Z):
        y = self.G[ self.__find_label__(Z), :] * self.q[self.__find_label__(Z)]
        x = self.x
        params = [1.0, 1.0, 1.0, y[0]]
        popt, pcov = curve_fit(cube_model, x, y, params)
        print(popt)
        plt.plot(x, y)
        plt.plot(x, cube_model(x, popt[0], popt[1], popt[2], popt[3]))
        self.params = popt

    def evaluate(self, R, x):
        return R * cube_model(x, self.params[0], self.params[1], self.params[2], self.params[3])

def cube_model(x, a, b, c, d):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x

def static_evaluate(x):
    return 0.50420766 * x ** 3 + 0.14775449 * x ** 2 + 0.59157877 * x - 0.12411147


    #
    # --- End of the class ---
#
# End of file
