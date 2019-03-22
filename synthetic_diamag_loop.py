'''
Created on Apr 28, 2017

@author: sdenk
'''
from equilibrium_utils_AUG import EQData
from electron_distribution_utils import load_f_from_ASCII, get_E_perp_and_E_par_profile
from plotting_configuration import *
import numpy as np
from scipy.integrate import simps
class diamag_loop:
    def __init__(self, working_dir, shot, time, dstf="Re", EQ_exp="AUGD", EQ_diag="IDE", EQ_ed=0):
        self.EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
        self.dist_obj = load_f_from_ASCII(working_dir)
        A_surf = self.EQ_obj.get_surface_area(time, self.dist_obj.rhop)
#        E_perp = np.zeros(len(A_surf))
#        E_par = np.zeros(len(A_surf))
        # Only LFS distributions
        E_perp, E_par = get_E_perp_and_E_par_profile(self.dist_obj)
        plt.plot(self.dist_obj.rhop, E_perp * 1.e-3, "-", label=r"$E_\perp$")
        plt.plot(self.dist_obj.rhop, E_par * 1.e-3, "--", label=r"$E_\parallel$")
        plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
        plt.gca().set_ylabel(r"$E_\mathrm{kin} [\si{\electronvolt}]$")
        plt.legend()
        E_perp_total = simps(E_perp * A_surf, self.dist_obj.rhop)
        E_par_total = simps(E_par * A_surf, self.dist_obj.rhop)
        print("Total E_perp:", E_perp_total)
        print("Total E_par:", E_par_total)
        print("Ratio: ", E_perp_total / E_par_total)
        plt.show()


if(__name__ == "__main__"):
    diamag_loop("/ptmp1/work/sdenk/nssf/33697/4.80/OERT/ed_39/ecfm_data/fRe/", 33697, 4.80)
