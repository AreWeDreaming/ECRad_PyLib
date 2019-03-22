'''
Created on Aug 3, 2017

@author: sdenk
'''
import numpy as np
# import matplotlib.pyplot as plt
from plotting_configuration import *

def stretch_profiles(rhop, rhop_center, width, factor):

    rhop_new = []
    drhop = rhop[1] - rhop[0]
    rhop_new.append(rhop[0])
    for i in range(1, len(rhop)):
        rhop_new.append(rhop_new[i - 1] + drhop * (1.0 - factor * np.exp(-(rhop_new[i - 1] + drhop - rhop_center) ** 2 / width ** 2)))
    rhop_out = np.array(rhop_new)
    return rhop_out

if __name__ == "__main__":
    # make_iso_flux("/ptmp1/work/sdenk/nssf/32740/5.96/OERT/ecfm_data/", 32740, 5.964)
    Tefile = np.loadtxt("/ptmp1/work/sdenk/ECFM4/ecfm_data/Te_file.dat", skiprows=1)
    rhop = Tefile.T[0]
    Te = Tefile.T[1]
    plt.plot(rhop, Te * 1.e-3)
    rhop_out = stretch_profiles(rhop, 0.98, 0.02, 0.5)
    plt.plot(rhop_out, Te * 1.e-3, '--')
    rhop_out = stretch_profiles(rhop, 0.98, 0.02, -0.5)
    plt.plot(rhop_out, Te * 1.e-3, ':')
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.gca().set_ylabel(r"$T_\mathrm{e}$ [keV]")
    plt.show()
