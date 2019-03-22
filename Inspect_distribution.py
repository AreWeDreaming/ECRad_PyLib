'''
Created on Sep 14, 2016

@author: sdenk
'''
from electron_distribution_utils import read_LUKE_data, load_f_from_ASCII
from plotting_configuration import *
import numpy as np


def plot_all_slices(path):
    rhop, x, y, Fe = read_LUKE_data(path, no_preprocessing=True)
    # rhop, x, y, Fe = load_f_from_ASCII(path + "fRe/")
    for irhop in range(len(rhop)):
        for ix in range(5):  # len(x)
            plt.plot(y, Fe[irhop, ix], "+", label=r"$f_\mathrm{LUKE}(\rho_\mathrm{pol} = " + r"{0:1.4f}".format(rhop[irhop]) + \
                     r", \,u_\perp = " + r"{0:1.2f}".format(x[ix]) + r",\,\mu)$")
            if((ix + 1) % 5 == 0):
                plt.gca().legend()
                plt.gca().set_xlabel(r"$\mu\, [rad]$")
                plt.gca().set_ylabel(r"$f_\mathrm{LUKE}(\rho_\mathrm{pol}, \,u_\perp ,\,\mu)$")
                plt.show()

plot_all_slices("/ptmp1/work/sdenk/nssf/33698/5.00/OERT/ecfm_data/")
