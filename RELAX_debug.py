'''
Created on Nov 27, 2017

@author: g2sdenk
'''
import numpy as np
import os
from plotting_configuration import *

def DebugRadiationReaction(folder):
    rad_u_coeff = np.loadtxt(os.path.join(folder, "rad_u_coeff"))
    rad_pitch_coeff = np.loadtxt(os.path.join(folder, "rad_pitch_coeff"))
    e_u_coeff = np.loadtxt(os.path.join(folder, "e_u_coeff"))
    e_pitch_coeff = np.loadtxt(os.path.join(folder, "e_pitch_coeff"))
    col_u_coeff = np.loadtxt(os.path.join(folder, "col_u_coeff"))
    col_pitch_coeff = np.loadtxt(os.path.join(folder, "col_pitch_coeff"))
    u = np.loadtxt(os.path.join(folder, "u_diff"))
    pitch = np.loadtxt(os.path.join(folder, "mu_diff"))
    coeff_list = [rad_u_coeff, rad_pitch_coeff, e_u_coeff, e_pitch_coeff, col_u_coeff, col_pitch_coeff]
    for i in range(len(coeff_list)):
        coeff_list[i] = coeff_list[i].reshape((len(u), len(pitch)))
    i_pitch_perp = np.argmin(np.abs(pitch - np.deg2rad(45)))
    plt.plot(u, coeff_list[1].T[i_pitch_perp], "-")
    plt.plot(u, coeff_list[3].T[i_pitch_perp], "--")
    plt.plot(u, coeff_list[5].T[i_pitch_perp], ":")
    plt.show()

if(__name__ == "__main__"):
    DebugRadiationReaction("/afs/eufus.eu/g2itmdev/user/g2sdenk/ECRad/34663_3.600/ed_26")
