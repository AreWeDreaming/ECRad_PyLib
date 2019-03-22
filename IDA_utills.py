'''
Created on Jun 13, 2017

@author: sdenk
'''

import numpy as np
from plotting_configuration import *
from shotfile_handling_AUG import load_IDA_ECE_residues, load_IDA_data

def compare_IDA_residues(shot, time, IDA_exp_list, IDA_ed_list):
    IDA_rhop_list = []
    IDA_ECE_res_list = []
    IDA_ECE_dat_list = []
    IDA_ECE_unc_list = []
    IDA_ECE_mod_list = []
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    for i in range(len(IDA_exp_list)):
        IDA_rhop, IDA_ECE_res, IDA_ECE_dat, IDA_ECE_unc, IDA_ECE_mod = load_IDA_ECE_residues(shot, time, IDA_exp_list[i], IDA_ed_list[i])
        IDA_ed, time, plasma_dict = load_IDA_data(shot, timepoints=[time], exp=IDA_exp_list[i], ed=IDA_ed_list[i])
        print(time[0])
        IDA_rhop_list.append(IDA_rhop)
        IDA_ECE_res_list.append(IDA_ECE_res)
        IDA_ECE_dat_list.append(IDA_ECE_dat)
        IDA_ECE_unc_list.append(IDA_ECE_unc)
        IDA_ECE_mod_list.append(IDA_ECE_mod)
        ax1.errorbar(IDA_rhop[IDA_ECE_unc < np.max(IDA_ECE_dat)], IDA_ECE_dat[IDA_ECE_unc < np.max(IDA_ECE_dat)], \
                     IDA_ECE_unc[IDA_ECE_unc < np.max(IDA_ECE_dat)], fmt="+", label="ECE data IDA exp {0:s} ed {1:d}".format(IDA_exp_list[i], IDA_ed_list[i]))
        ax1.plot(IDA_rhop, IDA_ECE_mod, "+", label="Mod data IDA exp {0:s} ed {1:d}".format(IDA_exp_list[i], IDA_ed_list[i]))
        ax1.plot(plasma_dict["rhop"][0], plasma_dict["Te"][0], "-", label="Te IDA exp {0:s} ed {1:d}".format(IDA_exp_list[i], IDA_ed_list[i]))
        ax2.plot(IDA_rhop, IDA_ECE_res, "+", label="Residue IDA exp {0:s} ed {1:d}".format(IDA_exp_list[i], IDA_ed_list[i]))
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel(r"$T_\mathrm{e/rad}$ [ev]")
    ax2.set_xlabel(r"$\sigma$")
    ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.show()

if(__name__ == "__main__"):
    compare_IDA_residues(32097, 2.29, ["SDENK", "SDENK", "SDENK"], [26, 28, 29])
