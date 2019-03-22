'''
Created on May 9, 2018

@author: sdenk
'''

from shotfile_handling_AUG import load_IDA_data
from equilibrium_utils_AUG import EQData
from plotting_configuration import *

def prepare_GENE_Te_input(shot, time, IDA_exp, IDA_ed, EQ_exp, EQ_diag, EQ_ed):
    IDA_ed, time, IDA_dict = load_IDA_data(shot, timepoints=[time], exp=IDA_exp, ed=IDA_ed)
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    rhop = IDA_dict["rhop"][0]
    Te = IDA_dict["Te"][0]
    ne = IDA_dict["ne"][0]
    Te = Te[rhop < 0.99]
    ne = ne[rhop < 0.99]
    rhop = rhop[rhop < 0.99]
    rhot = EQ_obj.rhop_to_rot(time, rhop)
    plt.plot(rhot, Te)
    plt.figure()
    plt.plot(rhot, ne)
    plt.show()


if(__name__ == "__main__"):
    prepare_GENE_Te_input(34663, 3.600, "SDENK", 7, "AUGD", "IDE", 0)
