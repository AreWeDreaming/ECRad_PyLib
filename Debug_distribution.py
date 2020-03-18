'''
Created on Feb 16, 2020

@author: sdenk
'''
import numpy as np
from distribution_io import load_f_from_mat
from plotting_configuration import plt

def test_dstribution(rhop, filename):
    dist_obj = load_f_from_mat(filename, use_dist_prefix=None)
    f_ind = np.argmin(np.abs(dist_obj.rhop - rhop))
    print("Rhop: ", dist_obj.rhop[f_ind])
    try:
        cmap = plt.cm.get_cmap("plasma")
    except ValueError:
        cmap = plt.cm.get_cmap("jet")
    levels = np.linspace(-13, 5, 10)
    cont1 = plt.contourf(dist_obj.pitch, dist_obj.u, dist_obj.f_log10[f_ind], levels=levels, cmap=cmap)
    plt.show()
    
if(__name__ == "__main__"):
    test_dstribution(0.18, "/tokp/work/sdenk/DRELAX_35662_rdiff_prof/ECRad_35662_ECECTCCTA_run3224.mat")
