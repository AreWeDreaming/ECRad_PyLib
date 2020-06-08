'''
Created on Aug 24, 2019

@author: sdenk
'''
import sys
import os
from glob import glob
from Diags import Diag
library_list = glob("../*pylib") + glob("../*Pylib")
found_lib = False
ECRadPylibFolder = None
for folder in library_list:
    if("ECRad" in folder or "ecrad"in folder ):
        sys.path.append(folder)
        found_lib = True
        ECRadPylibFolder = folder
        break
if(not found_lib):
    print("Could not find pylib")
    print("Important: ECRad_GUI must be launched with its home directory as the current working directory")
    print("Additionally, the ECRad_Pylib must be in the parent directory of the GUI and must contain one of ECRad, ecrad and Pylib or pylib")
    exit(-1)
from GlobalSettings import globalsettings
globalsettings.ECRadGUIRoot = os.getcwd()
globalsettings.ECRadPylibRoot = ECRadPylibFolder
from plotting_configuration import *
from shotfile_handling_AUG import load_IDA_data, get_shot_heating, get_z_mag, \
                                  moving_average, get_prof, get_Thomson_data
from plotting_core import plotting_core
from ECRad_Results import ECRadResults
from distribution_io import read_waves_mat_to_beam
from scipy.io import loadmat
from equilibrium_utils import EQDataExt

def time_trace(shot, z_axis=False, vline= None):
    plasma_dict = load_IDA_data(shot)
    heating_array = get_shot_heating(shot)
    for i, heating in enumerate(heating_array):
        heating_array[i][0], heating_array[i][1] = moving_average(heating_array[i][0], heating_array[i][1], 5.e-2)
    if(z_axis):
        time_z_axis, z_axis = get_z_mag(shot)
    else:
        time_z_axis = None
        z_axis = None
    fig = plt.figure(figsize=(12,6))
    pc_obj = plotting_core(fig)
    fig = pc_obj.time_trace(shot, plasma_dict["time"], plasma_dict["Te"].T[0], plasma_dict["ne"].T[0], \
                            heating_array=heating_array, time_z_axis = time_z_axis, z_axis=z_axis, vline=vline)
    pc_obj.gridspec.tight_layout(fig, h_pad=0.0)
    plt.show()
    

def plot_shot_geometry(ECRadResFile, wave_mat_file, itime, ch_list, mode_str, tb_data_folder):
    ECRadRes = ECRadResults()
    ECRadRes.from_mat_file(ECRadResFile)
    wave_mat = loadmat(wave_mat_file)
    Beam = read_waves_mat_to_beam(wave_mat, ECRadRes.Scenario.plasma_dict["eq_data"][itime], use_wave_prefix=None)
    args = [None, ECRadRes.Scenario.shot, ECRadRes.Scenario.plasma_dict["time"][itime], \
            ECRadRes.Scenario.plasma_dict["eq_data"][itime].R, ECRadRes.Scenario.plasma_dict["eq_data"][itime].z - 0.01 ,\
            ECRadRes.Scenario.plasma_dict["eq_data"][itime].rhop, ECRadRes.Scenario.plasma_dict["eq_data"][itime].R_ax, \
            ECRadRes.Scenario.plasma_dict["eq_data"][itime].z_ax - 0.01, Beam, ECRadRes, [itime, ch_list], mode_str, tb_data_folder]
    fig = plt.figure(figsize=(12,8))
    fig_2 = plt.figure(figsize=(8,8))
    pc_obj = plotting_core(fig, fig_2)
    pc_obj.beam_plot(args)
    plt.show()
    
    
def compare_IDI_IDF(shot, time):
    rhop_IDI, Ti_IDI = get_prof(shot, time, "IDI", "Ti", exp="AUGD", edition=0)
    rhop_IDF, Ti_IDF = get_prof(shot, time, "IDF", "cde_ti", exp="AUGD", edition=0)
    plt.plot(rhop_IDI, Ti_IDI, "-", label="IDI Ti")
    plt.plot(rhop_IDF, Ti_IDF, "--", label="IDF cde Ti")
    plt.legend()
    plt.gcf().suptitle("IDI vs IDF: {0:d} {1:2.3f}".format(shot, time))
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.gca().set_ylabel(r"$T_\mathrm{i}\,[\si{\electronvolt}]$")
    plt.show()
    
def plot_ne_prof(shot, time):
    plasma_dict = load_IDA_data(shot, [time])
    plt.plot(plasma_dict["rhop_prof"][0], plasma_dict["ne"][0] / 1.e19)
    plt.show()
    
def plot_Thomson_edge_ne(shot, time):    
    diag =Diag("VTA", "AUGD", "VTA",0)
    stddev, data = get_Thomson_data(shot, time, diag, Te=False, ne=True, edge=True, core=False, EQ_exp='AUGD', 
                     EQ_diag="IDE", EQ_ed=0, smoothing=0.5)
    plt.errorbar(data[0], data[1], stddev, fmt="+")
    plt.show()
    
    
    
if(__name__ == "__main__"):
#     plot_Thomson_edge_ne(35662, 3.84)
#     plot_ne_prof(35662, 4.4)
#     time_trace(35662, z_axis=False, vline=3.84)
    plot_shot_geometry("/tokp/work/sdenk/DRELAX_final/DRELAX_run_3224.mat", \
                       "/tokp/work/sdenk/DRELAX_35662_rdiff_prof/ECRad_35662_ECECTCCTA_run3224.mat", 0, [44,88,136], "X", \
                        None)  #94, 144
    
#     compare_IDI_IDF(35662, 4.41)
    
    