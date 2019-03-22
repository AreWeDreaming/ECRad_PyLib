'''
Created on Jan 29, 2017

@author: sdenk
'''
'''
Created on Dec 9, 2015

@author: sdenk
'''
import numpy as np
import sys
import os
# sys.path.append('/afs/ipp/home/g/git/python/repository/py_rep2.0/')
# import kk
from GlobalSettings import TCV, AUG
if(AUG):
    from equilbrium_utils_AUG import EQData
elif(TCV):
    from equilibrium_utils_TCV import EQData
else:
    print('Neither AUG nor TCV selected')
    raise(ValueError('No system selected!'))
from glob import glob
from plotting_configuration import *
from scipy.signal import medfilt
from Diags import Diag
from scipy.io import loadmat
# import matlab.engine

def get_diag_data_no_calib_wrapper(shot, name, exp="AUGD", diag=None, ed=0):
    if(diag is None):
        if(name == "ECE"):
            diag = "RMD"
        elif(name == "ECN"):
            diag = "TDI"
        elif(name == "ECO"):
            diag = "TDI"
        elif(name == "ECI"):
            diag = "ECI"
        else:
            diag = name
    diagnostic = Diag(name, exp, diag, ed)
    return get_diag_data_no_calib(diagnostic, shot, preview=False, single_channel=0)

def get_Rz_ECI(shot, name):
    raise(ValueError('Routine get_Rz_ECI not available for TCV'))

def get_elm_times(shot):
    raise(ValueError('Routine get_elm_times not available for TCV'))

def get_divertor_currents(shot):
    raise(ValueError('Routine get_divertor_currents not available for TCV'))

def smooth(y_arr, median=False):
    if(median):
#        kernel_size = int(len(y_arr) / 10.e0)
#        if(kernel_size % 2 == 0):
#            kernel_size -= 1
        if(len(y_arr) > 100):
            d = 10
            y_median = y_arr[:(len(y_arr) / d) * d].reshape(-1, d).mean(1)  # this downsamples factor 10
        else:
            y_median = y_arr
        if(len(y_median) / 3 / 2 * 2 + 1 > 3):
            y_median = medfilt(y_median, len(y_median) / 3 / 2 * 2 + 1)  # broad median filter
        if(len(y_median) > 1):
            y_smooth = np.mean(y_median)  # make sure we get only one value
            std_dev = np.std(y_median, ddof=1)
        else:
            y_smooth = np.mean(y_arr)
            std_dev = 0.0
    else:
        if(len(y_median) > 1):
            y_smooth = np.mean(y_arr)  # make sure we get only one value
            std_dev = np.std(y_arr, ddof=1)
        else:
            y_smooth = y_arr[0]
            std_dev = 0.0
    return y_smooth, std_dev

def get_diag_data_no_calib(diag, shot, preview=False, single_channel=0):
    raise(ValueError('Routine get_diag_data_no_calib not available for TCV'))

def get_data_calib_entire_shot(diag, shot, ext_resonances=None, calib=None):
    raise(ValueError('Routine get_data_calib_entire_shot not available for TCV'))

def get_data_calib(diag=None, diag_id=None, shot=0, time=None, exp="AUGD", ed=0, eq_exp="AUGD", eq_diag="EQH", \
                   eq_ed=0, calib=None, std_dev_calib=None, ext_resonances=None, name="", t_smooth=None, median=True):
    raise(ValueError('Routine get_data_calib not available for TCV'))

def get_CTA_no_pinswitch(shot, diag, exp, ed, ch_in=None, t_shift_back=175.e-6, t_shift_fwd=100.e-6):
    raise(ValueError('Routine get_CTA_no_pinswitch not available for TCV'))


def make_CCE_diag_launch(shot, shot_data_file):
    at_least_1d_keys = ["phi_tor", "theta_pol", 'beam_width', 'dist_foc']
    variable_names = at_least_1d_keys + ["R_launch", "phi_launch", "z_launch"]
    try:
        mdict = loadmat(shot_data_file, chars_as_strings=True, squeeze_me=True, variable_names=variable_names)
    except IOError:
        print("Error: " + shot_data_file + " does not exist")
        return
    for key in mdict.keys():
        if(not key.startswith("_")):  # throw out the .mat specific information
            try:
                if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                    mdict[key] = np.array([mdict[key]])
            except Exception as e:
                print(key)
                print(e)
    eng = matlab.engine.start_matlab()
    f_val = eng.load_freq_CECE(shot)
    f = np.zeros((len(f_val), len(mdict['phi_tor'])))
    launch_geo = []
    launch_geo.append(f)
    df = np.zeros(f.shape)
    df[:] = 0.1e9
    launch_geo.append(df)
    R_launch = np.zeros(f.shape)
    R_launch[:] = mdict["R_launch"]
    launch_geo.append(R_launch)
    phi_launch = np.zeros(f.shape)
    phi_launch[:] = mdict["phi_launch"]
    launch_geo.append(phi_launch)
    z_launch = np.zeros(f.shape)
    z_launch[:] = mdict["z_launch"]
    launch_geo.append(z_launch)
    theta_pol = []
    phi_tor = []
    dist_foc = []
    width = []
    for i in range(len(f)):
        theta_pol.append(np.rad2deg(mdict["theta_pol"]))
        phi_tor.append(np.rad2deg(mdict["phi_tor"]))
        dist_foc.append(mdict["dist_foc"])
        width.append(mdict["width"])
    theta_pol = np.array(theta_pol)
    launch_geo.append(theta_pol)
    phi_tor = np.array(phi_tor)
    launch_geo.append(phi_tor)
    dist_foc = np.array(dist_foc)
    launch_geo.append(dist_foc)
    width = np.array(width)
    launch_geo.append(width)
    return launch_geo




def filter_CTA(shot, time, diag, exp, ed):
    raise(ValueError('Routine filter_CTA not available for TCV'))

def filter_ECRH(shot, time, diag, exp, ed):
    raise(ValueError('Routine filter_ECRH not available for TCV'))

def get_ECRH_PW(shot, diag, exp, ed):
    raise(ValueError('Routine get_ECRH_PW not available for TCV'))

def get_freqs(shot, diag):
    raise(ValueError('Routine get_freqs not available for TCV'))

def get_ECI_launch(diag, shot):
    raise(ValueError('Routine get_ECI_launch not available for TCV'))

def get_shot_heating(shot):
    raise(ValueError('Routine get_shot_heating not available for TCV'))

def get_NPA_data(shot):
    raise(ValueError('Routine get_NPA_data not available for TCV'))

def get_Thomson_data(shot, times, diag, Te=False, ne=False, edge=False, core=False, eq_diag="EQH"):
    raise(ValueError('Routine get_Thomson_data not available for TCV'))

def get_cold_resonances_S_ECE(shot, time, diag_name, R_min, R_max, z_min, z_max, B_spline, ch_no, exp="AUGD", diag="None", ed=0):
    raise(ValueError('Routine get_cold_resonances_S_ECE not available for TCV'))

def test_resonance():
    raise(ValueError('Routine test_resonance not available for TCV'))

def load_IDA_data(shot, timepoints=None, exp="AUGD", ed=0):
    raise(ValueError('Routine load_IDA_data not available for TCV'))

def make_ext_data_for_testing(ext_data_folder, shot, times, eq_exp, eq_diag, eq_ed, bt_vac_correction=1.005, IDA_exp="AUGD", IDA_ed=0):
    raise(ValueError('Routine get_current not available for TCV'))

def make_ext_data_for_testing_grids(ext_data_folder, shot, times, eq_exp, eq_diag, eq_ed, bt_vac_correction=1.005):
    raise(ValueError('Routine get_current not available for TCV'))

def get_RELAX_target_current(shot, time, exp="AUGD", ed=0, smoothing=1.e-3):
    raise(ValueError('Routine get_current not available for TCV'))

def get_ECE_spectrum(folder, shotno, time, diag, Te):
    raise(ValueError('Routine get_current not available for TCV'))
