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
from glob import glob
from plotting_configuration import *
from scipy.signal import medfilt
from Diags import Diag
from scipy.io import loadmat
from get_ECRH_config import get_ECRH_viewing_angles
# TODO import matlab.engine "matlab"


def make_CCE_diag_launch(shot, shot_data_file):
    at_least_1d_keys = ["phi_tor", "theta_pol", 'beam_width', 'dist_foc']
    variable_names = at_least_1d_keys + ["R_launch", "phi_launch", "z_launch"]
    try:
        mdict = loadmat(shot_data_file, chars_as_strings=True, squeeze_me=True, variable_names=variable_names)
    except IOError:
        print("Error: " + shot_data_file + " does not exist")
        return
    for key in mdict:
        if(not key.startswith("_")):  # throw out the .mat specific information
            try:
                if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                    mdict[key] = np.array([mdict[key]])
            except Exception as e:
                print(key)
                print(e)
    eng = matlab.engine.start_matlab() # Will not work until this is fixed
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

