'''
Created on Mar 14, 2016

@author: sdenk
'''
from plotting_core import plotting_core
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
def tb_plot(working_dir=None, shot=None, time=None):
    if(working_dir is None):
        folder = os.getcwd()
    else:
        folder = working_dir
    if(shot is None):
        shot = int(sys.argv[1])
    if(time is None):
        time = float(sys.argv[2])
    fig = plt.figure(figsize=(12.0, 8.5), tight_layout=False)
    pc_obj = plotting_core(fig)
    pc_obj.tb_plot(shot, time, folder)
    plt.show()

def ECFM_plot():
    folder = os.getcwd()
    shot = int(sys.argv[1])
    time = float(sys.argv[2])
    fig = plt.figure(figsize=(12.0, 8.5), tight_layout=False)
    pc_obj = plotting_core(fig)
    pc_obj.ECFM_plot(shot, time, folder)
    plt.show()

# tb_plot()

def comp_tb_ECFM_input():
    R_mat = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/Ecfm_Model_new/R_test")
    z_mat = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/Ecfm_Model_new/z_test")
    rhop = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/Ecfm_Model_new/rhop_test")
    ECFM_ray = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/Ecfm_Model_new/k_out_ecfm")
    tb_ray = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90//torbeam/k_out")
    svec = np.loadtxt("/ptmp1/work/sdenk/ECFM/ecfm_data/chdata050.dat")
    fig = plt.figure(figsize=(12.0, 8.5), tight_layout=False)
    fig2 = plt.figure(figsize=(12.0, 8.5), tight_layout=False)
    pc_obj = plotting_core(fig, fig2)
    N_tb = func_N(tb_ray.T[7], tb_ray.T[8], tb_ray.T[9], -1.0)
    N_ECFM = func_N(ECFM_ray.T[7], ECFM_ray.T[8], ECFM_ray.T[9], -1.0)
    pc_obj.tb_check_plot(tb_ray, ECFM_ray, R_mat, z_mat, rhop, N_tb, N_ECFM)
    plt.show()

def func_N(X, Y, theta, mode):  # Following eq 13(a -e) of ref [1]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    rho = Y ** 2 * sin_theta ** 4 + 4.e0 * (1.e0 - X) ** 2 * cos_theta ** 2
    rho[rho < 0.e0] = 0.e0
    rho = np.sqrt(rho)
    f = (2.e0 * (1.e0 - X)) / (2.e0 * (1.e0 - X) - Y ** 2 * sin_theta ** 2 + mode * Y * rho)
    func_N = 1.e0 - X * f
    func_N[func_N < 0.e0] = 0.e0
    func_N = np.sqrt(func_N)
    return func_N

tb_plot("/afs/ipp-garching.mpg.de/home/s/sdenk/F90//torbeam/", 32028, 4.20)
# comp_tb_ECFM_input()
