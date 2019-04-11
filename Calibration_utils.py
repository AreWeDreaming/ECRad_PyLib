'''
Created on Dec 29, 2015

@author: sdenk
'''
from scipy.signal import medfilt
import numpy as np
import os
from Fitting import make_fit
from GlobalSettings import TCV, AUG
if(AUG):
    from shotfile_handling_AUG import get_data_calib
elif(TCV):
    from shotfile_handling_TCV import get_data_calib
else:
    print('Neither AUG nor TCV selected')
    raise(ValueError('No system selected!'))

def smooth(y_arr, median=False):
    if(median):
#        kernel_size = int(len(y_arr) / 10.e0)
#        if(kernel_size % 2 == 0):
#            kernel_size -= 1
        y_median = medfilt(y_arr)
        y_smooth = np.mean(y_median)
        std_dev = np.std(y_median, ddof=1)
    else:
        y_smooth = np.mean(y_arr)
        std_dev = np.std(y_arr, ddof=1)
    return y_smooth, std_dev

def calibrate(shot, timepoints, Trad_matrix, diag, smoothing, masked_channels=None):
    if(masked_channels is None):
        masked_channels = np.zeros(len(Trad_matrix[0]), dtype=np.bool)
    t_smooth = 1.e-3
    median = True
    calib_mat = []
    std_dev_mat = []
    err_mat = []
    calib_dummy = np.zeros(Trad_matrix[0].shape)
    std_dev_dummy = np.zeros(Trad_matrix[0].shape)  # std dev zero
    ext_resonances_dummy = np.zeros(Trad_matrix.shape)
    calib_dummy[:] = 1.e0  # calibration 1.0
    ext_resonances_dummy[:, :] = 0.e0
    std_dev_data, data = get_data_calib(diag=diag, shot=shot, time=timepoints, \
                   calib=calib_dummy, std_dev_calib=std_dev_dummy, \
                   ext_resonances=ext_resonances_dummy, t_smooth=t_smooth, median=median)
    std_dev_data = std_dev_data[0]
    # Retrieves std deviation and signal from the diagnostic
    ch_cnt = len(Trad_matrix[0])
    for time_index in range(len(timepoints)):
        calib_mat.append([])
        std_dev_mat.append([])
        err_mat.append([])
        for ch in range(ch_cnt):
            try:
                calib_mat[-1].append(Trad_matrix[time_index][ch] / data[1][time_index][ch])
                std_dev_mat[-1].append(np.abs(std_dev_data[time_index][ch]) * np.abs(Trad_matrix[time_index][ch] / data[1][time_index][ch] ** 2))
                err_mat[-1].append(np.abs(std_dev_data[time_index][ch]))
            except Exception as e:
                print("Could not calibrate", "ch" + "{0:d}".format(ch + 1))
                print(e)
                return calib_mat, std_dev_mat, [], [], []
        calib_mat[-1] = np.array(calib_mat[-1])
        std_dev_mat[-1] = np.array(std_dev_mat[-1])
        err_mat[-1] = np.array(err_mat[-1])
    calib_mat = np.array(calib_mat)
    std_dev_mat = np.array(std_dev_mat)
    err_mat = np.array(err_mat)
    sys_dev = np.zeros(ch_cnt)
    if(len(timepoints) == 1):
        calib = calib_mat[0]
        relative_dev = np.copy(std_dev_mat[0] / calib_mat[0])
    else:
        relative_dev = np.zeros(ch_cnt)
        calib = np.zeros(ch_cnt)
        for i in range(len(calib)):
            try:
                popt, perr = make_fit('linear', Trad_matrix.T[i], data[1].T[i], err_mat.T[i], [0.0, 1.0 / np.mean(calib_mat.T[i])])
                # Now add an adhoc systematic error estimated as the largest deviation of the calibration coefficient from measurement
                # minus standard deviation of the measurement
#                max_y_systematic_error = np.max(np.abs(popt[0] + popt[1] * Trad_matrix.T[i] - (data[1].T[i] - np.sign(data[1].T[i]) * err_mat.T[i])))
#                popt, perr = make_fit('linear', Trad_matrix.T[i], data[1].T[i], \
#                                      np.sqrt(err_mat.T[i] ** 2 + max_y_systematic_error ** 2), [0.0, 1.0 / np.mean(calib_mat.T[i])])
                systematic_error = np.sqrt(np.sum((1.0 / popt[1] - (Trad_matrix.T[i] / data[1].T[i])) ** 2) / len(Trad_matrix.T[i]))
                calib[i] = 1.0 / popt[1]
                print("Channel ", i + 1, 'calibration [keV/V] and error [keV/V]', 1.0 / popt[1], perr[1] / popt[1] ** 2)
                print("Pseudo systematic error [%] and systematic vs. statistical error ", np.abs(systematic_error / calib[i] * 100.0), np.abs(systematic_error / (perr[1] / popt[1] ** 2)))
                print("intercept [V] and error [V]", popt[0], perr[0])
                print("Channel ", i + 1, 'mean', np.mean(calib_mat.T[i]))
                relative_dev[i] = np.abs(np.sqrt(perr[1] ** 2 / popt[1] ** 4) / calib[i]) * 100.0
                sys_dev[i] = np.abs(systematic_error / calib[i] * 100.0)
                if(relative_dev[i] > 30):
                    print("Channel ", i + 1, " not usable!")
            except (ValueError, IndexError) as e:
                print(e)
                print("Something wrong with Trad_matrix or calib matrix shape")
                print(Trad_matrix.T[i].shape, calib_mat.T[i].shape)
                raise ValueError
        relative_dev[masked_channels] = 1.0
    return calib_mat, std_dev_mat, calib, relative_dev, sys_dev  # We want percent
