'''
Created on Jan 4, 2021

@author: root
'''
import numpy as np

class ECRadCalibResults(dict):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self["calib"] = ["channelno", "calib", "rel_dev", "sys_dev"]
        self["calib_trace"] = ["calib_mat"]
        
    def from_mat(self, mdict):
        try:
            if("calib_diags" in mdict  and len(mdict["calib"]) > 0):
                if(len(mdict["calib_diags"]) == 1):
                    self.calib[mdict["calib_diags"][0]] = mdict["calib"][0]
                    self.calib_mat[mdict["calib_diags"][0]] = mdict["calib_mat"][0].T
                    self.std_dev_mat[mdict["calib_diags"][0]] = mdict["std_dev_mat"][0].T
                    self.rel_dev[mdict["calib_diags"][0]] = mdict["rel_dev"][0]
                    try:
                        self.sys_dev[mdict["calib_diags"][0]] = mdict["sys_dev"][0]
                    except KeyError:
                        print("No systematic errors in .mat file")
                        self.sys_dev[mdict["calib_diags"][0]] = np.zeros(self.rel_dev[mdict["calib_diags"][0]].shape)
                    try:
                        self.masked_time_points[mdict["calib_diags"][0]] = bool(mdict["masked_time_points"][0])
                    except KeyError:
                        print("Masked time points for calibration not specified")
                        self.masked_time_points[mdict["calib_diags"][0]] = np.zeros(self.time.shape, dtype=bool)
                        self.masked_time_points[mdict["calib_diags"][0]][:] = True
                else:
                    for i in range(len(mdict["calib_diags"])):
                        self.calib[mdict["calib_diags"][i]] = mdict["calib"][i]
                        self.calib_mat[mdict["calib_diags"][i]] = mdict["calib_mat"][i].T
                        self.std_dev_mat[mdict["calib_diags"][i]] = mdict["std_dev_mat"][i].T
                        self.rel_dev[mdict["calib_diags"][i]] = mdict["rel_dev"][i]
                        try:
                            self.sys_dev[mdict["calib_diags"][i]] = mdict["sys_dev"][i]
                        except KeyError:
                            print("No systematic errors in .mat file")
                            self.sys_dev[mdict["calib_diags"][i]] = np.zeros(self.rel_dev[mdict["calib_diags"][i]].shape)
                        try:
                            self.masked_time_points[mdict["calib_diags"][i]] = bool(mdict["masked_time_points"][i])
                        except KeyError:
                            print("Masked time points for calibration not specified")
                            self.masked_time_points[mdict["calib_diags"][i]] = np.zeros(self.time.shape, dtype=bool)
                            self.masked_time_points[mdict["calib_diags"][i]][:] = True
        except TypeError:
            print("Error loading calibration factors - please recalculate")
        