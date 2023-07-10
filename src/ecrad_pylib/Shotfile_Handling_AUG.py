'''
Created on Dec 9, 2015

@author: Severin Denk
'''
import numpy as np
import sys
import os
# sys.path.append('/afs/ipp/home/g/git/python/repository/py_rep2.0/')
# import kk
sys.path.append('/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib')
import dd
from scipy.signal import medfilt
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, UnivariateSpline, interp1d
from ecrad_pylib.Equilibrium_Utils_AUG import EQData
import scipy.constants as cnst
from ecrad_pylib.Diag_Types import Diag
from ecrad_pylib.Data_Processing import remove_mode
from ecrad_pylib.Get_ECRH_Config import get_ECRH_viewing_angles
from ecrad_pylib.Plotting_Configuration import plt
AUG_profile_diags = ["IDA", "RMD", "CEC", "VTA", "CEZ", "COZ", "CUZ"]

def get_HEP_ne(shot, exp="AUGD", ed=0):
    sf = dd.shotfile(diagnostic="HEP", pulseNumber=shot, experiment=exp,edition=ed)
    ne = sf.getSignalGroup("ne", dtype=np.double)
    time = sf.getTimeBase("ne", dtype=np.double).data
    rho_pol = sf.getAreaBase("ne", dtype=np.double).data
    return time, rho_pol, ne

def shotfile_exists(shot, diag):
    if(hasattr(diag, "diag")):
        try:
            dd.shotfile(diagnostic=diag.diag, pulseNumber=shot, experiment=diag.exp, edition=diag.ed)
            return True
        except dd.PyddError:
            return False
    else:
        return False

def get_prof(shot, time, diag, sig, exp="AUGD", edition=0):
    DIAG = dd.shotfile(diag, int(shot), experiment=exp, edition=edition)
    prof = DIAG.getSignalGroup(\
                    sig, dtype=np.double)
    rhop = DIAG.getAreaBase(\
                    sig, dtype=np.double).data
    time_trace = DIAG.getTimeBase(sig)
    itime = np.argmin(np.abs(time_trace - time))
    return rhop[itime], prof[itime]



def get_diag_data_no_calib_wrapper(shot, name, exp="AUGD", diag="None", ed=0):
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
    if(name == "ECO"):
        diag_shotfile = dd.shotfile("RZO", int(shot), experiment="ECEI", edition=0)
    else:
        diag_shotfile = dd.shotfile("RZO", int(shot), experiment="ECEI", edition=0)
    R = diag_shotfile.getSignalGroup("R")
    z = diag_shotfile.getSignalGroup("z")
    t = diag_shotfile.getTimeBase("R")
    return t, R, z

def get_elm_times(shot):
    try:
        diag_shotfile = dd.shotfile("ELM", int(shot))
        elm_end = diag_shotfile.getSignalCalibrated("t_endELM")[0]
        elm_beg = diag_shotfile.getTimeBase("t_endELM")
        time_last = 0.0
        time = np.array([])
        elm = np.array([])
        for i in range(len(elm_beg)):
            no_elms = np.zeros(20, dtype=bool)
            elms = np.zeros(20, dtype=bool)
            elms[:] = 1
            time = np.concatenate([time, np.linspace(time_last, elm_beg[i], 20)])
            elm = np.concatenate([elm, no_elms])
            time = np.concatenate([time, np.linspace(elm_beg[i], elm_end[i] , 20)])
            elm = np.concatenate([elm, elms])
            time_last = elm_end[i]
    except Exception as e:
        print("Could not open shotfile ELM for", shot)
        print("reason", e)
        raise Exception(e)
        return [], []
    return time, elm

def get_divertor_currents(shot):
    try:
        diag_shotfile = dd.shotfile("MAC", int(shot))
        signal = diag_shotfile.getSignalCalibrated("Ipolsola")[0]
        time = diag_shotfile.getTimeBase("Ipolsola")
        return time, signal
    except Exception as e:
        print("Could not open shotfile MAC for", shot)
        print("reason", e)
        raise Exception(e)
        return [], []

def smooth(y_arr, median=False, use_std_err=False):
    if(median):
#        kernel_size = int(len(y_arr) / 10.e0)
#        if(kernel_size % 2 == 0):
#            kernel_size -= 1
        if(len(y_arr) > 100):
            d = 10
            y_median = y_arr[:(len(y_arr) // d) * d].reshape(-1, d).mean(1)  # this downsamples factor 10 with a mean
        else:
            y_median = y_arr
        if(len(y_median) // 3 // 2 * 2 + 1 > 3):
            y_median = medfilt(y_median, len(y_median) // 3 // 2 * 2 + 1)  # broad median filter
        if(len(y_median) > 1):
            y_smooth = np.mean(y_median)  # make sure we get only one value
            if(len(y_median) > 100 and use_std_err):
                print("Using standard error for uncertainty")
                std_dev = np.std(y_median, ddof=1) / np.sqrt(np.float(len(y_arr)))
            else:
                print("Using standard deviation for uncertainty")
                std_dev = np.std(y_median, ddof=1)
        else:
            print("Singe data point -> no means of computing uncertainty")
            y_smooth = np.mean(y_arr)
            std_dev = 0.0
    else:
        if(len(y_arr) > 1):
            y_smooth = np.mean(y_arr)  # make sure we get only one value
            if(len(y_arr) > 100 and use_std_err):
                print("Using standard error for uncertainty")
                std_dev = np.std(y_arr, ddof=1) / np.sqrt(np.float(len(y_arr)))
            else:
                print("Using standard deviation for uncertainty")
                std_dev = np.std(y_arr, ddof=1)
        else:
            print("Singe data point -> no means of computing uncertainty")
            y_smooth = y_arr[0]
            std_dev = 0.0
    return y_smooth, std_dev

def moving_average(time, signal, window):
    step = np.mean(np.gradient(time))
    N = int(np.floor(window / step))
    if(N < 4):
        print("Window too small for resolution of signal - returning original signal")
        return time, signal
    else:
        return time[:len(time) - N + 1], np.convolve(signal, np.ones(N)/N, mode="valid")

def get_diag_data_no_calib(diag, shot, preview=False, single_channel=0):
    print("Told to get " + diag.diag + " data")
    try:
        diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
    except Exception as e:
        print("Could not open shotfile for ", diag.exp, " ", diag.diag, " ", str(diag.ed))
        print("reason", e)
        return [], []
    signals = []
    if(diag.name == "CTC" or diag.name == "IEC" or diag.name == "CTA"):
        if(diag.name == "IEC"):
            time = diag_shotfile.getTimeBase("T-" + diag.diag)
        else:
            try:
                time = diag_shotfile.getTimeBase("ch" + "1     ") - 4.1e-4
            except dd.PyddError:
                steps = len(diag_shotfile.getSignal("ch" + "1     "))
                time = (np.linspace(0.0, 10.0, steps) - 4.1e-4) * 0.899971209232552
                # correction factor for the corrupted time base
                # Obtained by correlation off the pin switch and ECRH power
                print('Warning time base incorrect - adjusting time base')
        if(diag.diag == "IEC"):
            ch_cnt = 6
        elif(diag.diag == "CTC"):
            ch_cnt = 42
        else:
            ch_cnt = 50
        if(single_channel == 0):
            for ch in range(ch_cnt):
                try:
                    if(diag.name == "CTC" or diag.name == "CTA"):
                        if(ch + 1 < 10):
                            ch_str = "ch" + "{0:d}     ".format(ch + 1)
                        else:
                            ch_str = "ch" + "{0:d}    ".format(ch + 1)
                        signal = diag_shotfile.getSignal(ch_str)
                    elif(diag.name == "IEC"):
                        signal = diag_shotfile.getSignal(\
                          "ECE" + str(ch + 1) + "_raw")
                    signals.append(signal)
                except Exception as e:
                    print("Could not read signal", "ch" + "{0:d}    ".format(ch + 1))
                    print(e)
                    return [], []
        else:
            if(single_channel < 10):
                ch_str = "ch" + "{0:d}     ".format(single_channel)
            else:
                ch_str = "ch" + "{0:d}    ".format(single_channel)
            signal = diag_shotfile.getSignal(ch_str)
            return time, signal
    elif(diag.diag == "ECI"):
        time = diag_shotfile.getTimeBase("time")
        if(preview):
            # Only load one LOS
            LOS = 4
            try:
                signalGroup = diag_shotfile.getSignalGroup(\
                          "LOS" + str(LOS + 1)).T
                for i in range(8):
                    signals.append(signalGroup[i])
            except Exception as e:
                print("Could not read signal", "LOS" + "{0:d}".format(LOS + 1))
                print(e)
                return  [], []
        else:
            for LOS in range(16):
                try:
                    signalGroup = diag_shotfile.getSignalGroup(\
                              "LOS" + str(LOS + 1)).T
                    for i in range(8):
                        signals.append(signalGroup[i])
                except Exception as e:
                    print("Could not read signal", "LOS" + "{0:d}".format(LOS + 1))
                    print(e)
                    return  [], []
    elif(diag.diag == "TDI" and diag.name == "ECO"):
        time = diag_shotfile.getTimeBase("Sig1")
        if(preview):
            try:
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig2", tBegin=0.e0)
                for i in range(8):
                    signals.append(signalGroup[i])
            except Exception as e:
                print("Could not read signal for Sig2")
                print(e)
                return [], []
        else:
            try:
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig1", tBegin=0.e0)
                for i in range(72):
                    signals.append(signalGroup[i])
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig2", tBegin=0.e0)
                for i in range(56):
                    signals.append(signalGroup[i])
            except Exception as e:
                print("Could not read signal for either Sig1 or Sig2")
                print(e)
                return [], []
    elif(diag.diag == "TDI" and diag.name == "ECN"):
        time = diag_shotfile.getTimeBase("Sig1")
        if(preview):
            try:
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig3", tBegin=0.e0)
                for i in range(56, 64):
                    signals.append(signalGroup[i])
            except Exception as e:
                print("Could not read signal for Sig3")
                print(e)
                return [], []
        else:
            try:
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig2")
                for i in range(56, 72):
                    signals.append(signalGroup[i])
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig3")
                for i in range(72):
                    signals.append(signalGroup[i])
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig4")
                for i in range(72):
                    signals.append(signalGroup[i])
            except Exception as e:
                print("Could not read signal for either Sig2, Sig3 or Sig4")
                print(e)
                return [], []
    else:
        print("Error: Unknown diagnostic: ", diag.name, diag.diag)
        raise Exception
    return time, np.array(signals)

def get_data_calib_entire_shot(diag, shot, ext_resonances=None, calib=None):
    if(diag.diag == "RMD"):
        try:
            diag_shotfile = dd.shotfile("RMD", int(shot), experiment=diag.exp, edition=diag.ed)
        except:
            try:
                print("Warning no RMD shotfile found - falling back to CEC")
                diag_shotfile = dd.shotfile("CEC", int(shot), experiment=diag.exp, edition=diag.ed)
                diag.diag = "CEC"
            except Exception as e:
                print("Could not open shotfile for ", diag.exp, " ", diag.diag)
                print("reason", e)
                return None, None
    elif(diag.diag == "CEC"):
        try:
            diag_shotfile = dd.shotfile("CEC", int(shot), experiment=diag.exp, edition=diag.ed)
        except:
            try:
                print("Warning no CEC shotfile found - trying RMD")
                diag_shotfile = dd.shotfile("RMD", int(shot), experiment=diag.exp, edition=diag.ed)
                diag.diag = "RMD"
            except Exception as e:
                print("Could not open shotfile for ", diag.exp, " ", diag.diag)
                print("reason", e)
                return None, None
    else:
        if(ext_resonances is None or calib is None):
            print("Cannot load calibrated diagnostic " + diag + " without calibration and resonances")
    if(diag.diag == "CEC" or diag.diag == "RMD"):
        print("Getting " + diag.diag + " data")
        signalGroup = diag_shotfile.getSignalGroup("Trad-A")
        # use_Channel = np.array(diag_shotfile.getParameter('parms-A', 'AVAILABL').data)
        t = diag_shotfile.getTimeBase("Trad-A")
        return t, signalGroup.T


def get_data_calib(diag, shot=0, time=None, eq_exp="AUGD", eq_diag="EQH", \
                   eq_ed=0, calib=None, std_dev_calib=None, sys_dev_calib=None, \
                   ext_resonances=None, name="", t_smooth=None, median=True, \
                   aux_diag=None, use_std_err=True):
    # Gets the data from all ECE diagnostics that have shotfiles
    # Returns std deviation in keV, rho poloIDAl resonance and Trad in keV
    if(t_smooth is None):
        t_smooth = diag.t_smooth
    if(len(name) == 0):
        name = diag.name
    print("Told to get " + diag.diag + " data")
    if(type(time) == str):
        print("time must be either a scalar number or an array of numbers")
    if(not hasattr(time, "__len__")):
        times = np.array([time])
        if(ext_resonances is not None):
            ext_resonances = np.array([ext_resonances])
        print("Single time point")
    else:
        times = time
        print("Multiple time points")
    data = []
    rhop = []
    std_dev_data = []
    sys_dev_data = []
    if(diag.diag == "RMD"):
        try:
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=int(diag.ed))
        except:
            try:
                print("Warning no RMD shotfile found - falling back to CEC")
                diag_shotfile = dd.shotfile("CEC", int(shot), experiment=diag.exp, edition=diag.ed)
                diag.diag = "CEC"
            except Exception as e:
                print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
                print("reason", e)
                return None, None
    elif(diag.diag == "CEC"):
        try:
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
        except:
            try:
                print("Warning no CEC shotfile found - trying RMD")
                diag_shotfile = dd.shotfile("RMD", int(shot), experiment=diag.exp, edition=diag.ed)
                diag.diag = "RMD"
            except Exception as e:
                print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
                print("reason", e)
                return None, None
    else:
        if(calib is None):
            print("Cannot load uncalibrated diagnostic " + diag.diag + " without cross calibration")
            raise IOError
    # Diagnostics with no calibrated shot file
    if(diag.diag != "CEC" and diag.diag != "RMD"):
        ch_cnt = len(calib)
    if(diag.name == "CTC" or diag.name == "IEC" or diag.name == "CTA"):
        try:
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
        except Exception as e:
            print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
            print("reason", e)
            return None, None
    elif(diag.name == "ECI"):
        try:
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
            diag_aux_shotfile = dd.shotfile(diag.Rz_diag, int(shot), experiment=diag.Rz_exp, edition=diag.Rz_ed)
        except Exception as e:
            print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
            print("reason", e)
            return None, None
    elif(diag.name == "ECO"):
        try:
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
            diag_aux_shotfile = dd.shotfile(diag.Rz_diag, int(shot), experiment=diag.Rz_exp, edition=diag.Rz_ed)
        except Exception as e:
            print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
            print("reason", e)
            return None, None
    elif(diag.name == "ECN"):
        try:
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
            diag_aux_shotfile = dd.shotfile(diag.Rz_diag, int(shot), experiment=diag.Rz_exp, edition=diag.Rz_ed)
        except Exception as e:
            print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
            print("reason", e)
            return None, None
    elif(diag.name == "ECE" and diag.diag == "RMC"):
        try:
            if(aux_diag is None):
                print("To load RMC data aux_diag must be provided to get_data_calib")
                raise(ValueError)
            diag_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
            diag_aux_shotfile = dd.shotfile(aux_diag.diag, int(shot), experiment=aux_diag.exp, edition=aux_diag.ed)
        except Exception as e:
            print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
            print("reason", e)
            return None, None
    if(diag.diag == "CEC" or diag.diag == "RMD"):
        print("Getting " + diag.diag + " data")
        use_Channel = np.array(diag_shotfile.getParameter('parms-A', 'AVAILABL').data)
        SNR_cali_channel = np.array(diag_shotfile.getParameter('parms-A', 'SNR_cali').data)[use_Channel == 1]
        try:
            signals = diag_shotfile.getSignalGroup("Trad-A").T[use_Channel == 1]
        except Exception as e:
            print("Could not open shotfile for " + diag.exp + " " + diag.diag + " ed " + str(diag.ed))
            print("reason", e)
            return None, None
        diag_time = diag_shotfile.getTimeBase("Trad-A")
        R = diag_shotfile.getSignalGroup("R-A").T[use_Channel == 1]
        z = diag_shotfile.getSignalGroup("z-A").T[use_Channel == 1]
        R_z_timebase = diag_shotfile.getTimeBase("rztime")
    elif(diag.name == "CTC" or diag.name == "IEC" or diag.name == "CTA"):
        signals = []
        for ch in range(ch_cnt):
            try:
                if(diag.name == "CTC" or diag.name == "CTA"):
                    if(ch + 1 < 10):
                        ch_str = "ch" + "{0:d}     ".format(ch + 1)
                    else:
                        ch_str = "ch" + "{0:d}    ".format(ch + 1)
                    signals.append(diag_shotfile.getSignal(ch_str))
                elif(diag.name == "IEC"):
                    signals.append(diag_shotfile.getSignal(\
                      "ECE" + str(ch + 1) + "_raw"))
            except Exception as e:
                print("Could not read signal for data", "ch" + "{0:d}".format(ch + 1))
                print(e)
                return None, None
        if(diag.name == "CTC" or diag.name == "CTA"):
            ch = 1
            ch_str = "ch" + "{0:d}     ".format(ch + 1)
            try:
                diag_time = diag_shotfile.getTimeBase("ch" + "1     ") - 4.1e-4
            except dd.PyddError:
                steps = len(diag_shotfile.getSignal("ch" + "1     "))
                diag_time = (np.linspace(0.0, 10.0, steps) - 4.1e-4) * 0.8999816246417855
                # correction factor for the corrupted time base
                # Obtained by correlation off the pin switch and ECRH power
                print('Warning time base incorrect - adjusting time base')
        elif(diag.name == "IEC"):
            ch = 1
            diag_time = diag_shotfile.getTimeBase(\
              "ECE" + str(ch + 1) + "_raw")
    elif(diag.name == "ECO"):
        if(diag.diag == "TDI"):
            try:
                signals = []
                diag_time = diag_shotfile.getTimeBase("Sig1")
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig1")
                ch = 0
                for i in range(72):
                    signals.append(signalGroup[i])
                signalGroup = diag_shotfile.getSignalGroup(\
                          "Sig2")
                for i in range(56):
                    signals.append(signalGroup[i])
            except Exception as e:
                print("Could not read signal for either Sig1 or Sig2")
                print(e)
                return None, None
        else:
            ch = 0
            signals = []
            diag_time = diag_shotfile.getTimeBase("LOS1")
            for LOS in range(16):
                try:
                    signalGroup = diag_shotfile.getSignalGroup(\
                              "LOS" + str(LOS + 1)).T
                    for i in range(8):
                        for j in range(len(signalGroup[i])):
                            signals.append(signalGroup[i][j])
                except Exception as e:
                    print("Could not read signal for scale", "LOS" + "{0:d}".format(LOS + 1))
                    print(e)
                    return None, None
    elif(diag.name == "ECN"):
        signals = []
        diag_time = diag_shotfile.getTimeBase("Sig2")
        try:
            signalGroup = diag_shotfile.getSignalGroup(\
                      "Sig2")
            ch = 0
            for i in range(56, 72):
                signals.append(signalGroup[i])
            signalGroup = diag_shotfile.getSignalGroup(\
                      "Sig3")
            for i in range(72):
                signals.append(signalGroup[i])
            signalGroup = diag_shotfile.getSignalGroup(\
                      "Sig4")
            for i in range(72):
                signals.append(signalGroup[i])
        except Exception as e:
            print("Could not read signal for either Sig2, Sig3 or Sig4")
            print(e)
            return None, None
    elif(diag.diag == "RMC"):
        avail = diag_aux_shotfile.getParameter('parms-A', 'AVAILABL', dtype=np.float64).data
        signals = (np.concatenate((diag_shotfile('Trad-A1', dtype=np.float64).data, \
                                  diag_shotfile('Trad-A2', dtype=np.float64).data), axis=1)[:, avail==1]).T
        diag_time = diag_shotfile.getTimeBase('Trad-A1')
    else:
        print("Error diag", diag.diag, " is unknown")
        return None, None
    print("Processing a total of {0:d} timepoints".format(len(times)))
    print("Processing a total of {0:d} channels".format(len(signals)))
    if(diag.diag != "CEC" and diag.diag != "RMD"):
        # Get the offset on the first go then get the actual data
        new_times = np.concatenate([np.array([0.0]), times])
#        print("times with offset", new_times)
        offset = np.zeros(len(signals))
        if(ext_resonances is None):
            print("ERROR! get_data_calib requires external resonances for all non-standard ECE diagnostics!")
            print("Setting all resonances to zeros")
            ext_resonances = np.zeros((len(times), len(signals)))
    else:
        new_times = times
    if(diag.name == "ECN" or diag.name == "ECO"):
        aux_time = diag_aux_shotfile.getTimeBase("offset")
        ECEI_offset = diag_aux_shotfile.getSignalGroup("offset")
        ECEI_norm = diag_aux_shotfile.getSignalGroup("norm")
        ECEI_offset_arr = []
        ECEI_norm_arr = []
        for it in range(len(aux_time)):
            ECEI_offset_arr.append(ECEI_offset[it].flatten())
            ECEI_norm_arr.append(ECEI_norm[it].flatten())
        ECEI_offset = np.array(ECEI_offset_arr).T
        ECEI_norm = np.array(ECEI_norm_arr).T
        ECEI_offset_spls = []
        ECEI_norm_spls = []
        for ich in range(len(signals)):
            ECEI_norm_spls.append(InterpolatedUnivariateSpline(aux_time, ECEI_norm[ich]))
            ECEI_offset_spls.append(InterpolatedUnivariateSpline(aux_time, ECEI_offset[ich]))
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    for it in range(len(new_times)):
        if(diag.diag == "CEC" or diag.diag == "RMD" or it >= 1):
            data.append([])
        it0 = np.argmin(np.abs(diag_time - (new_times[it] - 0.5 * t_smooth)))
        it1 = np.argmin(np.abs(diag_time - (new_times[it] + 0.5 * t_smooth)))
        if(it1 - it0 == 0):
            it1 += 1
        if((diag.diag == "CEC" or diag.diag == "RMD")):
            if(ext_resonances is None):
                itR0 = np.argmin(np.abs(R_z_timebase - (new_times[it] - 0.5 * t_smooth)))
                itR1 = np.argmin(np.abs(R_z_timebase - (new_times[it] + 0.5 * t_smooth)))
                if(itR1 - itR1 <= 0):
                    R_cur = R.T[itR0]
                    z_cur = z.T[itR0]
                else:
                    R_cur = np.mean(R[itR0:itR1 + 1], axis=0)
                    z_cur = np.mean(z[itR0:itR1 + 1], axis=0)
                rhop.append(EQ_obj.map_Rz_to_rhop(new_times[it], R_cur, z_cur))
            else:
                rhop.append(ext_resonances[it])
            std_dev_data.append([])
            sys_dev_data.append([])
        elif(it >= 1):
            rhop.append([])
            std_dev_data.append([])
            sys_dev_data.append([])
        for i in range(len(signals)):
            if(diag.diag == "CEC" or diag.diag == "RMD" or it >= 1):
                sys_dev_data[-1].append(0.0)
                std_dev_data[-1].append(0.0)
            if(diag.diag != "CEC" and diag.diag != "RMD"):

                if(it == 0):  # Get the offset on the first got
                    if(diag.name == "ECN" or diag.name == "ECO"):
                        temp_sig = (signals[i][it0:it1] - ECEI_offset_spls[ich](new_times[it])) / \
                                               ECEI_norm_spls[ich](new_times[it])
                    else:
                        temp_sig = signals[i][it0:it1]
                    sig, std_dev = smooth(temp_sig, median, use_std_err)
                    offset[i] = sig
                else:
#                    print(ext_resonances)
                    if(ext_resonances is None):
                        rhop[-1].append(0.0)
                    else:
                        rhop[-1].append(ext_resonances[it - 1][i])
                    if(diag.name == "ECN" or diag.name == "ECO"):
                        temp_sig = (signals[i][it0:it1] - ECEI_offset_spls[ich](new_times[it])) / \
                                               ECEI_norm_spls[ich](new_times[it])
                    else:
                        temp_sig = signals[i][it0:it1]
                    mode_sys_dev = 0.0
                    if(diag.mode_filter):
                        sig, mode_size = remove_mode(diag_time[it0:it1], temp_sig, harmonics=diag.mode_harmonics, mode_width=diag.mode_width, low_freq=diag.freq_cut_off)[0]
                        mode_sys_dev = mode_size
                    else:
                        sig = np.copy(temp_sig)
                    sig, std_dev = smooth(sig, median, use_std_err)
                    data[-1].append((sig - offset[i]) * calib[i])
                    std_dev_data[-1][-1] += np.sqrt(std_dev_calib[i] ** 2 * \
                                               (sig - offset[i]) ** 2 + \
                                               (std_dev) ** 2 * calib[i] ** 2)
                    if(sys_dev_calib is not None):
                        sys_dev_data[-1][-1] += np.sqrt(sys_dev_calib[i] ** 2 * \
                                               (sig - offset[i]) ** 2 + mode_sys_dev ** 2 * calib[i] ** 2)
            else:
                sig = signals[i][it0:it1] * 1.e-3  # eV -> keV
                if(diag.mode_filter):
                    sig, mode_size = remove_mode(diag_time[it0:it1], sig, harmonics=diag.mode_harmonics, mode_width=diag.mode_width, low_freq=diag.freq_cut_off)[0]
                    sys_dev_data[-1][-1] += mode_size
                sig, std_dev = smooth(sig, median , use_std_err)
                data[-1].append(sig)
                std_dev_data[-1][-1] += np.sqrt(std_dev ** 2 + sig ** 2 / SNR_cali_channel[i] ** 2)  # 7 % systematic calibration error already in SNR
                # 7 percent of systematic calibration error and the statistical error from the calibration as another systematic uncertainty
        if(diag.diag == "CEC" or diag.diag == "RMD" or it >= 1):
            rhop[-1] = np.array(rhop[-1])
            data[-1] = np.array(data[-1])
            std_dev_data[-1] = np.array(std_dev_data[-1])
            sys_dev_data[-1] = np.array(sys_dev_data[-1])
        # print("{0:d}/{1:d} done".format(it, len(times)))
    rhop = np.array(rhop)
    data = np.array(data)
    std_dev_data = np.array(std_dev_data)
    sys_dev_data = np.array(sys_dev_data)
    # print(data.shape)
    if not hasattr(time, "__len__"):
        return [std_dev_data[0], sys_dev_data[0]], [rhop[0], data[0]]
    else:
        return [std_dev_data, sys_dev_data], [rhop, data]

def get_RMC_data_calib(shot, time_window=None, ch_off=[]):
    sf_RMD = dd.shotfile('RMD', shot, "AUGD", 0)
    sf_RMC = dd.shotfile('RMC', shot, "AUGD", 0)
    avail = sf_RMD.getParameter('parms-A', 'AVAILABL', dtype=np.float64).data
    for ch in ch_off:
        avail[ch - 1] = 0.0
    Freq = sf_RMD.getParameter('parms-A', 'f', dtype=np.float64).data
    SortIndex = np.argsort(Freq)
    SortIndex = SortIndex[avail[SortIndex] == 1]
    Multi00 = np.concatenate((sf_RMD.getParameter('eCAL-A1', 'MULTIA00', dtype=np.float64).data, \
                                  sf_RMD.getParameter('eCAL-A2', 'MULTIA00', dtype=np.float64).data), axis=0)[SortIndex]
    Shift00 = np.concatenate((sf_RMD.getParameter('eCAL-A1', 'SHIFTB00', dtype=np.float64).data, \
                                  sf_RMD.getParameter('eCAL-A2', 'SHIFTB00', dtype=np.float64).data), axis=0)[SortIndex]
    if(time_window is None):
        Trad = np.concatenate((sf_RMC('Trad-A1', dtype=np.float64).data, \
                             sf_RMC('Trad-A2', dtype=np.float64).data), axis=1)[:, SortIndex]
        time = sf_RMC.getTimeBase('Trad-A1')
    else:
        Trad = np.concatenate((sf_RMC('Trad-A1', tBegin=time_window[0], \
                                    tEnd=time_window[1], dtype=np.float64).data, \
                             sf_RMC('Trad-A2', tBegin=time_window[0], tEnd=time_window[1], \
                                dtype=np.float64).data), axis=1)[:, SortIndex]
        time = sf_RMC.getTimeBase('Trad-A1', tBegin=time_window[0], tEnd=time_window[1])
    Trad *= Multi00
    Trad += Shift00
    return time, Freq[SortIndex], Trad

def get_CTA_no_pinswitch(shot, diag, exp, ed, ch_in=None, t_shift_back=175.e-6, t_shift_fwd=100.e-6):
    # t_shift_back = 175.e-6  time window where the pin switch opens after being closed
    # t_shift_fwd = 100.e-6  time lag, where the pin switch is already closed but the pin-switch signal is not yet in closed state
    try:
        diag_shotfile = dd.shotfile(diag, int(shot), experiment=exp, edition=ed)
    except Exception as e:
        print("Could not open shotfile for ", exp, " ", diag, " ", str(ed))
        print("reason", e)
        raise dd.PyddError
    pin_switch = diag_shotfile.getSignal("ch52")  # 52 shows pin-switch data in CTA
    t = diag_shotfile.getTimeBase("ch52")
    t = t - 4.1e-4
    t1_offset = np.argmin(np.abs(t - 0.005))
    t2_offset = np.argmin(np.abs(t - 0.015))
    offset = smooth(pin_switch[t1_offset:t2_offset], True)[0]
    pin_switch -= offset
    threshhold = 0.06
    sig_list = []
    filtered_sig_list = []
    index_shift_back = np.argmin(np.abs(t - t_shift_back)) - np.argmin(np.abs(t))
    index_shift_fwd = np.argmin(np.abs(t - t_shift_fwd)) - np.argmin(np.abs(t))
    t_mask = np.zeros(len(t), dtype=bool)
    t_mask[0:index_shift_back] = True
    t_mask[len(t) - 1 - index_shift_fwd:len(t)] = True
    print("Indentifying time points with Pin switch attenuation!")
    for i in range(index_shift_back, len(t_mask) - index_shift_fwd):
        if(np.all(pin_switch[i - index_shift_back:i + index_shift_fwd] < threshhold)):
            t_mask[i] = True
    edges = np.where(t_mask[1:len(t_mask)] != t_mask[0:len(t_mask) - 1])[0]
    print("Found " + str(len(edges)) + " Pin switch events")
    # plt.plot(t, pin_switch)
    # plt.plot(t[index_shift:len(t)], np.abs(pin_switch[index_shift:len(t)]) + np.abs(pin_switch[0:len(t) - index_shift]), '+')
    # plt.plot(t[t_mask], pin_switch[t_mask], '+')
    # plt.show()
    masked_time = t[t_mask]
    if(ch_in is None):
        if(diag == "CTA"):
            ch_list = range(50)
        else:
            ch_list = range(42)
    else:
        ch_list = [ch_in - 1]
    for ch in ch_list:
        print("Fixing channel " + str(ch + 1))
        if(ch + 1 < 10):
            ch_str = "ch" + "{0:d}     ".format(ch + 1)
        else:
            ch_str = "ch" + "{0:d}    ".format(ch + 1)
        signal = diag_shotfile.getSignal(ch_str)
        offset = smooth(signal[t1_offset:t2_offset], True)[0]
        signal -= offset
        signal = np.array(np.split(signal, edges))
        for i in range(len(edges)):
            signal[i] = medfilt(signal[i], 7)
        signal = np.concatenate(signal)
        sig_list.append(signal)
#        plt.plot(t, signal)
#        plt.plot(t[t_mask], signal[t_mask], "+")
#        plt.show()
        sig_int = interp1d(masked_time, signal[t_mask])
#        i_last = 0
#        for i in range(len(t_mask)):
#            if(t_mask[i]):
#                i_last = i
#            else:
#                i_next = np.where(t_mask[i:len(t_mask)])[0][0]
#                signal[i] =
        filtered_sig_list.append(sig_int(t))
    return t, sig_list, filtered_sig_list


def filter_CTA(shot, time, diag, exp, ed):
    try:
        diag_shotfile = dd.shotfile(diag, int(shot), experiment=exp, edition=ed)
    except Exception as e:
        print("Could not open shotfile for ", exp, " ", diag, " ", str(ed))
        print("reason", e)
        raise dd.PyddError
    signal = diag_shotfile.getSignal(\
                      "ch52", tBegin=0, tEnd=np.max(time))  # 52 shows pin-switch data in CTA
    try:
        t_CTA = diag_shotfile.getTimeBase("ch52")
        t_CTA = t_CTA - 4.1e-4
    except dd.PyddError:
        steps = len(diag_shotfile.getSignal("ch52"))
        t_CTA = (np.linspace(0.0, 10.0, steps) - 4.1e-4) * 0.8999816246417855
        # correction factor for the corrupted time base
        # Obtained by correlation off the pin switch and ECRH power
        print('Warning time base incorrect - adjusting time base')
    offset, scatter = smooth(signal[np.argmin(np.abs(t_CTA - 0.01)): np.argmin(np.abs(t_CTA - 0.02))], True)
    print("Pin switch att", offset)
    print("Pin switch scatter", scatter)
    signal = signal - offset
    idx = []
    bad_idx = []
    threshold = 0.06
    t_shift_back = 175.e-6
    t_shift_fwd = 100.e-6
    index_shift_back = np.argmin(np.abs(t_CTA - t_shift_back)) - np.argmin(np.abs(t_CTA))
    index_shift_fwd = np.argmin(np.abs(t_CTA - t_shift_fwd)) - np.argmin(np.abs(t_CTA))
    for t_index in range(len(time)):
        t_CTA_closest = np.argmin(np.abs(time[t_index] - t_CTA))
        t_1 = t_CTA_closest - index_shift_back
        t_2 = t_CTA_closest + index_shift_fwd
        if(t_1 < 0):
            t_1 = 0
        if(t_2 > len(t_CTA)):
            t_2 = len(t_CTA)
        if(np.any(signal[t_1:t_2] > threshold)):
            bad_idx.append(t_index)
        else:
            idx.append(t_index)
    print("Elimated a total of", len(bad_idx), " timepoints with pin-switch attenuation")
    # print("Example times", time[np.array(bad_idx)][::100])
    return np.array(idx)
# filter_CTA(33040, np.linspace(0.0, 7.0, 900), "CTA", "AUGD", 0)

def filter_ECRH(shot, time, diag, exp, ed):
    try:
        diag_shotfile = dd.shotfile("ECS", int(shot), experiment=exp, edition=ed)
    except Exception as e:
        print("Could not open shotfile for ", exp, " ", diag, " ", str(ed))
        print("reason", e)
        return [], []
    signal = diag_shotfile.getSignal(\
                      "PECRH", tBegin=0, tEnd=np.max(time))
    t_ECRH = diag_shotfile.getTimeBase("T-" + diag)
    idx = []
    bad_idx = []
    for i in range(len(time)):
        t = time[i]
        sig = np.mean(signal[np.argmin(np.abs(t_ECRH - (t - 0.0005))):np.argmin(np.abs(t_ECRH - (t + 0.0005)))])
        if(np.abs(sig) < 1.0):
            idx.append(i)
        else:
            bad_idx.append(i)
    print("Elimated a total of", len(bad_idx), " timepoints with ECRH power")
    print("Example times", time[np.array(bad_idx)][::100])
    return np.array(idx)

def get_ECRH_PW(shot, diag, exp, ed):
    try:
        diag_shotfile = dd.shotfile("ECS", int(shot), experiment=exp, edition=ed)
    except Exception as e:
        print("Could not open shotfile for ", exp, " ", diag, " ", str(ed))
        print("reason", e)
        return [], []
    signal = diag_shotfile.getSignal("PECRH")
    t_ECRH = diag_shotfile.getTimeBase("PECRH")
    return t_ECRH, signal

def get_ECE_launch_params(shot, diag):
    CEC = dd.shotfile(diag.diag, int(shot), \
                       experiment=diag.exp, edition=diag.ed)
    try:
        ECE_launch_dict = {}
        ECE_launch_dict["f"] = np.array(CEC.getParameter('parms-A', 'f').data)
        available = np.array(CEC.getParameter('parms-A', 'AVAILABL').data, dtype=int)
        ECE_launch_dict["df"] = np.array(CEC.getParameter('parms-A', 'df').data)
        ECE_launch_dict["waveguide"] = np.zeros(len(ECE_launch_dict["f"]), dtype=int)
        ifgroup = np.array(CEC.getParameter('parms-A', 'IFGROUP').data)
        wg = np.array(CEC.getParameter('METHODS', 'WAVEGUID').data)
        ECE_launch_dict["z_lens"] = float(CEC.getParameter('METHODS', 'ZLENS').data) * 1.e-2  # cm -> m
        for i in range(len(ifgroup)):
            ECE_launch_dict["waveguide"][i] = wg[ifgroup[i] - 1]
        ECE_launch_dict["f"] = ECE_launch_dict["f"][available == 1]
        ECE_launch_dict["df"] = ECE_launch_dict["df"][available == 1]
        ECE_launch_dict["waveguide"] = ECE_launch_dict["waveguide"][available == 1]
    except dd.PyddError:
        print("Failed to read " + diag.diag + " shotfile.")
        print("Is this an old shotfile?")
        raise IOError("Shofile read failed")
    return ECE_launch_dict


def get_freqs(shot, diag):
    if(diag.name == "IEC"):
        f = []
        dfreq_IEC = 3.0e9
        for i in range(6):
            f.append(132.5e9 + i * dfreq_IEC)
        f = np.array(f)
    elif(diag.name == "CTC"):
        if(hasattr(diag, "base_freq_140")):
            if(diag.base_freq_140):
                f = [137.0000, 137.6500, 138.0750, 138.3750, 138.5700, \
                     138.6600, 138.7400, 138.8200, 138.9000, 138.9800, \
                     139.0600, 139.1400, 139.2200, 139.3000, 139.3800, \
                     139.4600, 139.5400, 139.6200, 139.7000, 139.7800, \
                     139.8600, 139.9400, 140.0200, 140.1000, 140.1800, \
                     140.2600, 140.3400, 140.4200, 140.5000, 140.5800, \
                     140.6600, 140.7400, 140.8200, 140.9000, 140.9800, \
                     141.0600, 141.1400, 141.2800, 141.5300, 141.8800, 142.3550, 143.0000]
            else:
                f = [102., 102.65 , 103.075, 103.375, 103.57 , 103.66, \
                     103.74 , 103.82 , 103.9  , 103.98 , 104.06 , 104.14 , \
                     104.22 , 104.3  , 104.38 , 104.46 , 104.54 , 104.62 , \
                     104.7  , 104.78 , 104.86 , 104.94 , 105.02 , 105.1  , \
                     105.18 , 105.26 , 105.34 , 105.42 , 105.5  , 105.58 , \
                     105.66 , 105.74 , 105.82 , 105.9  , 105.98 , 106.06 , \
                     106.14 , 106.28 , 106.53 , 106.88 , 107.355, 108.   ]

        else:
            f = [137.0000, 137.6500, 138.0750, 138.3750, 138.5700, \
                 138.6600, 138.7400, 138.8200, 138.9000, 138.9800, \
                 139.0600, 139.1400, 139.2200, 139.3000, 139.3800, \
                 139.4600, 139.5400, 139.6200, 139.7000, 139.7800, \
                 139.8600, 139.9400, 140.0200, 140.1000, 140.1800, \
                 140.2600, 140.3400, 140.4200, 140.5000, 140.5800, \
                 140.6600, 140.7400, 140.8200, 140.9000, 140.9800, \
                 141.0600, 141.1400, 141.2800, 141.5300, 141.8800, 142.3550, 143.0000]
        f = np.array(f) * 1.e9
    elif(diag.name == "CTA"):
        f = np.array([ 135.57, 136.32, 136.82, 137.32, 137.82, 138.12, 138.22, \
            138.32, 138.42, 138.52, 138.62, 138.72, 138.82, 138.92, \
            139.02, 139.12, 139.22, 139.32, 139.42, 139.52, 139.62, \
            139.72, 139.82, 139.92, 140.02, 140.12, 140.22, 140.32, \
            140.42, 140.52, 140.62, 140.72, 140.82, 140.92, 141.02, \
            141.12, 141.22, 141.32, 141.42, 141.52, 141.62, 141.72, \
            141.82, 141.92, 142.02, 142.32, 142.82, 143.32, 143.82, \
            144.57]) * 1.e9
    elif(diag.name == "ECE"):
        CEC = dd.shotfile(diag.diag, int(shot), \
                           experiment=diag.exp, edition=diag.ed)
        f = np.array(CEC.getParameter('parms-A', 'f').data)
        f = f[np.array(CEC.getParameter('parms-A', 'AVAILABL').data, dtype=int) == 1]
    elif(diag.name == "ECN" or diag.name == "ECO"):
        ECI = dd.shotfile(diag.Rz_diag, int(shot), \
                           experiment=diag.Rz_exp, edition=diag.Rz_ed)
        x = np.array(ECI.getParameter('BEAMS', 'x').data)
        freq_ECI_in = np.array(ECI.getParameter('PAR', 'freq').data) * 1.e9
        f = []
        for i in range(len(x)):
            for j in range(len(freq_ECI_in)):
                f.append(freq_ECI_in[j])
        f = np.array(f)
    return f

def get_ECI_launch(diag, shot):
    try:
        ECI = dd.shotfile(diag.Rz_diag, int(shot), \
                       experiment=diag.Rz_exp, edition=diag.Rz_ed)
    except Exception as e:
        print(e)
        print("Error when trying to read diag " + diag.Rz_diag + \
              " exp. " + diag.Rz_exp + " ed. " + str(diag.Rz_ed))
        return None
    ECEI_data = {}
    ECI_launch_dict = {}
    for key in ['freq', 'x', "y", "z", "tor_ang", "pol_ang", "dist_foc", "w"]:
    # Load shotfile data
        if(key is not "freq"):
            ECEI_data[key] = np.array(ECI.getParameter('BEAMS', key).data)
        else:
            ECEI_data[key] = np.array(ECI.getParameter('PAR', key).data) * 1.e9
    for key in ['freq', 'x', "y", "z", "tor_ang", "pol_ang", "dist_foc", "w"]:
    # Store the launch data in a one dimensional array with one entry for each channel (i.e. len = len(freq) * len(x)
        ECI_launch_dict[key] = []
        for i_LOS in range(len(ECEI_data["x"])):
            for i_FREQ in range(len(enumerate(ECEI_data["freq"]))):
                if(key is "freq"):
                    ECI_launch_dict[key].append(ECEI_data[key][i_FREQ])
                else:
                    ECI_launch_dict[key].append(ECEI_data[key][i_LOS])
        ECI_launch_dict[key] = np.array(ECI_launch_dict[key])
    return ECI_launch_dict

def get_shot_heating(shot):
    data = []
    try:
        ECS = dd.shotfile('ECS', int(shot))
        signal = ECS.getSignal(\
                      "PECRH") * 1.e-6
        t = ECS.getTimeBase("PECRH")
        data.append([t, signal])
    except:
        print("No ECRH shot file for current shot")
        print("Setting ECRH power to zero")
        t = np.linspace(0.0, 10.0, 10000)
        signal = np.zeros(len(t))
        data.append([t, signal])
    try:
        NIS = dd.shotfile('NIS', int(shot))
        signal = NIS.getSignal(\
                      "PNI") * 1.e-6
        t = NIS.getTimeBase("PNI")
        data.append([t, signal ])
    except:
        print("No NBI shot file for current shot")
        print("Setting NBI power to zero")
        t = np.linspace(0.0, 10.0, 10000)
        signal = np.zeros(len(t))
        data.append([t, signal])
    try:
        ICP = dd.shotfile('ICP', int(shot))
        signal = ICP.getSignal(\
                      "Picr") * 1.e-6
        t = ICP.getTimeBase("Picr")
        data.append([t, signal ])
    except:
        print("No ICRH shot file for current shot")
        print("Setting ICRH power to zero")
        t = np.linspace(0.0, 10.0, 10000)
        signal = np.zeros(len(t))
        data.append([t, signal])
    return data

def get_plasma_current(shot):
    FPC = dd.shotfile('FPC', int(shot))
    signal = FPC.getSignal("IpiFP") * 1.e-6
    t = FPC.getTimeBase("IpiFP")
    return t, signal

def get_z_mag(shot):
    IDG = dd.shotfile('IDG', int(shot))
    signal = IDG.getSignal("Zmag")
    t = IDG.getTimeBase("Zmag")
    return t, signal

def get_NPA_data(shot):
    NFL = dd.shotfile('NFL', int(shot))
    signal = NFL.getSignalGroup(\
                      "D")
    t = NFL.getTimeBase("D")
    t = t[0:len(t) - len(t) % 10]  # cut off a bit to make the array dividedable by 10
    print(len(t))
    signal = np.mean(signal.T[4][0:len(t) - len(t) % 10].reshape((len(t) / 10, 10)), axis=1)
    t = np.mean(t.reshape((len(t) / 10, 10)), axis=1)  # average over 10 indidual points
    return [t, signal]

def get_Thomson_data(shot, times, diag, Te=False, ne=False, edge=False, core=False, EQ_exp='AUGD', EQ_diag="EQH", EQ_ed=0, smoothing=None):
    scalar_times = False
    if(smoothing is not None):
        t_smooth = smoothing
    else:
        t_smooth = diag.t_smooth
    if(np.isscalar(times)):
        scalar_times = True
        times = np.array([times])
    VTA_shotfile = dd.shotfile(diag.diag, int(shot), experiment=diag.exp, edition=diag.ed)
    if(edge and core):
        print("Please select either core or edge and not both")
        return None, None
    elif(edge):
        t_diag = VTA_shotfile.getTimeBase("R_edge")
        r = VTA_shotfile.getSignal("R_edge")
        z = VTA_shotfile.getSignal("Z_edge")
        sig_end = "_e"
    elif(core):
        t_diag = VTA_shotfile.getTimeBase("R_core")
        r = VTA_shotfile.getSignal("R_core")
        z = VTA_shotfile.getSignal("Z_core")
        sig_end = "_c"
    else:
        print("Neither core nor edge selected - returning")
        return None, None
    R_spline = UnivariateSpline(t_diag, r)
    R = np.array(np.tile(R_spline(times), len(z)))
    R = R.T
    R = R.reshape((len(times)), len(z))
    z = np.tile(z, len(times))
    z = z.T
    z = z.reshape((len(times), R.shape[1]))
    rhop = []
    signals = []
    std_dev_signals = []
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    if(Te and ne):
        print("Please select either ne or Te not both")
        return None, None
    elif(Te):
        sig = VTA_shotfile.getSignalGroup("Te" + sig_end)
        sig_error = VTA_shotfile.getSignalGroup("SigTe" + sig_end)
    elif(ne):
        sig = VTA_shotfile.getSignalGroup("Ne" + sig_end)
        sig_error = VTA_shotfile.getSignalGroup("SigNe" + sig_end)
    else:
        print("Neither Te nor ne selected - returning")
        return None, None
    i = 0
    for time in times:
        t1 = np.argmin(np.abs(t_diag - time + 0.5 * t_smooth))
        t2 = np.argmin(np.abs(t_diag - time - 0.5 * t_smooth))
        if(t2 == t1):
            t2 += 1
        signals.append(np.mean(sig[t1:t2], axis=0))
        std_dev_signals.append(np.sqrt((np.mean(sig_error[t1:t2], axis=0) / np.sqrt(t2 - t1)) ** 2 + (np.std(sig[t1:t2], axis=0)) ** 2))
        rhop.append(EQ_obj.map_Rz_to_rhop(time, R[i], z[i]))
        i += 1
    if(scalar_times):
        times = times[0]
        return std_dev_signals[0], [rhop[0], signals[0]]
    else:
        return np.array(std_dev_signals), [np.array(rhop), np.array(signals)]

def get_cold_resonances_S_ECE(shot, time, diag_name, R_min, R_max, z_min, z_max, B_spline, ch_no, exp="AUGD", diag="None", ed=0):
    # B_spline is expected to be a UnivariateSpline of the total magnetic field
    if(diag_name == "CTC" or diag_name == "IEC"):
        beamline = 5
    elif(diag_name == "CTA"):
        beamline = 6
    else:
        print("Unknown Diag name: {0:s}".format(diag_name))
        raise ValueError("Unknown Diag name: {0:s}".format(diag_name))
    diagnostic = Diag(diag_name, exp, diag, ed)
    f = get_freqs(shot, diagnostic) * 100
    if(ch_no > len(f)):
        print("For {0:s} there are only {1:d} channels < ch_no = {2:d} available".format(diag_name, len(f), ch_no))
        raise ValueError("For {0:s} there are only {1:d} channels < ch_no = {2:d} available".format(diag_name, len(f), ch_no))
    gy = get_ECRH_viewing_angles(shot, beamline, True)
    x = np.zeros(3)
    x[0] = gy.x
    x[1] = gy.y
    x[2] = gy.z
    t1 = np.argmin(np.abs(gy.time - time + 0.005))
    t2 = np.argmin(np.abs(gy.time - time - 0.005))
    if(t1 == t2):
        t2 += 1
    phi_tor = -np.mean(gy.phi_tor[t1:t2]) / 180.0 * np.pi
    theta_pol = -np.mean(gy.theta_pol[t1:t2]) / 180.0 * np.pi
    k = np.array([1.0, np.arctan2(-x[1], -x[0]) + phi_tor, np.pi / 2.e0 + theta_pol])
    k_x = np.array([np.cos(k[1]) * np.sin(k[2]), np.sin(k[1]) * np.sin(k[2]), np.cos(k[2])])
    plt.show()
    los = []
    los.append(x)
    s = [0.0]
    los_finished = False
    while(not los_finished):
        next_step = los[-1] + k_x / 10.0
        R_ray = np.sqrt(next_step[0] ** 2 + next_step[1] ** 2)
        if(R_ray > R_max or R_ray < R_min or next_step[2] > z_max or next_step[2] < z_min):
            los_finished = True
        else:
            los.append(next_step)
            s.append(s[-1] + 0.1)
    los = np.array(los)
    s = np.array(s)
    R_ray = np.sqrt(los.T[0] ** 2 + los.T[1] ** 2)
    plt.plot(R_ray, los.T[2])
    f_cyc_2 = B_spline(R_ray , los.T[2], grid=False) * cnst.e / cnst.m_e / cnst.pi
    R_spl = InterpolatedUnivariateSpline(s, R_ray, k=1)
    z_spl = InterpolatedUnivariateSpline(s, los.T[2], k=1)
    root = InterpolatedUnivariateSpline(s, f_cyc_2 - f[ch_no - 1]).roots()
    try:
        return (R_spl(root[0]), z_spl(root[0]))
    except IndexError:
        print("No resonance found along ray!")
        return np.array([None, None])

def test_resonance():
    shot = 33585
    time = 1.68
    eq_exp = "AUGD"
    eq_diag = "EQH"  # "IDE"
    eq_ed = 0
    bt_vac_correction = 1.005
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    EQ_t = EQ_obj.GetSlice(time)
    B_spl = RectBivariateSpline(EQ_t.R, EQ_t.z, np.sqrt(EQ_t.Br ** 2 + EQ_t.Bt ** 2 + EQ_t.Bz ** 2))
    R, z = get_cold_resonances_S_ECE(shot, time, "IEC", np.min(EQ_t.R), np.max(EQ_t.R), np.min(EQ_t.z), np.max(EQ_t.z), B_spl, 4)
    plt.plot(R, z, "+")
    plt.show()

# test_resonance()


def get_CECE_launch(shot, angpol):
    shot_openings = np.array([27405, 28553, 30150, 31777])
    poly_params = [[0.00555281, -0.130428, -3.83216], \
                     [0, -0.254476, -2.87293], \
                     [0.0017159, -0.2141350, -3.42321], \
                     [0.003006190, -0.180083, -3.51089]]
    ind = np.where(shot_openings <= shot)[0][::-1][0]
    a = poly_params[ind]
    # determine toroIDAl angle (depends on which campaign due to
    # recalibration of system during openings)
    print('shot requested / first shot of corresponding campaign: ', shot, shot_openings[ind])
    print('=> polygon parameters: ', a)
    ret = a[0] * angpol ** 2 + a[1] * angpol + a[2]
    return ret

def load_IDA_ECE_residues(shot, time, exp, ed):
    IDA = dd.shotfile("IDA", pulseNumber=int(shot), experiment=exp, edition=ed)
    IDA_ECE_res_mat = IDA.getSignalGroup(\
                                         "ece_resi", dtype=np.double)
    IDA_ECE_rhop_mat = IDA.getSignalGroup(\
                                          "ece_rhop", dtype=np.double)
    IDA_ECE_dat_mat = IDA.getSignalGroup(\
                                         "ece_dat", dtype=np.double)
    IDA_ECE_unc_mat = IDA.getSignalGroup(\
                                         "ece_unc", dtype=np.double)
    IDA_ECE_mod_mat = IDA.getSignalGroup(\
                                         "ece_mod", dtype=np.double)
    IDA_time = IDA.getTimeBase(\
                    "time", dtype=np.double)
    if(len(IDA_time) == 1):
        IDA_ECE_rhop_mat = np.array([IDA_ECE_rhop_mat])
        IDA_ECE_res_mat = np.array([IDA_ECE_res_mat])
        IDA_ECE_dat_mat = np.array([IDA_ECE_dat_mat])
        IDA_ECE_unc_mat = np.array([IDA_ECE_unc_mat])
        IDA_ECE_mod_mat = np.array([IDA_ECE_mod_mat])
    index = np.argmin(np.abs(IDA_time - time))
    return IDA_ECE_rhop_mat[index], np.mean(IDA_ECE_res_mat[index], axis=0), np.mean(IDA_ECE_dat_mat[index], axis=0), np.mean(IDA_ECE_unc_mat[index], axis=0), IDA_ECE_mod_mat[index]

def get_last_edition_number(shot, exp, diag):
    try:
        shotfile = dd.shotfile(diag, pulseNumber=shot, experiment=exp)
        return shotfile.edition
    except:
        return 0


def load_IDA_data(shot, timepoints=None, exp="AUGD", ed=0, double_entries_allowed=False):
    IDA_dict = { }
    IDA = dd.shotfile("IDA", pulseNumber=int(shot), experiment=exp, edition=ed)
    IDA_dict["ed"] = IDA.edition
    IDA_time = IDA.getTimeBase(\
                    "time", dtype=np.double)
    IDA_Te_mat = IDA.getSignalGroup(\
                    "Te", dtype=np.double)
    IDA_Te_low_mat = IDA.getSignalGroup(\
                    "Te_lo", dtype=np.double)
    IDA_Te_up_mat = IDA.getSignalGroup(\
                    "Te_up", dtype=np.double)
    IDA_ne_mat = IDA.getSignalGroup(\
                    "ne", dtype=np.double)
    IDA_rhop_mat = IDA.getAreaBase(\
                    "rhop".encode("utf-8"), dtype=np.double).data
    rhot_available=True
    try:
        IDA_rhot_mat = IDA.getSignalGroup(\
                                          "rhot", dtype=np.double)
    except Exception as e:
        rhot_available=False
        print("No rho toroIDAl profile in IDA shotfile")
        print("Getting rho tor later")
    # This parameter is not used according to Rainer Fischer
    # IDA_ne_rhop_scal_mat = IDA.getSignal(\
    #                "ecenrpsc", dtype=np.double)
    try:
        if(sys.version_info.major == 3):
            raise Exception("Cannot load IDA ECE data in python 3")
        IDA_ECE_rhop_mat = IDA.getSignalGroup("ece_rhop", dtype=np.double)
        IDA_ECE_dat_mat = IDA.getSignalGroup("ece_dat", dtype=np.double)
        IDA_ECE_unc_mat = IDA.getSignalGroup("ece_unc", dtype=np.double)
        IDA_ECE_mod_mat = IDA.getSignalGroup("ece_mod", dtype=np.double)
        IDA_ECE_data = True
    except Exception as e:
        print("Could not find any ECE data in the IDA shotfile")
        print("Reason", e)
        IDA_ECE_data = False
    if(len(IDA_time) == 1):
        IDA_Te_mat = np.atleast_2d(IDA_Te_mat)
        IDA_Te_low_mat = np.atleast_2d(IDA_Te_low_mat)
        IDA_Te_up_mat = np.atleast_2d(IDA_Te_up_mat)
        IDA_ne_mat = np.atleast_2d(IDA_ne_mat)
        # IDA_rhop_mat = np.array([IDA_rhop_mat])
        if(IDA_ECE_data):
            IDA_ECE_rhop_mat = np.atleast_2d(IDA_ECE_rhop_mat)
            if(np.ndim(IDA_ECE_dat_mat)):
                IDA_ECE_dat_mat = np.expand_dims(IDA_ECE_dat_mat,0)
                IDA_ECE_unc_mat = np.expand_dims(IDA_ECE_unc_mat,0)
            IDA_ECE_mod_mat = np.atleast_2d(IDA_ECE_mod_mat)
    if(IDA_ECE_data):
        if(np.ndim(IDA_ECE_dat_mat) == 3):
            IDA_ECE_dat_mat=np.swapaxes(IDA_ECE_dat_mat, 2, 1)
            IDA_ECE_unc_mat=np.swapaxes(IDA_ECE_unc_mat, 2, 1)
            IDA_ECE_dat_rhop_mat = np.zeros(IDA_ECE_dat_mat.shape)
            for i in range(len(IDA_ECE_dat_mat.T)):
                IDA_ECE_dat_rhop_mat[:,:,i] = IDA_ECE_rhop_mat
    Te_mat = []
    Te_up_mat = []
    Te_low_mat = []
    ne_mat = []
    rhop_mat = []
    if(rhot_available):
        rhot_mat = []
    ne_rhop_scale_mat = []
    ECE_rhop_mat = []
    ECE_dat_rhop_mat = []
    ECE_dat_mat = []
    ECE_unc_mat = []
    ECE_mod_mat = []
    IDA_dict["time"] = []
    if(timepoints is None):
        for index in range(len(IDA_time)):
            if(IDA_time[index] not in IDA_dict["time"]):  # No double entries unless specifically requested!
                IDA_dict["time"].append(IDA_time[index])
                Te_mat.append(IDA_Te_mat[index])
                Te_up_mat.append(IDA_Te_up_mat[index])
                Te_low_mat.append(IDA_Te_low_mat[index])
                ne_mat.append(IDA_ne_mat[index])
                rhop_mat.append(IDA_rhop_mat[index])
                if(rhot_available):
                    rhot_mat.append(IDA_rhot_mat[index])
                if(IDA_ECE_data):
                # ne_rhop_scale_mat.append(IDA_ne_rhop_scal_mat[index])
                    ECE_rhop_mat.append(IDA_ECE_rhop_mat[index])
                    ECE_dat_rhop_mat.append(IDA_ECE_dat_rhop_mat[index])
                    ECE_dat_mat.append(IDA_ECE_dat_mat[index])
                    ECE_unc_mat.append(IDA_ECE_unc_mat[index])
                    ECE_mod_mat.append(IDA_ECE_mod_mat[index])
    else:
        for t in timepoints:  # Finds closest - NO interpolation
            index = np.argmin(np.abs(IDA_time - t))
            if(IDA_time[index] not in IDA_dict["time"]  or double_entries_allowed):  # No double entries !
                # print(index, len(IDA_time), IDA_time[index], t)
                IDA_dict["time"].append(IDA_time[index])
                Te_mat.append(IDA_Te_mat[index])
                Te_up_mat.append(IDA_Te_up_mat[index])
                Te_low_mat.append(IDA_Te_low_mat[index])
                ne_mat.append(IDA_ne_mat[index])
                rhop_mat.append(IDA_rhop_mat[index])
                if(rhot_available):
                    rhot_mat.append(IDA_rhot_mat[index])
                # print(rhop_mat[-1], Te_mat[-1])
                # ne_rhop_scale_mat.append(IDA_ne_rhop_scal_mat[index])
                if(IDA_ECE_data):
                    ECE_rhop_mat.append(IDA_ECE_rhop_mat[index])
                    ECE_dat_rhop_mat.append(IDA_ECE_dat_rhop_mat[index])
                    ECE_dat_mat.append(IDA_ECE_dat_mat[index])
                    ECE_unc_mat.append(IDA_ECE_unc_mat[index])
                    ECE_mod_mat.append(IDA_ECE_mod_mat[index])
    IDA_dict["time"] = np.array(IDA_dict["time"])
    Te_mat = np.array(Te_mat)
    Te_up_mat = np.array(Te_up_mat)
    Te_low_mat = np.array(Te_low_mat)
    ne_mat = np.array(ne_mat)
    rhop_mat = np.array(rhop_mat)
    if(rhot_available):
        rhot_mat = np.array(rhot_mat)
    ne_rhop_scale_mat = np.array(ne_rhop_scale_mat)
    ne_rhop_scale_mat[ne_rhop_scale_mat == 0] = 1
    if(IDA_ECE_data):
        ECE_rhop_mat = np.array(ECE_rhop_mat)
        ECE_dat_rhop_mat = np.array(ECE_dat_rhop_mat)
        ECE_dat_rhop_mat = np.reshape(ECE_dat_rhop_mat,(ECE_dat_rhop_mat.shape[0],ECE_dat_rhop_mat.shape[1] * ECE_dat_rhop_mat.shape[2]))
        ECE_dat_mat = np.array(ECE_dat_mat)
        ECE_dat_mat = np.reshape(ECE_dat_mat,(ECE_dat_mat.shape[0],ECE_dat_mat.shape[1] * ECE_dat_mat.shape[2]))
        ECE_unc_mat = np.array(ECE_unc_mat)
        ECE_unc_mat = np.reshape(ECE_unc_mat,(ECE_unc_mat.shape[0],ECE_unc_mat.shape[1] * ECE_unc_mat.shape[2]))
        ECE_mod_mat = np.array(ECE_mod_mat)
    IDA_dict["Te"] = Te_mat
    IDA_dict["Te_up"] = Te_up_mat
    IDA_dict["Te_low"] = Te_low_mat
    IDA_dict["ne"] = ne_mat
    IDA_dict["rhop_prof"] = rhop_mat
    if(rhot_available):
        IDA_dict["rhot_prof"] = rhot_mat
    IDA_dict["prof_reference"] = "rhop_prof"
    try:
        IDA_dict["ne_rhop_scale"] = IDA.getSignal("ecenrpsc", dtype=np.double)
    except:
        IDA_dict["ne_rhop_scale"] = np.zeros(len(IDA_dict["time"]))
        IDA_dict["ne_rhop_scale"][:] = 1.0  # ne_rhop_scale_mat
    if(IDA_ECE_data):
        IDA_dict["ECE_rhop"] = ECE_rhop_mat
        IDA_dict["ECE_dat_rhop"] = ECE_dat_rhop_mat
        IDA_dict["ECE_dat"] = ECE_dat_mat
        IDA_dict["ECE_unc"] = ECE_unc_mat
        IDA_dict["ECE_mod"] = ECE_mod_mat
    else:
        IDA_dict["ECE_rhop"] = []
        IDA_dict["ECE_dat_rhop"] = []
        IDA_dict["ECE_dat"] = []
        IDA_dict["ECE_unc"] = []
        IDA_dict["ECE_mod"] = []
    try:
        IDA_dict["raytrace"] = bool(IDA.getParameter('ece_par', 'raytrace').data)
    except:
        IDA_dict["raytrace"] = False
    try:
        try:
            IDA_dict["RwallX"] = IDA.getParameter('ece_par', 'reflec').data
        except:
            try:
                IDA_dict["RwallX"] = np.mean(IDA.getSignal('ece_refl'))
            except:
                IDA_dict["RwallX"] = 0.92
        try:
            IDA_dict["RwallO"] = np.mean(IDA.getSignal('ece_reflO'))
        except:
            IDA_dict["RwallO"] = 0.92

    except:
        print("Warning!: Old IDA file detected")
        print("Warning!: Te might have been obtained without accurate forward modeling")
        try:
            IDA_dict["RwallX"] = np.mean(IDA.getSignal('ece_reflX'))
        except:
            try:
                IDA_dict["RwallX"] = np.mean(IDA.getSignal('ece_refl'))
            except:
                IDA_dict["RwallX"] = 0.92
        try:
            IDA_dict["RwallO"] = np.mean(IDA.getSignal('ece_reflO'))
        except:
            IDA_dict["RwallO"] = 0.92
    try:
        IDA_dict["Bt_vac_scale"] = IDA.getParameter('ece_par', 'Btf_corr').data
    except:
        if(shot < 30160):
            IDA_dict["Bt_vac_scale"] = 1.0
        else:
            IDA_dict["Bt_vac_scale"] = 1.005
    try:
        IDA_dict["EQ_diag"] = IDA.getParameter('depends', 'map_diag').data.replace(" ", "")
        IDA_dict["EQ_exp"] = IDA.getParameter('depends', 'map_exp').data.replace(" ", "")
        IDA_dict["EQ_ed"] = IDA.getParameter('depends', 'map_edit').data
    except:
        IDA_dict["EQ_diag"] = "EQH"
        IDA_dict["EQ_exp"] = "AUGD"
        IDA_dict["EQ_ed"] = 0
    IDA_dict["ne_rhop_scale_mean"] = np.mean(IDA_dict["ne_rhop_scale"])
    IDA_dict["eq_data"] = None
    return IDA_dict

def load_IDI_data(shot, timepoints=None, exp="AUGD", ed=0):
    IDI_dict = { }
    IDI = dd.shotfile("IDI", pulseNumber=int(shot), experiment=exp, edition=ed)
    IDI_dict["ed"] = IDI.edition
    IDI_time = IDI.getTimeBase(\
                    "time", dtype=np.double)
    IDI_Ti_mat = IDI.getSignalGroup(\
                    "Ti", dtype=np.double)
    IDI_Ti_unc_mat = IDI.getSignalGroup(\
                    "Ti_unc", dtype=np.double)
    IDI_rhop_mat = IDI.getAreaBase(\
                    "rp_Ti", dtype=np.double).data
    if(len(IDI_time) == 1):
        IDI_Ti_mat = np.array([IDI_Ti_mat])
        IDI_Ti_unc_mat = np.array([IDI_Ti_unc_mat])
    Ti_mat = []
    Ti_unc_mat = []
    ne_mat = []
    rhop_mat = []
    ne_rhop_scale_mat = []
    IDI_dict["time"] = []
    if(timepoints is None):
        for index in range(len(IDI_time)):
            if(IDI_time[index] not in IDI_dict["time"]):  # No double entries unless specifically requested!
                IDI_dict["time"].append(IDI_time[index])
                Ti_mat.append(IDI_Ti_mat[index])
                Ti_unc_mat.append(IDI_Ti_unc_mat[index])
                rhop_mat.append(IDI_rhop_mat[index])
    else:
        for t in timepoints:  # Finds closest - NO inTirpolation
            index = np.argmin(np.abs(IDI_time - t))
            if(IDI_time[index] not in IDI_dict["time"]):  # No double entries !
                # print(index, len(IDI_time), IDI_time[index], t)
                IDI_dict["time"].append(IDI_time[index])
                Ti_mat.append(IDI_Ti_mat[index])
                Ti_unc_mat.append(IDI_Ti_unc_mat[index])
                rhop_mat.append(IDI_rhop_mat[index])
                # print(rhop_mat[-1], Te_mat[-1])
                # ne_rhop_scale_mat.append(IDI_ne_rhop_scal_mat[index])
    IDI_dict["time"] = np.array(IDI_dict["time"])
    Ti_mat = np.array(Ti_mat)
    Ti_unc_mat = np.array(Ti_unc_mat)
    ne_mat = np.array(ne_mat)
    rhop_mat = np.array(rhop_mat)
    ne_rhop_scale_mat = np.array(ne_rhop_scale_mat)
    ne_rhop_scale_mat[ne_rhop_scale_mat == 0] = 1
    IDI_dict["Ti"] = Ti_mat
    IDI_dict["Ti_unc"] = Ti_unc_mat
    IDI_dict["ne"] = ne_mat
    IDI_dict["rhop_prof"] = rhop_mat
    IDI_dict["prof_reference"] = "rhop_prof"
    return IDI_dict

#Deprecated!
# def make_ext_data_for_testing(ext_data_folder, shot, times, eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005, IDA_exp="AUGD", IDA_ed=0):
#     index = 0
#     EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed, bt_vac_correction=bt_vac_correction)
#     plasma_data = load_IDA_data(shot, timepoints=times, exp=IDA_exp, ed=IDA_ed)
#     if(not os.path.isdir(ext_data_folder)):
#         try:
#             os.mkdir(ext_data_folder)
#         except OSError:
#             try:
#                 os.mkdir(ext_data_folder.replace('Ext_data', ''))
#                 os.mkdir(ext_data_folder)
#             except OSError:
#                 print('Please create the parent directory: ' + ext_data_folder.replace(os.sep + 'Ext_data', '').rsplit(os.sep)[0])
#                 return
#     np.savetxt(os.path.join(ext_data_folder, "t"), plasma_data["time"])
#     for time in plasma_data["time"]:
#         EQ_t = EQ_obj.GetSlice(time)
#         np.savetxt(os.path.join(ext_data_folder, "special_points{0:d}".format(index)), np.array([EQ_t.R_ax, EQ_t.Psi_sep]))
#         np.savetxt(os.path.join(ext_data_folder, "R{0:d}".format(index)), EQ_t.R)
#         np.savetxt(os.path.join(ext_data_folder, "z{0:d}".format(index)), EQ_t.z)
#         np.savetxt(os.path.join(ext_data_folder, "Psi{0:d}".format(index)), EQ_t.Psi)
#         np.savetxt(os.path.join(ext_data_folder, "Br{0:d}".format(index)), EQ_t.Br)
#         np.savetxt(os.path.join(ext_data_folder, "Bt{0:d}".format(index)), EQ_t.Bt)
#         # plt.contour(EQ.R, EQ.z, EQ.rhop, levels=np.array([0.1, 1.2]))
#         # cont = plt.contourf(EQ.R, EQ.z, np.arctan(EQ.Bz / EQ.Bt) / np.pi * 180.0, levels=np.linspace(-15, 15.0, 40))
#         # cb = plt.gcf().colorbar(cont, ax=plt.gca(), ticks=[-15, -5, 0, 5, 15])
#         # plt.show()
#         np.savetxt(os.path.join(ext_data_folder, "Bz{0:d}".format(index)), EQ_t.Bz)
#         Te_data = np.array([plasma_data["rhop"][index], plasma_data["Te"][index]]).T  # for coloumn
#         ne_data = np.array([plasma_data["rhop"][index], plasma_data["ne"][index]]).T
#         np.savetxt(os.path.join(ext_data_folder, "Te{0:d}".format(index)), Te_data)
#         np.savetxt(os.path.join(ext_data_folder, "ne{0:d}".format(index)), ne_data)
#         index += 1
#     copyfile('../ECRad_Pylib/ASDEX_Upgrade_vessel.txt', os.path.join(ext_data_folder, "Ext_vessel.bd"))
#     print('External data ready!')

def make_ext_data_for_testing_from_data(ext_data_folder, shot, times, R, z, Br, Bt, Bz, Psi, R_ax, z_ax, Psi_ax, Psi_sep, rhop, ne, Te):
    # SI UNITS!
    if(not os.path.isdir(ext_data_folder)):
        try:
            os.mkdir(ext_data_folder)
        except OSError:
            try:
                os.mkdir(ext_data_folder.replace('Ext_data', ''))
                os.mkdir(ext_data_folder)
            except OSError:
                print('Please create the parent directory: ' + ext_data_folder.replace(os.sep + 'Ext_data', '').rsplit(os.sep)[0])
                return False
    np.savetxt(os.path.join(ext_data_folder, "t"), times)
    index = 0
    for time in times:
        np.savetxt(os.path.join(ext_data_folder, "special_points{0:d}".format(index)), np.array([R_ax, Psi_sep]))
        np.savetxt(os.path.join(ext_data_folder, "R{0:d}".format(index)), R)
        np.savetxt(os.path.join(ext_data_folder, "z{0:d}".format(index)), z)
        np.savetxt(os.path.join(ext_data_folder, "Psi{0:d}".format(index)), Psi)
        np.savetxt(os.path.join(ext_data_folder, "Br{0:d}".format(index)), Br)
        np.savetxt(os.path.join(ext_data_folder, "Bt{0:d}".format(index)), Bt)
        # plt.contour(EQ.R, EQ.z, EQ.rhop, levels=np.array([0.1, 1.2]))
        # cont = plt.contourf(EQ.R, EQ.z, np.arctan(EQ.Bz / EQ.Bt) / np.pi * 180.0, levels=np.linspace(-15, 15.0, 40))
        # cb = plt.gcf().colorbar(cont, ax=plt.gca(), ticks=[-15, -5, 0, 5, 15])
        # plt.show()
        np.savetxt(os.path.join(ext_data_folder, "Bz{0:d}".format(index)), Bz)
        Te_data = np.array([rhop, Te]).T  # for coloumn
        ne_data = np.array([rhop, ne]).T  # for coloumn
        np.savetxt(os.path.join(ext_data_folder, "Te{0:d}".format(index)), Te_data)
        np.savetxt(os.path.join(ext_data_folder, "ne{0:d}".format(index)), ne_data)
        index += 1
    print('External data ready!')
    return True

def make_ext_data_equil_for_testing(ext_data_folder, shot, times, eq_exp="AUGD", eq_diag="EQH", eq_ed=0, \
                                    IDA_exp="AUGD", IDA_ed=0):
    index = 0
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed, bt_vac_correction=bt_vac_correction)
    if(not os.path.isdir(ext_data_folder)):
        try:
            os.mkdir(ext_data_folder)
        except OSError:
            try:
                os.mkdir(ext_data_folder.replace('Ext_data', ''))
                os.mkdir(ext_data_folder)
            except OSError:
                print('Please create the parent directory: ' + ext_data_folder.replace(os.sep + 'Ext_data', '').rsplit(os.sep)[0])
                return
    np.savetxt(os.path.join(ext_data_folder, "t"), times)
    for time in times:
        EQ_t = EQ_obj.GetSlice(time)
        np.savetxt(os.path.join(ext_data_folder, "special_points{0:d}".format(index)), np.array([EQ_t.R_ax, EQ_t.Psi_sep]))
        np.savetxt(os.path.join(ext_data_folder, "R{0:d}".format(index)), EQ_t.R)
        np.savetxt(os.path.join(ext_data_folder, "z{0:d}".format(index)), EQ_t.z)
        np.savetxt(os.path.join(ext_data_folder, "Psi{0:d}".format(index)), EQ_t.Psi)
        np.savetxt(os.path.join(ext_data_folder, "Br{0:d}".format(index)), EQ_t.Br)
        np.savetxt(os.path.join(ext_data_folder, "Bt{0:d}".format(index)), EQ_t.Bt)
        # plt.contour(EQ.R, EQ.z, EQ.rhop, levels=np.array([0.1, 1.2]))
        # cont = plt.contourf(EQ.R, EQ.z, np.arctan(EQ.Bz / EQ.Bt) / np.pi * 180.0, levels=np.linspace(-15, 15.0, 40))
        # cb = plt.gcf().colorbar(cont, ax=plt.gca(), ticks=[-15, -5, 0, 5, 15])
        # plt.show()
        np.savetxt(os.path.join(ext_data_folder, "Bz{0:d}".format(index)), EQ_t.Bz)
        index += 1
    print('External data ready!')

def make_ext_data_for_testing_grids(ext_data_folder, shot, times, eq_exp, eq_diag, eq_ed, \
                                    IDA_exp="AUGD", IDA_ed=0):
    index = 0
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    plasma_data = load_IDA_data(shot, timepoints=times, exp="AUGD", ed=0)
    np.savetxt(os.path.join(ext_data_folder, "t"), plasma_data["time"])
    for time in plasma_data["time"]:
        EQ_t = EQ_obj.GetSlice(time)
        np.savetxt(os.path.join(ext_data_folder, "special_points{0:d}".format(index)), np.array([EQ_t.R_ax, EQ_t.Psi_sep]))
        np.savetxt(os.path.join(ext_data_folder, "R{0:d}".format(index)), EQ_t.R)
        np.savetxt(os.path.join(ext_data_folder, "z{0:d}".format(index)), EQ_t.z)
        np.savetxt(os.path.join(ext_data_folder, "Psi{0:d}".format(index)), EQ_t.Psi)
        np.savetxt(os.path.join(ext_data_folder, "Br{0:d}".format(index)), EQ_t.Br)
        np.savetxt(os.path.join(ext_data_folder, "Bt{0:d}".format(index)), EQ_t.Bt)
        np.savetxt(os.path.join(ext_data_folder, "Bz{0:d}".format(index)), EQ_t.Bz)
        Te_spline = InterpolatedUnivariateSpline(plasma_data["rhop"][index], np.log(plasma_data["Te"][index] + 5.0), k=1)
        ne_spline = InterpolatedUnivariateSpline(plasma_data["rhop"][index], np.log(plasma_data["ne"][index] + 1.e16), k=1)
        R_mat, z_mat = np.meshgrid(EQ_t.R, EQ_t.z)
        rho_spl = RectBivariateSpline(EQ_t.R, EQ_t.z, EQ_t.rhop)
        Te = np.exp(Te_spline(rho_spl(R_mat, z_mat, grid=False))) * \
            (1.e0 - 0.9 * np.exp(-(rho_spl(R_mat, z_mat, grid=False) - 0.90) ** 2 \
            / 0.05 ** 2) * np.sin(np.arcsin(z_mat / np.sqrt(R_mat ** 2 + z_mat ** 2)) * 180.0)) + 1.0
        ne = np.exp(ne_spline(rho_spl(R_mat, z_mat, grid=False))) * \
            (1.e0 - 0.5 * np.exp(-(rho_spl(R_mat, z_mat, grid=False) - 0.90) ** 2 / 0.05 ** 2) * \
            np.sin(np.arcsin(z_mat / np.sqrt(R_mat ** 2 + z_mat ** 2)) * 90)) + 1.e16
        np.savetxt(os.path.join(ext_data_folder, "Te{0:d}".format(index)), Te.T)
        np.savetxt(os.path.join(ext_data_folder, "ne{0:d}".format(index)), ne.T)
        plt.figure()
        plt.contourf(EQ_t.R, EQ_t.z, ne * 1.e-19, levels=np.linspace(0, 8, 30))
        plt.figure()
        plt.contourf(EQ_t.R, EQ_t.z, Te, levels=np.linspace(0, 5000, 30))
        plt.show()
        index += 1

def export_ASDEX_Upgrade_grid(ext_data_folder, shot, times, eq_exp, eq_diag, eq_ed, \
                              IDA_exp="AUGD", IDA_ed=0, offset=False):
    index = 0
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    plasma_data = load_IDA_data(shot, timepoints=times, exp="AUGD", ed=0)
    np.savetxt(os.path.join(ext_data_folder, "t"), plasma_data["time"])
    for time in plasma_data["time"]:
        EQ_t = EQ_obj.GetSlice(time)
        np.savetxt(os.path.join(ext_data_folder, "special_points{0:d}".format(index)), np.array([EQ_t.R_ax, EQ_t.Psi_sep]))
        np.savetxt(os.path.join(ext_data_folder, "R{0:d}".format(index)), EQ_t.R)
        np.savetxt(os.path.join(ext_data_folder, "z{0:d}".format(index)), EQ_t.z)
        np.savetxt(os.path.join(ext_data_folder, "Psi{0:d}".format(index)), EQ_t.Psi)
        np.savetxt(os.path.join(ext_data_folder, "Br{0:d}".format(index)), EQ_t.Br)
        np.savetxt(os.path.join(ext_data_folder, "Bt{0:d}".format(index)), EQ_t.Bt)
        np.savetxt(os.path.join(ext_data_folder, "Bz{0:d}".format(index)), EQ_t.Bz)
        rhop_max = 1.4
        rhop_max_prof = np.max(plasma_data["rhop"][index])
        rhop_ext = np.linspace(rhop_max_prof, rhop_max, 100)
        ne_min = np.min(plasma_data["ne"][index])
        dne_drhop = (plasma_data["ne"][index][-2] - plasma_data["ne"][index][-1]) / (plasma_data["rhop"][index][-2] - plasma_data["rhop"][index][-1])
        ne_ext = ne_min * np.exp((rhop_ext - rhop_max_prof) / ne_min * dne_drhop)
        ne_ext -= np.min(ne_ext) * np.exp(-(rhop_max - rhop_ext))
        if(offset):
            ne_ext += 1.e17
        ne_spline = InterpolatedUnivariateSpline(np.concatenate([plasma_data["rhop"][index], rhop_ext[1:len(rhop_ext)]]), \
                                                 np.concatenate([plasma_data["ne"][index] * 1.e-19, ne_ext[1:len(rhop_ext)] * 1.e-19]), \
                                                 k=1, ext=3)
        plt.plot(np.concatenate([plasma_data["rhop"][index], rhop_ext]), np.concatenate([plasma_data["ne"][index] * 1.e-19, ne_ext * 1.e-19]))
        print(np.min(ne_ext))
        plt.show()
        ne = ne_spline(EQ_t.rhop.flatten()) * 1.e19
        ne = np.reshape(ne, EQ_t.rhop.shape)
        print(np.max(ne), np.min(ne))
        np.savetxt(os.path.join(ext_data_folder, "ne{0:d}".format(index)), ne)
        # Te
        Te_min = np.min(plasma_data["Te"][index])
        dTe_drhop = (plasma_data["Te"][index][-2] - plasma_data["Te"][index][-1]) / (plasma_data["rhop"][index][-2] - plasma_data["rhop"][index][-1])
        Te_ext = Te_min * np.exp((rhop_ext - rhop_max_prof) / Te_min * dTe_drhop)
        Te_ext -= np.min(Te_ext) * np.exp(-(rhop_max - rhop_ext))
        if(offset):
            Te_ext += 2.e-2
        Te_spline = InterpolatedUnivariateSpline(np.concatenate([plasma_data["rhop"][index], rhop_ext[1:len(rhop_ext)]]), \
                                                 np.concatenate([plasma_data["Te"][index], Te_ext[1:len(rhop_ext)]]), \
                                                 k=1, ext=3)
        plt.plot(np.concatenate([plasma_data["rhop"][index], rhop_ext]), np.concatenate([plasma_data["Te"][index] , Te_ext]))
        print(np.min(Te_ext))
        plt.show()
        Te = Te_spline(EQ_t.rhop.flatten())
        Te = np.reshape(Te, EQ_t.rhop.shape)
        print(np.max(Te), np.min(Te))
        np.savetxt(os.path.join(ext_data_folder, "Te{0:d}".format(index)), Te)
        plt.figure()
        # plt.contourf(EQ_t.R, EQ_t.z, ne.T * 1.e-19, levels=np.linspace(0, 8, 30))
        plt.imshow(ne[:, ::-1].T / np.max(ne))
        plt.figure()
        plt.imshow(Te[:, ::-1].T / np.max(Te))
        index += 1
        plt.show()


def get_RELAX_target_current(shot, time, exp="AUGD", ed=0, smoothing=1.e-3):
    IDF = dd.shotfile("IDF", pulseNumber=int(shot), experiment=exp, edition=ed)
    IDG = dd.shotfile("IDG", pulseNumber=int(shot), experiment=exp, edition=ed)
    I_tor = IDG.getSignal("Itor", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    Ohmic_cur = IDF.getSignal("ohmi_tot", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    ECCD_cur = IDF.getSignal("eccd_tot", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    Bootstrap_cur = IDF.getSignal("bscd_tot", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    NBCD_cur = IDF.getSignal("nbcd_tot", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    if(len(Ohmic_cur) > 10):
        Ohmic_cur = smooth(Ohmic_cur, True)[0]
        ECCD_cur = smooth(ECCD_cur, True)[0]
        Bootstrap_cur = smooth(Bootstrap_cur, True)[0]
        NBCD_cur = smooth(NBCD_cur, True)[0]
    else:
        Ohmic_cur = np.mean(Ohmic_cur)
        ECCD_cur = np.mean(ECCD_cur)
        Bootstrap_cur = np.mean(Bootstrap_cur)
        NBCD_cur = np.mean(NBCD_cur)
    if(len(I_tor) > 10):
        I_tor = smooth(I_tor, True)[0]
    else:
        I_tor = np.mean(I_tor)
    print("Total current", I_tor * 1.e-6, "MA")
    j_tot = (Ohmic_cur + ECCD_cur + Bootstrap_cur + NBCD_cur)
    print("Sum of currents inside the plasma", j_tot * 1.e-6, "kA")
    print("Current fractions: ECCD, Ohmic, Bootstrap, NBCD [%]", \
          ECCD_cur / j_tot * 100.e0, \
          Ohmic_cur / j_tot * 100.e0, \
          Bootstrap_cur / j_tot * 100.e0, \
          NBCD_cur / j_tot * 100.e0)
    print("Current absolute value: ECCD, Ohmic, Bootstrap, NBCD [kA]", \
          ECCD_cur * 1.e-3, \
          Ohmic_cur * 1.e-3, \
          Bootstrap_cur * 1.e-3, \
          NBCD_cur * 1.e-3)
    return (Ohmic_cur + ECCD_cur) * 1.e-6

def get_total_current(shot, time, exp="AUGD", diag="FPC", ed=0, smoothing=1.e-3):
    I_diag = dd.shotfile(diag, pulseNumber=int(shot), experiment=exp, edition=ed)
    if(diag != "IDG"):
        Ip = I_diag.getSignal("IpiFP", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    else:
        Ip = I_diag.getSignal("Itor", tBegin=time - smoothing * 0.5, tEnd=time + smoothing * 0.5)
    if(len(Ip) > 10):
        Ip = smooth(Ip, True)[0]
    else:
        Ip = np.mean(Ip)
    return Ip


def get_Vloop(shot, time, exp="AUGD", ed=0, smoothing=1.e-2):
    MAG = dd.shotfile("MAG", pulseNumber=int(shot), experiment=exp, edition=ed)
    Vloop = MAG.getSignal("ULid12")
    time_base = MAG.getTimeBase("ULid12")
    offset = np.mean(Vloop[time_base < -5.0])
#    plt.plot(time_base, Vloop - offset)
#    plt.plot()
#    plt.show()
#    if(len(offset) > 10):
#        offset, y_err = smooth(offset, True)
#    else:
#        offset = np.mean(offset)
    print("Offset: {0:1.2e}".format(offset))
    imin = np.argmin(np.abs(time_base - (time - smoothing * 0.5)))
    imax = np.argmin(np.abs(time_base - (time + smoothing * 0.5)))
#    if(imax - imin > 10):
#        Vloop, y_err = smooth(Vloop[imin:imax], True)
#    el
    if(imax != imin):
        Vloop = np.mean(Vloop[imin:imax])
    else:
        Vloop = Vloop[imin]
    return Vloop - offset

def get_IDE_Vloop(shot, time, exp="AUGD", ed=0, smoothing=1.e-2):
    IDF = dd.shotfile("IDF", pulseNumber=int(shot), experiment=exp, edition=ed)
    uloop = -IDF.getSignalGroup("uloopsmo")
    time_base = IDF.getTimeBase("uloop")
    rhop = IDF.getAreaBase("uloop")
    Vloop = IDF.getSignalGroup("uloopmes")
#    plt.plot(time_base, Vloop - offset)
#    plt.plot()
#    plt.show()
#    if(len(offset) > 10):
#        offset, y_err = smooth(offset, True)
#    else:
#        offset = np.mean(offset)
    imin = np.argmin(np.abs(time_base - (time - smoothing * 0.5)))
    imax = np.argmin(np.abs(time_base - (time + smoothing * 0.5)))
    print(rhop.shape)
#    if(imax - imin > 10):
#        Vloop, y_err = smooth(Vloop[imin:imax], True)
#    el
    if(imax != imin):
        uloop = np.mean(uloop[imin:imax], axis=0)
        Vloop = np.mean(Vloop[imin:imax], axis=0)
        rhop = np.mean(rhop[imin:imax], axis=0)
    else:
        uloop = uloop[imin]
        Vloop = Vloop[imin]
        rhop = rhop[imin]
    print(uloop.shape)
    plt.plot(rhop, uloop)
    vloop_prof = np.zeros(len(rhop))
    for i in range(len(Vloop)):
        vloop_prof[:] = Vloop[i]
        plt.plot(rhop, vloop_prof)
    plt.show()
    return np.mean(uloop), np.mean(Vloop)

def get_ECE_spectrum(shotno, time, diag, Te):
    std_dev, diag_data = get_data_calib(diag, shot=int(shotno), time=float(time))
    freq = get_freqs(shotno, diag)
    if(Te):
        return [freq, diag_data[1], std_dev[0] + std_dev[1]]
    I_bb = freq ** 2 * cnst.e / (8.0 * np.pi * cnst.c ** 2)
    try:
        output = [freq, diag_data[1] * I_bb * 1.e3, (std_dev[0] + std_dev[1]) * I_bb * 1.e3]
    except ValueError as e:
        print(e)
        print(len(freq))
        print(len(diag_data[1]))
        print(len(std_dev[0] + std_dev[1]))
        output = []
    return output



def test_FPC():
    FPC_shot = dd.shotfile("FPC", int(32028))
    time = 2.4
    signal = FPC_shot.getSignal("IpiFP", tBegin=time,
                  tEnd=time)
    print(signal)
# if __name__ == "__main__":
#    shot = 33261
#    chs = [6, 40]
#    fig = plt.figure("CTA ECRH cprrelation", (12, 8.5))
#    ax2 = fig.add_subplot(212)
#    ax1 = fig.add_subplot(211, sharex=ax2)
#    for ch in chs:
#        t, raw, filtered = get_CTA_no_pinswitch(shot, 'CTA', 'AUGD', 0, ch)
#        binned_signal, binned_t = resample(filtered[0], len(filtered[0][::8]), t=t)
#        ax1.plot(binned_t, -binned_signal, "-", label=r"Ch. " + str(ch))
#    ax1.legend()
#    # ax1.set_xlabel(r"$t\,[\mathrm{s}]$")
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    ax1.set_ylabel(r"$U\,[\mathrm{V}]$")
#    gy_list = load_all_active_ECRH(shot)
#    for gy in gy_list:
#        ax2.plot(gy.time, gy.PW / 1.e6)
#    ax2.set_xlabel(r"$t\,[\mathrm{s}]$")
#    ax2.set_ylabel(r"$P_\mathrm{ECRH}\,[\mathrm{MW}]$")
#    plt.show()
    # y_err, TS_c_data = get_Thomson_data(33705, 4.95, Diag("TS_c", "AUGD", "VTA", 0), Te=True, ne=False, \
    #                                    edge=False, core=True, eq_diag=Diag("EQ", "AUGD", "EQH", 0))
    # print(TS_c_data[0], TS_c_data[1])
    # plt.errorbar(TS_c_data[0], TS_c_data[1], y_err)
    # plt.show()
    # plt.plot(t, raw[0], '+')


def compare_IDE_to_MBI(shot):
    IDF = dd.shotfile("IDF", int(shot), experiment="AUGD", edition=0)
    MBI = dd.shotfile('MBI', int(shot))
    B_IDE = IDF.getSignal("Btor")
    time_IDE = IDF.getTimeBase("Btor")
    B_MBI = MBI.getSignal("BTFABB")
    time_MBI = MBI.getTimeBase("BTFABB")
    IDE_spl = InterpolatedUnivariateSpline(time_IDE, B_IDE)
    MBI_spl = InterpolatedUnivariateSpline(time_MBI, B_MBI)
    t = np.linspace(max(np.min(time_IDE),np.min(time_MBI)), min(np.max(time_IDE),np.max(time_MBI)), 1000)
    plt.plot(t, IDE_spl(t)/MBI_spl(t))
#     plt.plot(time_IDE, B_IDE, label="IDE")
#     plt.plot(time_MBI, B_MBI, "--", label="MBI")
#     plt.legend()
    plt.show()
    

if(__name__ == '__main__'):
    print(get_Vloop(35662, 3.84))