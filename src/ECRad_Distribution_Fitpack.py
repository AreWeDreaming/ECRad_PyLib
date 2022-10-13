'''
Created on Jun 19, 2019

@author: sdenk
This module is outdated.
'''

import os
import numpy as np
from threading import Lock
from scipy.optimize import least_squares
from WX_Events import ThreadFinishedEvt, Unbound_EVT_FIT_FINISHED
import wx
from time import sleep
from subprocess import Popen, call
from Distribution_Functions import Gauss_norm
from scipy.interpolate import InterpolatedUnivariateSpline
# Basic routines to fit model distribution function data to measurements
# Most likely outdated, has not been maintained since 2015
def fit_TRad(args):
    Callee = args[0]
    path = args[1]
    beta = np.copy(args[2])
    fit = args[3]
    dstf = args[4]
    exec_ECRad_model = args[5]
    ECE_data = args[6]
    ECE_y_err = args[7]
    thread_Lock = Lock()
    sd_beta = np.zeros(len(beta))
    sd_beta[:] = -1.0  # No fit
    if(dstf == "BM" or dstf == "BJ"):
        model_func = evaluate_bi_max
        if(dstf == "BJ"):
            Trad_filename = os.path.join(path, "TRadM_BiMnJ.dat")
            res_filename = os.path.join(path, "bi_maxj.res")
        else:
            Trad_filename = os.path.join(path, "TRadM_BiMax.dat")
            res_filename = os.path.join(path, "bi_max.res")
        parameter_filename = "bi_max.dat"
        model_func = evaluate_bi_max
        beta_bounds = np.array([[1.e-7, 0.0, 1.e-3, 5.e3, 5.e3], [1.0, 1.0, np.inf, 5.e5, 5.e5]])
    elif(dstf == "DM"):
        Trad_filename = os.path.join(path, "TRadM_Drift.dat")
        parameter_filename = "drift_m.dat"
        res_filename = os.path.join(path, "Drift_max.res")
    elif(dstf == "MS"):
        model_func = evaluate_multi_slope
        Trad_filename = os.path.join(path, "TRadM_MultS.dat")
        parameter_filename = "multi_s.dat"
        res_filename = os.path.join(path, "multi_s.res")
        beta_bounds = np.array([[-100, 0.9999], [1.0, 1.5]])
    elif(dstf == "RA"):
        model_func = evaluate_runaway
        Trad_filename = os.path.join(path, "TRadM_RunAw.dat")
        res_filename = os.path.join(path, "run_away.res")
        parameter_filename = "runaway.dat"
    else:
        print("Invalid value for dstf", dstf)
        raise(ValueError)
    parameter_filename = os.path.join(path, parameter_filename)
    fun_args = {}
    fun_args["exec_ECRad_model"] = exec_ECRad_model
    fun_args["parameter_filename"] = parameter_filename
    fun_args["Trad_filename"] = Trad_filename
    os.environ['OMP_NUM_THREADS'] = "24"
    os.environ['OMP_STACKSIZE'] = "{0:d}".format(int(np.ceil(10000 * 3.125) * 3))
    if(fit):
        thread_Lock.acquire()
        res = least_squares(model_func, beta, bounds=beta_bounds, \
                            args=[exec_ECRad_model, Trad_filename, parameter_filename, ECE_data, ECE_y_err])
        print(res.message)
        thread_Lock.release()
        beta = res.x
        sd_beta = np.zeros(len(beta))
        if(res.success):
            print("Fit successfull")
        else:
            print("Fit failed")
            print("status: ", res.status)
    else:
        if(dstf == "BM"):
            state = make_bi_max(beta, parameter_filename)
        elif(dstf == "BJ"):
            state = make_bi_max(beta, parameter_filename)
        elif(dstf == "DM"):
            state = make_drift_m(beta, parameter_filename)
        elif(dstf == "MS"):
            state = make_multi_slope(beta, parameter_filename)
        elif(dstf == "RA"):
            state = make_runaway(beta, parameter_filename)
        if(state):
            print("Parametrized distribution ready")
        else:
            print("Error when preparing parametrization")
    res_file = open(res_filename, "w")
    for i in range(len(beta)):
        res_file.write("{0:1.5e} \t {1:1.5e} \n".format(beta[i], sd_beta[i]))
    res_file.flush()
    res_file.close()
    evt_out = ThreadFinishedEvt(Unbound_EVT_FIT_FINISHED, Callee.GetId())
    wx.PostEvent(Callee, evt_out)

def evaluate_bi_max(beta, exec_ECRad_model, trad_filename, bi_max_filename, ECE_data, ECE_y_err):
    if(make_bi_max(beta, bi_max_filename)):
        ECRad = Popen(exec_ECRad_model)
        sleep(0.1)
        os.system("renice -n 10 -p " + "{0:d}".format(ECRad.pid))
        stderr_log = []
        while(ECRad.poll() is None):
            stdout, stderr = ECRad.communicate(None)
            stderr_log.append(stderr)
            print(stdout)
            sleep(0.25)
        for stderr in stderr_log:
            print(stderr)
    else:
        print("Fit failed")
        return
    Trad = np.loadtxt(trad_filename)
    print("beta", beta)
    print(("Trad", Trad.T[1]))
    print("residues", (Trad.T[1] - ECE_data) / ECE_y_err)
    print("Sum of squares", np.sum((Trad.T[1] - ECE_data) ** 2 / ECE_y_err) ** 2)
    return (Trad.T[1] - ECE_data) / ECE_y_err

def make_bi_max(beta, bi_max_filename):
    bi_file = open(bi_max_filename, "w")
    Te_par = beta[3]
    Te_perp = beta[4]
    rhop = np.linspace(0.0, 1.0, 100)
    bi_file.write("{0: 1.8E}{1: 1.8E}\n".format(Te_par, Te_perp))
    bi_file.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        bi_file.write("{0: 1.8E}{1: 1.8E}\n"\
            .format(rhop[i], Gauss_norm(rhop[i], \
                beta)))  # j[i]
        if(Gauss_norm(rhop[i], beta) > 1.0 or Gauss_norm(rhop[i], beta) < 0.0):
            return False
    bi_file.flush()
    bi_file.close()
    return True

def evaluate_drift_m(beta, x, exec_efcm_model, trad_filename, drift_m_filename):
    make_drift_m(beta, x, drift_m_filename)
    call(exec_efcm_model)
    return np.loadtxt(trad_filename, unpack=True)[1]

def make_drift_m(beta, drift_m_filename):
    drift_m = open(drift_m_filename, "w")
    rhop = np.linspace(0.0, 1.0, 100)
    drift_m.write("{0: 1.8E}{1: 1.8E}{2: 1.8E}{3: 1.8E}"\
            .format(beta[3], beta[4], beta[5], beta[6]))
    drift_m.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        drift_m.write("{0: 1.8E}{1: 1.8E}"\
            .format(rhop[i], Gauss_norm(rhop[i], beta)))
        if(Gauss_norm(rhop[i], beta) > 1.0 or Gauss_norm(rhop[i], beta) < 0.0):
            return False
    drift_m.flush()
    drift_m.close()
    return True

def evaluate_multi_slope(beta, exec_ECRad_model, trad_filename, multi_slope_filename, ECE_data, ECE_y_err):
    if(make_multi_slope(beta, multi_slope_filename)):
        ECRad = Popen(exec_ECRad_model)
        sleep(0.1)
        os.system("renice -n 10 -p " + "{0:d}".format(ECRad.pid))
#        stderr_log = []
        while(ECRad.poll() is None):
#            stdout, stderr = ECRad.communicate(None)
#            stderr_log.append(stderr)
#            print(stdout)
            sleep(0.25)
#        for stderr in stderr_log:
#            print(stderr)
    else:
        print("Fit failed")
        return
    Trad = np.loadtxt(trad_filename)
    return  (Trad.T[1] - ECE_data) / ECE_y_err

def make_multi_slope(beta, multi_slope_filename):
    multi_slope_file = open(multi_slope_filename, "w")
    rhop = np.linspace(0.0, 1.0, 100)
    rhop_Te, Te = np.loadtxt(os.path.join(os.path.dirname(multi_slope_filename), "Te_file.dat"), skiprows=1, unpack=True)
    Te_spline = InterpolatedUnivariateSpline(rhop_Te, Te, k=1)
    scale = Te_spline(rhop) / np.max(Te)
    multi_slope_file.write("{0: 1.8E}\n".format(beta[1]))
    multi_slope_file.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        multi_slope_file.write("{0: 1.8E}{1: 1.8E}\n"\
                          .format(rhop[i], (1.0 - beta[0] * scale[i] ** 2) * Te_spline(rhop[i])))
    multi_slope_file.flush()
    multi_slope_file.close()
    return True

def evaluate_runaway(beta, x, invoke_ECRad, trad_filename, runaway_filename):
    # print("param set", beta)
    if(make_runaway(beta, runaway_filename)):
        call(invoke_ECRad)
    Trad = np.loadtxt(trad_filename)
    return Trad.T[1]

def make_runaway(beta, runaway_filename):
    run_file = open(runaway_filename, "w")
    rhop = np.linspace(0.0, 1.0, 100)
    run_file.write("{0: 1.8E}{1: 1.8E}\n"\
            .format(beta[3], beta[4]))  # j[i]
    run_file.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        run_file.write("{0: 1.8E}{1: 1.8E}\n"\
            .format(rhop[i], Gauss_norm(rhop[i], \
                beta)))  # j[i]
        if(Gauss_norm(rhop[i], beta) < 0.0):  # Gauss_norm(rhop[i], beta) > 1.0 or
            return False
    run_file.flush()
    run_file.close()
    return True