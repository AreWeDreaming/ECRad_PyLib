'''
Created on 09.05.2019

@author: Severin Denk
'''
from Ndarray_Helper import ndarray_check_for_None
from ECRad_Results import ECRadResults
import glob
import os
# from kk_local import KK
# from EQU import EQU
# from plotting_configuration import *
# import numpy as np
# from scipy.interpolate import RectBivariateSpline
# def benchmark_EQU_kk(shot, time, exp, diag, ed):
#     EQH = EQU()
#     EQH.Load(shot, Experiment=exp, Diagnostic=diag, Edition=ed)
#     KKobj = KK()
#     output = KKobj.kkeqpfm(shot, time, exp=exp, diag=diag, ed=ed)
#     R = EQH.getR(time)
#     z = EQH.getz(time)
#     Psi = EQH.getPsi(time)
#     R_los = np.linspace(-0.5, 0.5, 100) + 1.65
#     z_los = np.linspace(-0.2, 0.2, 100)
#     EQH_spl = RectBivariateSpline(R, z, Psi)
#     EQH_LOS = EQH_spl(R_los, z_los, grid=False)
#     kk_spl = RectBivariateSpline(output.R, output.z, output.Psi)
#     kk_LOS = kk_spl(R_los, z_los, grid=False)
#     plt.plot(R_los, EQH_LOS/kk_LOS)
#     print(np.max(np.abs(EQH_LOS/kk_LOS - 1)))
# #     levels = np.linspace(np.min(Psi), np.max(Psi), 10)
# #     plt.contour(R, z, Psi.T, linestyles="-", levels=levels, colors="k")
# #     plt.contour(output.R, output.z, output.Psi.T, linestyles="--", levels=levels, colors="r")
#     plt.show()
#     plt.hold(True)
    

def repair_ECRad_results(folder_in, folder_out=None):
    # Allows to make bulk modification of result files using glob
    # If folder_out is True it overwrites!
    filelist = glob.glob(os.path.join(folder_in, "*.mat"))
    cur_result = ECRadResults()
    for filename in filelist:
        cur_result.reset()
        cur_result.from_mat_file(filename)
        # Enter bulk modifcations here
        cur_result.Scenario.used_diags_dict["CTC"].diag = "CTC"
        if(folder_out is None):
            cur_result.to_mat_file(filename)
        else:
            cur_result.to_mat_file(os.path.join(folder_out, os.path.basename(filename)))    
    
def randomTests():
    mdict = {"key1":None, "key2":[None,None], "key3":[[None,None], [None,None]], "key4":[[[None], [None]]]}
    for key in mdict:
        print(key)
        if(ndarray_check_for_None(mdict[key])):
            print(key + " contains None or is None!")

if(__name__ == "__main__"):
#     benchmark_EQU_kk(35662, 1.5, "AUGD", "IDE", 0)
    randomTests()
