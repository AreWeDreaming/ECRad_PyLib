'''
Created on 09.05.2019

@author: sdenk
'''
from ndarray_helper import ndarray_check_for_None
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
    
def randomTests():
    mdict = {"key1":None, "key2":[None,None], "key3":[[None,None], [None,None]], "key4":[[[None], [None]]]}
    for key in mdict.keys():
        print(key)
        if(ndarray_check_for_None(mdict[key])):
            print(key + " contains None or is None!")

if(__name__ == "__main__"):
#     benchmark_EQU_kk(35662, 1.5, "AUGD", "IDE", 0)
    randomTests()
