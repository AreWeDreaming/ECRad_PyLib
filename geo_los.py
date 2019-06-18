'''
Created on Apr 13, 2018

@author: sdenk
'''
import os
import numpy as np
import ctypes as ct
from GlobalSettings import itm
geo_los_path = "/afs/ipp-garching.mpg.de/home/e/eced/CEC_working/libece/" + os.environ["SYS"]
if(not os.path.isdir(geo_los_path)):
    geo_los_path = "/afs/ipp-garching.mpg.de/home/e/eced/CEC_working/libece/amd64_sles15/"
lib_geo_los = ct.cdll.LoadLibrary(os.path.join(geo_los_path, "libece.so"))
from numpy.ctypeslib import ndpointer
geo_fun = lib_geo_los.geo_los
geo_fun.restype = None
geo_fun.argtypes = [ct.c_int32, \
                ct.c_int32, \
                ct.c_float, \
                ct.c_int32, \
                ndpointer(ct.c_float, flags="C_CONTIGUOUS"), \
                ndpointer(ct.c_float, flags="C_CONTIGUOUS")]

def geo_los(shot, wg, z_lens, R, z):
    R_float32 = np.array(np.copy(R), dtype=np.float32)
    z_float32 = np.array(np.copy(z), dtype=np.float32)
    geo_fun(shot, wg, z_lens, len(R), R_float32, z_float32)
    R[:] = R_float32[:]
    z[:] = z_float32[:]
    return R, z
