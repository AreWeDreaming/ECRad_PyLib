'''
Created on Mar 11, 2019

@author: sdenk
'''
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

def Integrate2D(x, y, f):
    f1D_int = []
    for i in range(len(x)):
        spl = InterpolatedUnivariateSpline(y, f[i])
        f1D_int.append(spl.integral(y[0], y[-1]))
    spl = InterpolatedUnivariateSpline(x, np.array(f1D_int))
    return spl.integral(x[0], x[-1])

