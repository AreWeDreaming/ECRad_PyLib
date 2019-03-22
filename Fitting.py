'''
Created on Dec 30, 2016

@author: sdenk
'''
from scipy.optimize import curve_fit

import numpy as np

def linear_fit_func(x, *p):
    return x * p[1] + p[0]

def gauss_fit_func(x, *p):
    return p[0] * np.exp(-((x - p[1]) / p[2]) ** 2)

def make_fit(mode, x, y, y_error=None, p_est=None, relative_error=False):
    if(p_est is None):
        print("A first guess for the parameters to be fitted has to be provided")
        raise ValueError
    if(mode == 'linear'):
        fit_func = linear_fit_func
    elif(mode == 'gauss'):
        fit_func = gauss_fit_func
    else:
        print("Just linear and gauss fits implemented so far")
        raise ValueError("Mmake_fit only supports linear fits at this time")
    if(y_error is not None):
        popt, pcov = curve_fit(fit_func, x, y, p0=p_est, sigma=y_error, absolute_sigma=not relative_error, \
              check_finite=True)
    else:
        popt, pcov = curve_fit(fit_func, x, y, p0=p_est)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def eval_func(mode, x, p):
    if(mode == 'linear'):
        fit_func = linear_fit_func
    if(mode == 'gauss'):
        fit_func = gauss_fit_func
    return fit_func(x, p[0], p[1], p[2])


if(__name__ == "__main__"):
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 4, 100)
    y += (0.5 - np.random.rand(100)) * 0.1 * y
    popt, perr = make_fit('linear', x, y, p_est=[0.0, 4.0])
    print("Offset {0:1.4f}, slope  {1:1.4f} | error offset  {2:1.4f}, error slope  {3:1.4f}".format(popt[0], popt[1], perr[0], perr[1]))
    plt.plot(x, y, "+")
    plt.plot(x, x * popt[1] + popt[0])
    plt.show()

