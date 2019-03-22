'''
Created on Jan 26, 2017

@author: sdenk
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares
from electron_distribution_utils import Gauss_norm
from scipy.integrate import simps
import os
afs_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/"
def linear_interpolation(x1, x2, y1, y2, x, y):
    return y1 + (y2 - y1) / (x2 - x1) * (x - x1)

def current_dif(params, rho, current_in, N_time, dt, rho_params):
    current = np.copy(current_in)
    init_current = simps(current, rho)
    print(params)
    D_spl = InterpolatedUnivariateSpline(rho_params, np.array(params[1:len(params)]))
    D = D_spl(rho)
    while(np.max(D_spl(rho)) * dt < 5.e-7):
        N_time /= 10
        dt *= 1.e1
    while(np.max(D_spl(rho)) * dt > 5.e-6):
        N_time *= 10
        dt *= 1.e-1
    print("Solving diffusion equation - current step count", N_time)
    N = len(current) - 1
    curr_tmp = current  # initialization
    dr = np.zeros(len(rho))
    dr2 = np.zeros(len(rho))
    for i in range(1, N - 1):
        dr[i] = 0.5e0 * (rho[i + 1] - rho[i - 1])
    dr[1] = rho[1] - rho[0]
    dr[N] = rho[N] - rho[N - 1]
    dr2[:] = dr[:] / 2.e0
    for it in range(N_time):
        current[1:N - 1] += D[1:N - 1] * dt / (rho[1:N - 1] * dr[1:N - 1] ** 2) * \
                     ((rho[1:N - 1] + dr2[1:N - 1]) * (curr_tmp[2:N] - curr_tmp[1:N - 1]) - \
                      (rho[1:N - 1] - dr2[1:N - 1]) * (curr_tmp[1:N - 1] - curr_tmp[0:N - 2]))
        # interpolation to end points
        current[0] = current[1]
        linear_interpolation(rho[N - 2], rho[N - 1], current[N - 2], current[N - 1], \
                             rho[N], current[N])
    print("Renormalizing by 1 /", simps(current, rho) / init_current)
    current *= init_current / simps(current, rho)
    current *= params[0]
    return current

def current_dif_fit_func(params, rho, current, N_time, dt, rho_params, j_RELAX):
    current = current_dif(params, rho, current, N_time, dt, rho_params)
    return (current - j_RELAX)

def fit_cur_diff(rho, j_linear, j_RELAX):
    # Both should use the same grid
    params = np.zeros(11)
    params[:] = 10.0
    params[0] = 0.5
    params_bounds = np.array([np.zeros(len(params)), np.zeros(len(params))])
    params_bounds[0] = 1.e-2
    params_bounds[1] = 1.e2
    rho_min = np.min(rho)
    rho_max = np.max(rho)
    rho_params = np.linspace(rho_min, rho_max, len(params) - 1)
    # res = least_squares(current_dif_fit_func, params, bounds=params_bounds, \
    #                    args=[rho, j_linear, 100, 1.e-5, rho_params, j_RELAX], \
    #                    x_scale=params)
    # print(res.message)
    # params = res.x
    diff_j_linear = current_dif(params, rho, j_linear, 100, 1.e-5, rho_params)
    return rho_params, params, diff_j_linear

def test_cur_diff(shot, time):
    j_file = os.path.join(afs_path, "j_" + str(shot) + "_" + "{0:1.2}".format(time) + ".dat")
    j_data = np.loadtxt(j_file)
    rho = j_data.T[0]
    if(np.all(np.abs(j_data.T[2]) < 1.e-6)):
        j_linear = j_data.T[1]
    else:
        j_linear = j_data.T[2]
    j_RELAX = j_data.T[3]
    rho_params, params, j_diff = fit_cur_diff(rho, j_linear, j_RELAX)
    plt.plot(rho, j_linear / 1.e6)
    plt.plot(rho, j_RELAX / 1.e6)
    plt.plot(rho, j_diff / 1.e6, "+")
    fig = plt.figure()
    plt.plot(rho_params, params[1:len(params)], "+")
    params_spl = InterpolatedUnivariateSpline(rho_params, np.array(params[1:len(params)]))
    plt.plot(rho, params_spl(rho))
    plt.show()

# 99.99966306   1.71203582  43.02627133  99.19400319   2.48685732
#  42.56539039   7.63117716  28.06249352   8.10759007  27.17908847
#
#
if(__name__ == "__main__"):
    test_cur_diff(33697, 4.80)
