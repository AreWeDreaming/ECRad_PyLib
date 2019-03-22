'''
Created on Oct 5, 2017

@author: sdenk
'''

import numpy as np
import matplotlib.pyplot as plt
from electron_distribution_utils import load_f_from_ASCII, export_fortran_friendly
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline

def drag_distribution(folder, beta, out_folder):
    dist_obj = load_f_from_ASCII(folder)
    u_crit = np.sqrt(beta[1] ** 2 - 1.0)
    drag_g = 1.0 / (1.0 - beta[0] * (np.pi / 2.0 + np.arctan((dist_obj.u - u_crit) * 100)) / np.pi)
    f_new = np.zeros(dist_obj.f.shape)
    for i in range(len(dist_obj.f)):
        f_spl = RectBivariateSpline(dist_obj.u, dist_obj.pitch, dist_obj.f_log[i])
        for ipitch in range(len(dist_obj.pitch)):
            pitch_helper = np.zeros(len(dist_obj.u))
            pitch_helper = dist_obj.pitch[ipitch]
            f_deriv = f_spl(dist_obj.u, pitch_helper, dx=1, grid=False)
            f_deriv_spl = InterpolatedUnivariateSpline(dist_obj.u, f_deriv * drag_g)
            for iu in range(len(dist_obj.u)):
                f_new[i, iu, ipitch] = f_deriv_spl.integral(0.0, dist_obj.u[iu]) + dist_obj.f_log[i].T[ipitch][0.0]
        f_new[i] = np.exp(f_new[i])
        ipitch = len(dist_obj.pitch) / 2
        u_mat = np.zeros(f_new[i].shape)
        for ipitch in range(len(dist_obj.pitch)):
            u_mat[:, ipitch] = dist_obj.u
        f_new[i] /= RectBivariateSpline(dist_obj.u, dist_obj.pitch, f_new[i] * \
                                        u_mat ** 2 * np.pi * 2.e0 * np.sin(dist_obj.pitch)).integral(np.min(dist_obj.u), \
                                                                                      np.max(dist_obj.u), \
                                                                                      np.min(dist_obj.pitch), \
                                                                                      np.max(dist_obj.pitch))
#        print(RectBivariateSpline(dist_obj.u, dist_obj.pitch, f_new[i] * \
#                                        u_mat ** 2 * np.pi * 2.e0 * np.sin(dist_obj.pitch)).integral(np.min(dist_obj.u), \
#                                                                                      np.max(dist_obj.u), \
#                                                                                      np.min(dist_obj.pitch), \
#                                                                                      np.max(dist_obj.pitch)))
#        plt.plot(dist_obj.u, f_new[i, :, ipitch])
        dist_obj.f[i] = f_new[i]
        dist_obj.f_log[i] = np.log(f_new[i])
        dist_obj.f_log10[i] = np.log10(f_new[i])
#        ipitch = len(dist_obj.pitch) / 2
#        plt.plot(dist_obj.u, dist_obj.f_log[i].T[ipitch])
#        plt.plot(dist_obj.u, f_new[i].T[ipitch])
#        plt.show()
    export_fortran_friendly([dist_obj, out_folder])
    # print(uxx, ull)

drag_distribution("/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_12/ecfm_data/fRe/", [0.5, 1.14], \
                  "/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_15/ecfm_data/fRe/")
