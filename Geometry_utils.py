'''
Created on Sep 7, 2017

@author: sdenk
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import matplotlib.pyplot as plt
from winding_number import wn_PnPoly
import scipy.optimize as scopt
from scipy import __version__ as scivers


def rotate_around_axis(N_vec, n_surf, theta=np.pi / 2.e0):
    rotated_N_vec = np.zeros(3)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    ux = np.zeros([3, 3])
    uxu = np.zeros([3, 3])
    R = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if(i == j):
                R[i, j] = cos_theta
            uxu[i, j] = n_surf[i] * n_surf[j]
    ux[0, 1] = -n_surf[2]
    ux[1, 0] = n_surf[2]
    ux[2, 0] = -n_surf[1]
    ux[0, 2] = n_surf[1]
    ux[1, 2] = -n_surf[0]
    ux[2, 1] = n_surf[0]
    for j in range(3):
        R[:, j] += sin_theta * ux[:, j] + (1.e0 - cos_theta) * uxu[:, j]
    print(R)
    for j in range(3):
        rotated_N_vec[j] = np.sum(N_vec[:] * R[j, :])
    return rotated_N_vec

def Snellius_3D(N_vec, n_surf, N_abs_1, N_abs_2):
    ratio = N_abs_1 / N_abs_2
    scalar_k_surf = -np.sum(n_surf * N_vec)
    if(1.e0 - ratio ** 2 * (1.e0 - scalar_k_surf ** 2) < 0.e0):
        print("Encountered NAN when making Snell's law")
        print("Refractive index", N_abs_1)
        print("Angle between k and flux surface", np.arccos(-scalar_k_surf) * 180.e0 / np.pi)
        print("Det", 1.e0 - ratio ** 2 * (1.e0 - scalar_k_surf ** 2))
    return ratio * N_vec + (ratio * scalar_k_surf - \
                      np.sqrt(1.e0 - ratio ** 2 * (1.e0 - scalar_k_surf ** 2))) * n_surf

def get_Surface_area_of_torus(R, z):  # R and z contour points of the poloidal cross section
    s = np.linspace(0.0, 1.0, len(R))
    dR_spl = InterpolatedUnivariateSpline(s, R).derivative(1)
    dz_spl = InterpolatedUnivariateSpline(s, z).derivative(1)
    IntSpl = InterpolatedUnivariateSpline(s, 2.0 * np.pi * R * np.sqrt(dR_spl(s) ** 2 + dz_spl(s) ** 2))
    return IntSpl.integral(0.0, 1.0)

def get_arclength(R, z):  # R and z contour points of the poloidal cross section
    s = np.linspace(0.0, 1.0, len(R))
    dR_spl = InterpolatedUnivariateSpline(s, R).derivative(1)
    dz_spl = InterpolatedUnivariateSpline(s, z).derivative(1)
    IntSpl = InterpolatedUnivariateSpline(s, np.sqrt(dR_spl(s) ** 2 + dz_spl(s) ** 2))
    return IntSpl.integral(0.0, 1.0)


def get_av_radius(R, z, S, R_ax, z_ax):
    # S is the integrated arc length
    s = np.linspace(0.0, S, len(R))
    r_spline = InterpolatedUnivariateSpline(s, np.sqrt((R - R_ax) ** 2 + (z - z_ax) ** 2))
#    plt.plot(s, r_spline(s))
#    plt.show()
    return r_spline.integral(0.0, S) / S

def eval_z(x, spl):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1])) ** 2
    else:
        return (spl(x[0], x[1], grid=False)) ** 2

def eval_spline(x, args):
    if(scivers == '0.12.0'):
        return args[0].ev(x[0], x[1])
    else:
        return args[0](x[0], x[1], grid=False)

def maximize_y(x):
    return -x[1] ** 3

def minimize_y(x):
    return x[1] ** 3

def check_neighbor(x1_cp, y1_cp, x2_cp, y2_cp, x, y, z):
    ix1_border = np.where(x > min(x1_cp, x2_cp))[0].flatten()[0]
    iy1_border = np.where(y > min(y1_cp, y2_cp))[0].flatten()[0]
    ix2_border = np.where(x <= max(x1_cp, x2_cp))[0].flatten()[-1]
    iy2_border = np.where(y <= max(y1_cp, y2_cp))[0].flatten()[-1]
    if(ix2_border - ix1_border == -1 and iy2_border - iy1_border == -1):
        # Same cell
        add_point = True
        print("Same cell")
    elif(ix2_border - ix1_border == -1 and iy2_border - iy1_border == 0 and \
         (z[ix1_border, iy1_border] * z[ix2_border, iy1_border] < 0)):
    # Crossing only in y
        add_point = True
        print("y-neighbors")
    elif(iy2_border - iy1_border == -1 and ix2_border - ix1_border == 0 and \
         (z[ix1_border, iy1_border] * z[ix1_border, iy2_border] < 0)):
        # Crossing only in x
        add_point = True
        print("x-neighbors")
    elif(ix2_border - ix1_border == 0 and iy2_border - iy1_border == 0):
        # Crossing cells in either x or y direction -> needs checking
        submat = np.sign(z[ix1_border - 1:ix1_border + 2, iy1_border - 1:iy1_border + 2])
        crossings = 0
        if(submat[0, 1] * submat[1, 1] < 0):
            crossings += 1
        if(submat[1, 0] * submat[1, 1] < 0):
            crossings += 1
        if(submat[1, 1] * submat[1, 2] < 0):
            crossings += 1
        if(submat[1, 1] * submat[2, 1] < 0):
            crossings += 1
        if(crossings > 0 and crossings <= 3):
            # four crossings -> Saddle point
            add_point = True
            print("diagonal neighbors")
        else:
            add_point = False
    else:
        # Multiple cells in between -> do not add
        add_point = False
    return add_point

def get_contour(x, y, z_in, val):
    # Only works if z has exactly a single, convex, nested and closed contour
    # Easiest if we look for the zero contour
    z = np.copy(z_in) - val
    # Now find the largest and smallest y with a sign change in y-direction
    iy_min = len(y)
    iy_max = 0
    for ix in range(len(x)):
        for isc in np.where(z[ix, 1:] - z[ix, 0:len(y) - 1])[0]:
            if(isc < iy_min):
                iy_min = isc
            if(isc > iy_max):
                iy_max = isc
    if(iy_min - 1 > 0):
        iy_min -= 1
    if(iy_max + 1 < len(y)):
        iy_max += 1
    spl = RectBivariateSpline(x, y, z)
    dxmin = np.min(np.abs(x[1:] - x[0:len(x) - 1]))  # smallest spacing in x
    dymin = np.min(np.abs(y[1:] - y[0:len(y) - 1]))  # smallest spacing in y
    while(y[iy_max] - y[iy_min] < 50 * dymin):
        dymin *= 2
    x_grid = np.arange(np.min(x), np.max(x), dxmin)
    y_grid = np.arange(y[iy_min], y[iy_max], dymin)
    # Assures that contours are less than one grid cell apart at worst
    y_inter = np.zeros(len(x_grid))
    cont = []
    for i in range(len(y_grid)):
        y_inter[:] = y_grid[i]
        z_inter = spl(x_grid, y_inter, grid=False)
        roots = InterpolatedUnivariateSpline(x_grid, z_inter).roots()
        for root in roots:
            cont.append([root, y_grid[i]])
    cont = np.array(cont)
    x_geo = np.mean(cont.T[0])
    y_geo = np.mean(cont.T[1])
    thetas = np.arctan2(cont.T[1] - y_geo, cont.T[0] - x_geo)
    isort = np.argsort(thetas)
    i_last = isort[0]
    i_start = i_last  # Starting point of current contour
    cur_cont = [[cont.T[0][i_last], cont.T[1][i_last]]]
    sorted_conts = []
    insort = 1
    finished = np.zeros(len(isort), dtype=np.bool)
    finished[0] = True
    move_direction = 1  # Reversed if looking for further points of open contour
    while False in finished:
        if(not finished[insort]):
            i = isort[insort]
            x1_cp = cont.T[0][i_last]
            x2_cp = cont.T[0][i]
            y1_cp = cont.T[1][i_last]
            y2_cp = cont.T[1][i]
            if(check_neighbor(x1_cp, y1_cp, x2_cp, y2_cp, x, y, z)):
#                plt.plot(cont.T[0], cont.T[1], "+")
#                plt.plot([x1_cp, x2_cp], [y1_cp, y2_cp], "-")
#                plt.show()
                cur_cont.append([x2_cp, y2_cp])
                finished[insort] = True
                i_last = i
#            else:
#                plt.plot([x[ix1_border - 1], x[ix1_border], x[ix1_border + 1], \
#                          x[ix1_border - 1], x[ix1_border], x[ix1_border + 1], \
#                          x[ix1_border - 1], x[ix1_border], x[ix1_border + 1]], \
#                         [y[iy1_border - 1], y[iy1_border - 1], y[iy1_border - 1], \
#                          y[iy1_border], y[iy1_border], y[iy1_border], \
#                          y[iy1_border + 1], y[iy1_border + 1], y[iy1_border + 1]], "+")
# #                plt.plot(cont.T[0], cont.T[1], "+")
#                plt.plot([x1_cp, x2_cp], [y1_cp, y2_cp], "-")
#                plt.plot([x1_cp, x2_cp], [y1_cp, y2_cp], "*")
#                plt.show()
        insort += move_direction
        if(insort == len(isort) and not np.all(finished)):
        # End of current contour reached
        # First check if start point of last contour can be continued
            i_last = i_start
            move_direction = -1
            insort += move_direction
        if(insort < 0  and not np.all(finished)):
        # Current contour finished
            sorted_conts.append(cur_cont)
        # -> start a new one
            cur_cont = []
            insort = np.where(np.logical_not(finished))[0][0]
            i_last = isort[insort]
            finished[insort] = True
            cur_cont.append([cont.T[0][i_last], cont.T[1][i_last]])
            if(len(isort[np.logical_not(finished)]) > 1):
                insort = np.where(np.logical_not(finished))[0][0]
            move_direction = +1
    # Finally go through all contours and check for closed ones
    closed_info = np.zeros(len(sorted_conts), dtype=np.bool)
    for i_cont in range(len(sorted_conts)):
        cont = sorted_conts[i_cont]
        i = isort[insort]
        x1_cp = cont[0][0]
        x2_cp = cont[0][1]
        y1_cp = cont[-1][0]
        y2_cp = cont[-1][1]
        # Build connection between points
        if(check_neighbor(x1_cp, y1_cp, x2_cp, y2_cp, x, y, z)):
        # Closed contour append first point at end
            sorted_conts[i_cont].append([x1_cp, y1_cp])
            closed_info[i_cont] = True
        # Convert all contours to np arrays
        sorted_conts[i_cont] = np.array(sorted_conts[i_cont])
        plt.plot(sorted_conts[i_cont].T[0], sorted_conts[i_cont].T[1], "-")
#    x_cont = np.concatenate([cont.T[0][isort], [cont.T[0][np.argmin(thetas)]]])
#    y_cont = np.concatenate([cont.T[1][isort], [cont.T[1][np.argmin(thetas)]]])
#    plt.plot(x_cont, y_cont, "-")
    plt.show()
    return closed_info, sorted_conts

if(__name__ == "__main__"):
    n_surf = np.array([-0.158822604797721, 0.976942169466021, -0.142686291297696])
    n_surf /= np.sqrt(np.sum(n_surf ** 2))
    vec_in = np.array([0.4, 0.2, 0.8])
    print(Snellius_3D(vec_in, n_surf, 1.0, 0.88))


#    print("Norm before", np.sqrt(np.sum(vec_in ** 2)))
#    vec_out = rotate_around_axis(vec_in, n_surf, theta=np.pi / 2.784e0)
#    print(vec_out, np.sqrt(np.sum(vec_out ** 2)))
