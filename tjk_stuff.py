'''
Created on Nov 13, 2017

@author: sdenk
'''
folder = "/ptmp1/work/sdenk/TJK/"
import numpy as np
import os
from plotting_configuration import *
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, RectBivariateSpline
from scipy.optimize import minimize

def eval_spline(x_vec, spl):
    return np.array([spl[0](x_vec[0], x_vec[1])])

def eval_spline_grad(x_vec, spl):
    return np.array([[spl[0].ev(x_vec[0], x_vec[1], dx=1, dy=0), spl[0].ev(x_vec[0], x_vec[1], dx=0, dy=1)]])


def convert_TJK_files():
    ne = np.loadtxt(os.path.join(folder, "ne.dat"))
    ext_data_folder = os.path.join(folder, "Ext_data")
    if(not os.path.isdir(ext_data_folder)):
        os.mkdir(ext_data_folder)
    index = 0
    R_ax = 0.62
    Psi_sep = 1.0
    m = len(ne[0])
    n = len(ne)
    R_min = 0.425
    z_min = -0.175
    R_max = R_min + float(m) * 1.e-3
    z_max = z_min + float(n) * 1.e-3
    print(n, m)
    print(R_min, R_max)
    print(z_min, z_max)
    R = np.linspace(R_min, R_max, m)
    z = np.linspace(z_min, z_max, n)
    Br = np.loadtxt(os.path.join(folder, "Br.dat"))
    Bt = np.loadtxt(os.path.join(folder, "Bt.dat"))
    Bz = np.loadtxt(os.path.join(folder, "Bz.dat"))
    Te = ne / np.max(ne) * 8.5
    Psi = ((ne - np.min(ne)) / (np.max(ne) - np.min(ne))) ** 2
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.contourf(R, z, np.sqrt(Br ** 2 + Bt ** 2 + Bz ** 2))
    ax2.contourf(R, z, ne)
    ax3.contourf(R, z, Te)
    times = np.array([0.02])
    np.savetxt(os.path.join(ext_data_folder, "t"), times)
    np.savetxt(os.path.join(ext_data_folder, "special_points{0:d}".format(index)), np.array([R_ax, Psi_sep]))
    np.savetxt(os.path.join(ext_data_folder, "R{0:d}".format(index)), R)
    np.savetxt(os.path.join(ext_data_folder, "z{0:d}".format(index)), z)
    np.savetxt(os.path.join(ext_data_folder, "Psi{0:d}".format(index)), Psi.T)
    np.savetxt(os.path.join(ext_data_folder, "Br{0:d}".format(index)), Br.T)
    np.savetxt(os.path.join(ext_data_folder, "Bt{0:d}".format(index)), Bt.T)
    np.savetxt(os.path.join(ext_data_folder, "Bz{0:d}".format(index)), Bz.T)
    np.savetxt(os.path.join(ext_data_folder, "Te{0:d}".format(index)), Te.T)
    np.savetxt(os.path.join(ext_data_folder, "ne{0:d}".format(index)), ne.T)
    plt.show()

def inspect_TJK_files():
    ne = np.loadtxt(os.path.join(folder, "ne.dat"))
    m = len(ne[0])
    n = len(ne)
    R_min = 0.425
    z_min = -0.175
    R_max = R_min + float(m) * 1.e-3
    z_max = z_min + float(n) * 1.e-3
    print(n, m)
    print(R_min, R_max)
    print(z_min, z_max)
    R = np.linspace(R_min, R_max, m)
    z = np.linspace(z_min, z_max, n)
    # Rhop matrix - main problem is that ne is flat
#    plt.plot(opt.x[0], opt.x[1], "+k")
#    plt.show()
    Br = np.loadtxt(os.path.join(folder, "Br.dat"))
    Bt = np.loadtxt(os.path.join(folder, "Bt.dat"))
    Bz = np.loadtxt(os.path.join(folder, "Bz.dat"))
    ne_profile = ne.T[n / 2][np.argmax(ne.T[n / 2]):m]
    rhop = np.linspace(0.0, 1.0, len(ne_profile))
    Psi = np.zeros((n, m))
#    plt.contourf(R, z, Bz)
#    plt.show()
    R_ax = 0.0
    for i in range(n):
        print("Working on line", i)
        Bz_l_spl = InterpolatedUnivariateSpline(R, Bz[i] * R)
        for j in range(m):
            Psi[i, j] = 2.0 * np.pi * (Bz_l_spl.integral(R_ax, R[j]))
#            Psi[i, j] = 2.0 * np.pi * (Bz_spl.integral(R_ax, z[i], R[j], z[i]))
    rhop_mat = Psi - np.min(Psi)
    rhop_mat = np.sqrt(rhop_mat / np.max(rhop_mat)) * 1.2
    plt.plot(R, rhop_mat[n / 2])
    plt.show()
    plt.contourf(R, z, rhop_mat, levels=np.linspace(0.0, 1.0, 20))
    # Rhop matrix - main problem is that ne is flat
#    rhop_mat *= np.sqrt(R ** 2 + z ** 2) / np.max(np.sqrt(R ** 2 + z ** 2))
 #    plt.contourf(R, z, rhop_mat)
 #    plt.show()
    indicies = np.unravel_index(np.argmin(rhop_mat), rhop_mat.shape)
    R_init = np.array([R[indicies[1]], z[indicies[0]]])
    rhop_spl = RectBivariateSpline(R, z, rhop_mat.T)
    opt = minimize(eval_spline, R_init, args=[rhop_spl], \
                 bounds=[[np.min(R), np.max(R)], [np.min(z), np.max(z)]])
#    print("Magnetic axis position: ", opt.x[0], opt.x[1])
    plt.contourf(R, z, rhop_mat)
    plt.plot(opt.x[0], opt.x[1], "+k")
    plt.show()
#    plt.plot(rhop, ne_profile)
    ne_profile *= np.exp(-(rhop / np.max(rhop)) ** 8)
#    plt.plot(rhop, ne_profile, "--")
    ne_spline = UnivariateSpline(rhop, np.log(ne_profile), s=1.e-2)
    rhop_new = np.linspace(0, 1.2, 200)
    ne_new = np.exp(ne_spline(rhop_new))
    Te = ne_new / np.max(ne_new) * 10.e0
#    plt.plot(rhop_new, np.exp(ne_spline(rhop_new)), ":")
#    plt.contour(R, z, rhop)
#    plt.gca().set_aspect("equal")


if(__name__ == "__main__"):
#    inspect_TJK_files()
    convert_TJK_files()
