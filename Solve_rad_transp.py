'''
Created on Nov 9, 2016

@author: sdenk
'''
import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from em_Albajar import em_abs_Alb, s_vec, distribution_interpolator, \
                                        gene_distribution_interpolator
from scipy.integrate import ode, quad
from scipy import constants as cnst
import matplotlib.pyplot as plt
from electron_distribution_utils import f_interpolator, \
                                        get_dist_moments_non_rel, load_f_from_ASCII

class rad_transp_data:
    def __init__(self, rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega, dist="thermal", store_results=False, f_interpolator=None, B0=None, B_min_spline=None):
        self.rhop_spl = rhop_spl
        self.ne_spl = ne_spl
        self.Te_spl = Te_spl
        self.theta_spl = theta_spl
        self.omega_c_spl = omega_c_spl
        self.omega = omega
        self.abs_obj = em_abs_Alb()
        self.dist = dist
        self.abs_obj.dist_mode = dist
        self.store_results = store_results
        if(self.store_results):
            self.new_svec = []
            self.abs = []
            self.j = []
        self.f_inter = f_interpolator
        self.B0 = B0
        self.B_min_spline = B_min_spline

    def get_s_vec(self, s):
        rhop = self.rhop_spl(s)
        ne = np.exp(self.ne_spl(s))
        Te = np.exp(self.Te_spl(s))
#        ne = self.ne_spl(s)
#        Te = self.Te_spl(s)
        theta = self.theta_spl(s)
        omega_c = self.omega_c_spl(s)
        return s_vec(rhop, Te, ne, omega_c / np.pi, theta)

    def get_j_alpha(self, s):
        svec = self.get_s_vec(s)
        if(self.dist == "gene" or self.dist == "ext"):
            u, pitch, spline = self.f_inter.get_spline(svec.rhop, svec.Te)
            if(self.dist == "ext"):
                self.abs_obj.Bmin = self.B_min_spline(svec.rhop).item()
                self.abs_obj.ext_dist = distribution_interpolator(u, pitch, spline)
            elif(self.dist == "gene"):
                self.abs_obj.B_min = self.f_inter.B0
                self.abs_obj.ext_dist = gene_distribution_interpolator(u, pitch, spline)
        c_abs, j, pol_coeff = self.abs_obj.abs_Albajar(svec, self.omega, 1)
        if(self.store_results):
            self.new_svec.append(svec)
            self.abs.append(c_abs)
            self.j.append(j)
        return c_abs, j

    def get_alpha(self, s):
        svec = self.get_s_vec(s)
        if(self.dist == "gene" or self.dist == "ext"):
            u, pitch, spline = self.f_inter.get_spline(svec.rhop, svec.Te)
            if(self.dist == "ext"):
                self.abs_obj.Bmin = self.B_min_spline(svec.rhop).item()
                self.abs_obj.ext_dist = distribution_interpolator(u, pitch, spline)
            elif(self.dist == "gene"):
                self.abs_obj.Bmin = self.f_inter.B0
                self.abs_obj.ext_dist = gene_distribution_interpolator(u, pitch, spline)
            self.abs_obj.ext_dist = distribution_interpolator(u, pitch, spline)
        c_abs, j, pol_coeff = self.abs_obj.abs_Albajar(svec, self.omega, 1)
        return c_abs

def dIds(s, I, rad_trans_obj):
    c_abs, j = rad_trans_obj.get_j_alpha(s)
#    print("c_abs , j, I", c_abs, j, I)
    return j * cnst.c ** 2 / ((rad_trans_obj.omega / (2.0 * np.pi)) ** 2 * cnst.e) - c_abs * I

def alpha(s, rad_trans_obj):
    c_abs = rad_trans_obj.get_alpha(s)
    return c_abs

def solve_rad_transp(folder, shot, time, ch, dist, eq_diag="EQH"):
    svec = np.loadtxt(os.path.join(folder, "chdata{0:03d}.dat".format(ch)))
    i_min = np.where(svec.T[0][svec.T[3] != -1])[0][0]
    i_max = np.where(svec.T[0][svec.T[3] != -1])[0][-1]
    s_max = svec.T[0][i_max]
    s_min = svec.T[0][i_min]
    svec.T[0] -= s_min
    rhop_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[3][i_min:i_max])
    ne_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], np.log(svec.T[4][i_min:i_max]))
    Te_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], np.log(svec.T[5][i_min:i_max]))
#    s_aux = np.linspace(0.0, s_max, 1000)
#    fig = plt.figure(1)
#    ax = fig.add_subplot(111)
#    ax2 = ax.twinx()
#    ax.plot(svec.T[0][i_min:i_max], svec.T[4][i_min:i_max], "+")
#    ax.plot(s_aux, np.exp(ne_spl(s_aux)), "-")
#    ax2.plot(svec.T[0][i_min:i_max], svec.T[5][i_min:i_max], "*")
#    ax2.plot(s_aux, np.exp(Te_spl(s_aux)), "--")
#    plt.show()
    theta_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[6][i_min:i_max])
    omega_c_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[-1][i_min:i_max] * np.pi)
    omega = np.loadtxt(os.path.join(folder, "f_ECE.dat"))[ch - 1] * 2.0 * np.pi
    # svec.T[8] freq_2X
    # svec.T[4] ne
    # svec.T[5] Te
    rhop_vec_B_min = np.linspace(0.0, 1.2, 200)
    if(dist == "ext"):
        f_inter = f_interpolator(folder, dist="Re")
        B_min_vec = make_B_min(shot, time, rhop_vec_B_min, diag=eq_diag)
        B_min_spline = InterpolatedUnivariateSpline(rhop_vec_B_min, B_min_vec)
        rad_transp_obj = rad_transp_data(rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega, \
                                         dist="ext", store_results=False, f_interpolator=f_inter, \
                                         B_min_spline=B_min_spline)
    elif(dist == "gene" or dist == "gene0"):
        if(dist == "gene0"):
            f_inter = f_interpolator(folder, dist="Ge0")
        else:
            f_inter = f_interpolator(folder, dist="Ge")
        B0 = f_inter.B0
        rad_transp_obj = rad_transp_data(rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega, dist="gene", \
                                         store_results=False, f_interpolator=f_inter, B0=B0)
    else:
        rad_transp_obj = rad_transp_data(rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega)
    Y_res_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[8][i_min:i_max] * 2 * np.pi / omega - 1)
    s_difficult = Y_res_spl.roots()
    print("s_res", s_difficult)
    tau, tau_err = quad(alpha, 0, s_max, rad_transp_obj, points=s_difficult)
    print("tau", tau)
    ode_obj = ode(dIds)
    ode_obj.set_integrator("vode", atol=1.e-5, max_step=0.00005, nsteps=100000)  # , ixpr=True, max_hnil =1, dopri5
    ode_obj.set_f_params(rad_transp_obj)
    ode_obj.set_initial_value(0, 0)
    Trad = ode_obj.integrate(s_max)
#    s_step = 0.01
#    while(s_step < s_max):
#        Trad = ode_obj.integrate(s_step)
#        ode_obj.set_initial_value(Trad, s_step)
#        omega_c = omega_c_spl(s_step)
#        if(omega_c / omega > 0.48 and omega_c / omega < 0.52):
#            s_step += 0.0002e0
#        else:
#            s_step += 0.002e0
#        print("s, Trad(s)", s_step, Trad)
    print("Trad", Trad)

def make_j_alpha_along_s(folder, shot, time, ch, dist, eq_diag="EQH"):
    svec = np.loadtxt(os.path.join(folder, "ecfm_data", "chdata{0:03d}.dat".format(ch)))
    i_min = np.where(svec.T[0][svec.T[3] != -1])[0][0]
    i_max = np.where(svec.T[0][svec.T[3] != -1])[0][-1]
    s_max = svec.T[0][i_max]
    s_min = svec.T[0][i_min]
    svec.T[0] -= s_min
    rhop_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[3][i_min:i_max])
    ne_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], np.log(svec.T[4][i_min:i_max]))
    Te_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], np.log(svec.T[5][i_min:i_max]))
    theta_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[6][i_min:i_max])
    omega_c_spl = InterpolatedUnivariateSpline(svec.T[0][i_min:i_max], svec.T[-1][i_min:i_max] * np.pi)
    omega = np.loadtxt(os.path.join(folder, "ecfm_data", "f_ECE.dat"))[ch - 1] * 2.0 * np.pi
    # svec.T[8] freq_2X
    # svec.T[4] ne
    # svec.T[5] Te
    rhop_vec_B_min = np.linspace(0.0, 1.2, 200)
    if(dist == "ext"):
        f_inter = f_interpolator(folder, dist="Re")
        B_min_vec = make_B_min(shot, time, rhop_vec_B_min, diag=eq_diag)
        B_min_spline = InterpolatedUnivariateSpline(rhop_vec_B_min, B_min_vec)
        rad_transp_obj = rad_transp_data(rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega, \
                                         dist="ext", store_results=True, f_interpolator=f_inter, \
                                         B_min_spline=B_min_spline)
    elif(dist == "gene" or dist == "gene0"):
        if(dist == "gene0"):
            f_inter = f_interpolator(folder, dist="Ge0")
        else:
            f_inter = f_interpolator(folder, dist="Ge")
        B0 = f_inter.B0
        rad_transp_obj = rad_transp_data(rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega, dist="gene", \
                                         store_results=True, f_interpolator=f_inter, B0=B0)
    else:
        rad_transp_obj = rad_transp_data(rhop_spl, ne_spl, Te_spl, theta_spl, omega_c_spl, omega, "thermal", True)
    N = 5000
    s_aux = np.linspace(0.0, s_max, N)
    rhop = np.zeros(N)
    alpha = np.zeros(N)
    j = np.zeros(N)
    for i_s in range(len(s_aux)):
        svec = rad_transp_obj.get_s_vec(s_aux[i_s])
        alpha[i_s], j[i_s] = rad_transp_obj.get_j_alpha(s_aux[i_s])
        rhop[i_s] = svec.rhop
    return rhop, alpha, j * cnst.c ** 2 / ((rad_transp_obj.omega / (2.0 * np.pi)) ** 2 * cnst.e)


def compare_Ibb(folder_list, title_list, shot, time, channel_list, dist):
    for ch in channel_list:
        print("Working on channel ", ch)
        for i_folder in range(len(folder_list)):
            folder = folder_list[i_folder]
            print("Working on folder ", folder)
            title = title_list[i_folder]
            rhop, alpha, j = make_j_alpha_along_s(folder, shot, time, ch, dist, eq_diag="EQH")
            plt.plot(rhop[alpha > 20.0], j[alpha > 20.0] / alpha[alpha > 20.0], "+", label=title + r" ch. {0:d}".format(ch))
    rhop, beta_par, mu_norm, f, B0 = load_f_from_ASCII(os.path.join(folder_list[0], "ecfm_data", "fGe"), Gene=True)
    f = np.exp(f)
    ne = 1.0  # already taken into account
    Te_perp, Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f, 605.0, ne, B0, slices=1)
    plt.plot(rhop, Te_perp, label=r"$T_{\mathrm{e},\perp}$")
    plt.legend(ncol=3)
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.gca().set_ylabel(r"$T_\mathrm{rad}$")
    plt.show()

#    fig = plt.figure(1)
#    ax = fig.add_subplot(111)
#    ax2 = ax.twinx()
# #    ax.plot(svec.T[0][i_min:i_max], svec.T[4][i_min:i_max], "+")
# #    ax.plot(s_aux, np.exp(ne_spl(s_aux)), "-")
# #    ax2.plot(svec.T[0][i_min:i_max], svec.T[5][i_min:i_max], "*")
# #    ax2.plot(s_aux, np.exp(Te_spl(s_aux)), "--")
#    ax.plot(s_aux, np.array(rad_transp_obj.j) * cnst.c ** 2 / ((omega / 2.0 * np.pi) ** 2 * cnst.e), "-")
#    ax2.plot(s_aux, np.array(rad_transp_obj.abs), "--")
#    plt.show()


# solve_rad_transp("/ptmp1/work/sdenk/nssf/33514/2.32/OERT/ed_1/ecfm_data/", 12)
# solve_rad_transp("/ptmp1/work/sdenk/nssf/33697/1.68/OERT/ed_17/ecfm_data/", 12)
# Trad = 967.07265419
# tau = 5.567843432956883
# From model:
# Trad = 967.05347 -> 20 meV deviation -> 0.002 % error
# tau = 5.5678735 -> 1.e-5 -> 0.0002 % error

compare_Ibb(["/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/", \
             "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_13/"], \
             ["GENE", "rel. BiMaxwellian"], 33585, 3.0, [4, 6, 7], "gene")
