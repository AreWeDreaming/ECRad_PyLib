'''
Created on Mar 23, 2016

@author: sdenk
'''
from scipy.interpolate import InterpolatedUnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from plotting_configuration import *
import numpy as np
import os
from electron_distribution_utils import read_svec_dict_from_file, f_interpolator, get_B_min_from_file, \
                                        read_waves_mat_to_beam, read_dist_mat_to_beam, \
                                        read_ray_dict_from_file, Juettner2D, load_f_from_mat
from equilibrium_utils_AUG import EQData
from equilibrium_utils import EQDataExt
from em_Albajar import em_abs_Alb, distribution_interpolator, s_vec, N_with_pol_vec
import scipy.constants as cnst
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.io import loadmat, savemat
import scipy.odr as odr
from ECRad_Results import ECRadResults

def func(beta, x):
    return beta[0] * np.exp(-(x - beta[1]) ** 2 / beta[2] ** 2)

def make_f_inter(dist, working_dir=None, dist_obj=None, EQObj=None, time=None):
    if(dist in ["Re", "ReComp", "Ge", "GeComp"]):
        res_dist = dist.replace("Comp", "")
    else:
        res_dist = "thermal"
    if(working_dir is not None):
        f_inter = f_interpolator(working_dir=working_dir, dist=res_dist)
    else:
        rhop_Bmin, Bmin = EQObj.get_B_min(time, dist_obj.rhop, append_B_ax=True)
        f_inter = f_interpolator(dist_obj=dist_obj, dist=res_dist, rhop_Bmin=rhop_Bmin, Bmin=Bmin)
    return f_inter


def make_3DBDOP_for_ray(result, time, ch, ir, m, B_ax, f_inter=None):
    dist = result.Config.dstf
    svec, freq, Trad, T = load_data_for_3DBDOP(time, dist, ch, ir=ir, Results=result)
    return BDOP_3D(svec, freq, Trad, T, f_inter, dist, B_ax, m=m, only_contribution=True, steps=50)
    

def load_data_for_3DBDOP(time, dist, ch, ir=1, working_dir=None, Results=None):
    if(working_dir is not None):
        svec, freq = read_svec_dict_from_file(os.path.join(working_dir, "ecfm_data"), ch)
        if(dist in ["Re", "ReComp"]):
            ich_folder = "IchRe"
            rhop_Trad, Trad, tau = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_RELAX.dat"), unpack=True)
        elif(dist in ["Ge", "GeComp"]):
            ich_folder = "IchGe"
            rhop_Trad, Trad, tau = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_GENE.dat"), unpack=True)
        elif(dist == "TB"):
            ich_folder = "IchTB"
            rhop_Trad, Trad, tau = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_therm.dat"), unpack=True)
        elif(dist == "ReTh"):
            ich_folder = "IchRe"
            rhop_Trad, Trad, tau = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_therm.dat"), unpack=True)
        else:
            print("dist not supported", dist)
        filename_transparency = os.path.join(working_dir, "ecfm_data", ich_folder, "Trhopch" + "{0:0>3}.dat".format(ch))
        filename_transparency = os.path.join(working_dir, "ecfm_data", ich_folder, "Trhopch" + "{0:0>3}.dat".format(ch))
        T_data = np.loadtxt(filename_transparency)
        if(dist == "ReTh"):
            T = T_data.T[2]
        else:
            T = T_data.T[1]
    else:
        ich = ch - 1
        ray_index = ir - 1
        itime_Scenario = np.argmin(np.abs(Results.Scenario.plasma_dict["time"] - time))
        freq = Results.Scenario.ray_launch[itime_Scenario]["f"][ich]
        itime = np.argmin(np.abs(Results.time - time))
        svec = {}
        svec["rhop"] = Results.ray["rhopX"][itime][ich][ray_index]
        svec["s"] = Results.ray["sX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["R"] = np.sqrt(Results.ray["xX"][itime][ich][ray_index][svec["rhop"] != -1.0] ** 2 + \
                            Results.ray["yX"][itime][ich][ray_index][svec["rhop"] != -1.0] ** 2)
        svec["z"] = Results.ray["zX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["Te"] = Results.ray["TeX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["theta"] = Results.ray["thetaX"][itime][ich][ray_index][svec["rhop"] != -1.0]

        svec["freq_2X"] = Results.ray["YX"][itime][ich][ray_index][svec["rhop"] != -1.0] * 2.0 * freq
        svec["N_abs"] = Results.ray["NcX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["rhop"] = svec["rhop"][svec["rhop"] != -1.0]
        ne_spl = InterpolatedUnivariateSpline(Results.Scenario.plasma_dict["rhop_prof"][itime_Scenario], \
                                              np.log(Results.Scenario.plasma_dict["ne"][itime_Scenario]), ext=3)
        svec["ne"] = np.exp(ne_spl(svec["rhop"]))
        if(dist != "ReTh"):
            Trad = Results.Trad[itime][ich]
            if(len(Results.ray["TX"]) == 0):
                T = np.zeros(len(svec["s"]))
                T[:] = 1.0
                print("WARNING THERE IS NO TRANSMIVITY DATA AVAILABLE")
                print("THE PLOTS ARE ONLY USEFUL FOR DEBUGGING!!!!!!!!!")
            else:
                T = Results.ray["TX"][itime][ich][ray_index]
        else:
            T = Results.ray["T_secondX"][itime][ich][ray_index]
    svec["ne"][svec["ne"] < 1.e15] = 1.e15
    svec["Te"][svec["Te"] < 2.e-2] = 2.e-2
    return svec, freq, Trad, T

class BDOP_3D:
    def __init__(self, svec, freq, Trad, T, f_inter, dist, B_ax, m=2, only_contribution=False, steps=2000, s_important=[]):
        rhop_max = 1.02
        u_par_max = 2.0
        if(dist in ["Re", "ReComp"]):
            dist_mode = "ext"
        elif(dist in ["Ge", "GeComp"]):
            dist_mode = "gene"
        else:
            dist_mode = "thermal"
        em_abs_Alb_obj = em_abs_Alb()
        em_abs_Alb_obj.dist_mode = dist_mode
        self.f_inter = f_inter
        if(dist == "Ge"):
            B0 = self.f_inter.B0
        stride = 1
        if(only_contribution):
            for key in svec.keys():
                svec[key] = svec[key][np.logical_and(T >= np.min(T) + (np.max(T) - np.min(T)) * 1.e-6, \
                                           T <= np.max(T) - (np.max(T) - np.min(T)) * 1.e-6)]
            T = T[np.logical_and(T >= np.min(T) + (np.max(T) - np.min(T)) * 1.e-6, \
                                      T <= np.max(T) - (np.max(T) - np.min(T)) * 1.e-6)]
        T = T[svec["rhop"] != -1.0]
        for key in svec.keys():
            if(key != "rhop"):
                svec[key] = svec[key][svec["rhop"] != -1.0]
        svec["rhop"] = svec["rhop"][svec["rhop"] != -1.0]
        T = T[svec["rhop"] < rhop_max]
        for key in svec.keys():
            if(key != "rhop"):
                svec[key] = svec[key][svec["rhop"] < rhop_max]
        svec["rhop"] = svec["rhop"][svec["rhop"] < rhop_max]
        self.s = []
        self.R = []
        self.rho = []
        self.u_perp = []
        self.u_par = []
        self.val = []
        self.j = []
        self.f = []
        self.zeta = []
        self.u_len = 0
        self.val_back = []
        self.f_back = []
        self.u_par_range = [np.inf, -np.inf]
        self.u_perp_range = [0.0, -np.inf]
        if(dist == "Ge"):
            f_inter_scnd = f_interpolator(working_dir, dist="Ge0")
        else:
            f_inter_scnd = f_interpolator(None, dist="thermal")
        self.u_par_max = -np.inf
        self.u_perp_max = -np.inf
        R_spl = InterpolatedUnivariateSpline(svec["s"], svec["R"])
        rhop_spl = InterpolatedUnivariateSpline(svec["s"], svec["rhop"])
        Te_spl = InterpolatedUnivariateSpline(svec["s"], np.log(svec["Te"]))
        ne_spl = InterpolatedUnivariateSpline(svec["s"], np.log(svec["ne"]))
        freq_2X_spl = InterpolatedUnivariateSpline(svec["s"], svec["freq_2X"])
        theta_spl = InterpolatedUnivariateSpline(svec["s"], svec["theta"])
        T_spl = InterpolatedUnivariateSpline(svec["s"], T)
        s_initial = np.linspace(np.min(svec["s"]), np.max(svec["s"]), steps)
        s = np.copy(s_initial)
        s = np.concatenate([s, s_important])
        s = np.sort(s)
        for i in range(len(s)):
            R = R_spl(s[i])
            rhop = rhop_spl(s[i])
            Te = np.exp(Te_spl(s[i]))
            ne = np.exp(ne_spl(s[i]))
            freq_2X = freq_2X_spl(s[i])
            theta = theta_spl(s[i])
            T_cur = T_spl(s[i])
            if("Re" == dist or "Lu" == dist):
                if(self.f_inter.B_min_spline(rhop).item() == 0.0):
                    self.zeta.append(1.0)
                else:
                    self.zeta.append(np.pi * freq_2X * cnst.m_e / (cnst.e * self.f_inter.B_min_spline(rhop).item()))
                    if(self.zeta[-1] < 1.0):
                        self.zeta[-1] = 1.0
            elif(dist == "Ge"):
                self.zeta.append(np.pi * freq_2X * cnst.m_e / (cnst.e * B0))
            if(em_abs_Alb_obj.is_resonant(rhop, Te, ne, \
                                     freq_2X, theta, freq, m)):
                x, y, spline = self.f_inter.get_spline(rhop, Te)
                dist_inter_slice = distribution_interpolator(x, y, spline)
                if(dist == "Ge"):
                    em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                             freq_2X, theta, freq, dist_inter_slice, B0, m=m)
                elif("Re" == dist or "Lu" == dist):
                    em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                             freq_2X, theta, freq, dist_inter_slice, \
                                             self.f_inter.B_min_spline(rhop).item(), m=m)
                else:
                    em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                             freq_2X, theta, freq)
                if(np.any(np.isnan(em_abs_Alb_obj.j))):
                    print("nan in em")
                    print(em_abs_Alb_obj.u_par, em_abs_Alb_obj.u_perp)
                    # plt.plot(em_abs_Alb_obj.u_par, em_abs_Alb_obj.j)
                    # plt.show()
                else:
                    if(self.u_len == 0):
                        self.u_len = len(em_abs_Alb_obj.u_par)
                    s_cur = np.zeros(self.u_len)
                    s_cur[:] = s[i]
                    self.s.append(s_cur)
                    R_cur = np.zeros(self.u_len)
                    R_cur[:] = R
                    self.R.append(R_cur[:])
                    rho_cur = np.zeros(self.u_len)
                    rho_cur[:] = rhop
                    if(B_ax < freq_2X * np.pi * cnst.m_e / cnst.e):
                        self.rho.append(-rho_cur)
                    else:
                        self.rho.append(rho_cur)
                    u_par = em_abs_Alb_obj.u_par
                    u_perp = em_abs_Alb_obj.u_perp
                    cur_val = em_abs_Alb_obj.j * T_cur
                    cur_j = em_abs_Alb_obj.j
                    if(np.any(np.abs(u_par) > u_par_max)):
                        u_par_cut = np.linspace(max(np.min(u_par), -u_par_max), min(np.max(u_par), u_par_max), len(u_par))
                        u_perp_spl = InterpolatedUnivariateSpline(u_par, u_perp)
                        cur_val_spl = InterpolatedUnivariateSpline(u_par, cur_val)
                        cur_j_spl = InterpolatedUnivariateSpline(u_par, cur_j)
                        self.u_par.append(u_par_cut)
                        self.u_perp.append(u_perp_spl(u_par_cut))
                        self.val.append(cur_val_spl(u_par_cut))
                        self.j.append(cur_j_spl(u_par_cut))
                    else:
                        self.u_par.append(em_abs_Alb_obj.u_par)
                        self.u_perp.append(em_abs_Alb_obj.u_perp)
                        self.val.append(em_abs_Alb_obj.j * T_cur)
                        self.j.append(em_abs_Alb_obj.j)
                    if(np.min(self.u_par[-1]) < self.u_par_range[0]):
                        self.u_par_range[0] = np.min(self.u_par[-1])
                    if(np.max(self.u_par[-1]) > self.u_par_range[1]):
                        self.u_par_range[1] = np.max(self.u_par[-1])
                    if(np.max(self.u_perp[-1]) > self.u_perp_range[1]):
                        self.u_perp_range[1] = np.max(self.u_perp[-1])
                    cur_svec = s_vec(rhop, Te, ne, \
                                     freq_2X, theta)
                    mu = cnst.c ** 2 * cnst.m_e / (Te * cnst.e)
                    self.f.append(em_abs_Alb_obj.dist(self.u_par[-1], self.u_perp[-1], mu, cur_svec))
                    u, pitch, spline = f_inter_scnd.get_spline(rhop, Te)
                    dist_inter_slice = distribution_interpolator(u, pitch, spline)
                    if(dist == "Ge"):
                        em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                                 freq_2X, theta, freq, dist_inter_slice, B0)
                    else:
                        em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                                 freq_2X, theta, freq)
                    self.val_back.append(em_abs_Alb_obj.j * T_cur)
                    self.f_back.append(em_abs_Alb_obj.dist(self.u_par[-1], self.u_perp[-1], mu, cur_svec))
            else:
                print("Channel not resonant at s = ", s[i])
            # plt.plot(u_par[-1], val[-1])
            # plt.show()
            print("point ", i + 1, " of ", len(s), " finished!")
        self.s = np.array(self.s)
        self.R = np.array(self.R)
        self.rho = np.array(self.rho)
        self.u_par = np.array(self.u_par)
        self.u_perp = np.array(self.u_perp)
        self.val = np.array(self.val)
        self.f = np.array(self.f)
        self.log10_f = self.f
        self.log10_f[self.log10_f < 1.e-20] = 1.e-20
        self.log10_f = np.log10(self.log10_f)
        I_norm = Trad / (cnst.c ** 2 / (freq ** 2 * cnst.e)) * 1.e3
        self.val /= I_norm
        self.j = np.array(self.j)
        self.val_back = np.array(self.val_back) / I_norm
        self.f_back = np.array(self.f_back)

class PowerDepo_3D:
    def __init__(self, freq, ray, f_inter, shot, time, dist, B_ax, EqSlice, Te_spl, ne_spl, m=2, only_contribution=False, steps=2000, s_important=[]):
        if(dist in ["Re", "ReComp"]):
            dist_mode = "ext"
            res_dist = dist.replace("Comp", "")
        elif(dist == "Ge"):
            dist_mode = "gene"
            res_dist = dist
        elif(dist == "TB"):
            dist_mode = "thermal"
            res_dist = "thermal"
        elif(dist == "ReTh"):
            dist_mode = "thermal"
            res_dist = "thermal"
        else:
            print("dist not supported", dist)
        self.m = m
        self.only_contribution = only_contribution
        self.f_inter = f_inter
        self.u_par_limit = 2.0
        self.freq = freq
        self.B_ax = B_ax
        # svec.T[8] freq_2X
        # svec.T[4] ne
        # svec.T[5] Te
        self.em_abs_Alb_obj = em_abs_Alb()
        self.em_abs_Alb_obj.dist_mode = dist_mode
        if(dist == "Ge"):
            self.B0 = self.f_inter.B0
        self.stride = 1
        self.s = []
        self.R = []
        self.rho = []
        self.u_perp = []
        self.u_par = []
        self.u_par_range = [np.inf, -np.inf]
        self.u_perp_range = [0.0, -np.inf]
        self.val = []
        self.abs = []
        self.f = []
        self.zeta = []
        self.u_len = 0
        self.val_back = []
        self.f_back = []
        self.dist = dist
        if(dist == "Ge"):
            self.f_inter_scnd = f_interpolator(working_dir, dist="Ge0")
        else:
            self.f_inter_scnd = f_interpolator(None, dist="thermal")
        self.u_par_min = np.Inf
        self.u_par_max = -np.inf
        self.u_perp_max = -np.inf
        self.R_spl = InterpolatedUnivariateSpline(ray["s"], ray["R"])
        self.phi_spl = InterpolatedUnivariateSpline(ray["s"], ray["phi"])
        self.z_spl = InterpolatedUnivariateSpline(ray["s"], ray["z"])
        self.rhop_spl = InterpolatedUnivariateSpline(ray["s"], ray["rhop"])
        self.ne_spl = ne_spl
        self.Te_spl = Te_spl
        self.freq_2X_spl = InterpolatedUnivariateSpline(ray["s"], ray["omega_c"] / np.pi)
        self.N_par_spl = InterpolatedUnivariateSpline(ray["s"], ray["Npar"])
        self.s_init = np.linspace(np.min(ray["s"]), np.max(ray["s"]), steps)
        self.B_r_spl = RectBivariateSpline(EqSlice.R, EqSlice.z, EqSlice.Br)
        self.B_t_spl = RectBivariateSpline(EqSlice.R, EqSlice.z, EqSlice.Bt)
        self.B_z_spl = RectBivariateSpline(EqSlice.R, EqSlice.z, EqSlice.Bz)
        self.B = np.zeros(3)
        self.k = np.zeros(3)
        R = self.R_spl(self.s_init[0])
        phi = self.phi_spl(self.s_init[0])
        self.z_old = self.z_spl(self.s_init[0])
        self.x_old = R * np.cos(phi)
        self.y_old = R * np.sin(phi)
        self.P = 1.0
        s = np.copy(self.s_init)
        s = np.concatenate([s, s_important])
        s = np.sort(s)
        for i in range(1, len(s)):
            ds = s[i] - s[i - 1]
            if(self.only_contribution):
                if(self.P < 1.e-5):
                    break
            R = self.R_spl(s[i])
            phi = self.phi_spl(s[i])
            x = R * np.cos(phi)
            y = R * np.sin(phi)
            z = self.z_spl(s[i])
            self.k[0] = x - self.x_old
            self.k[1] = y - self.y_old
            self.k[2] = z - self.z_old
            self.k[:] /= np.linalg.norm(self.k)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            Br = self.B_r_spl(R, z, grid=False)
            Bt = self.B_t_spl(R, z, grid=False)
            self.B[2] = self.B_z_spl(R, z, grid=False)
            self.B[0] = cos_phi * Br - sin_phi * Bt
            self.B[1] = sin_phi * Br + cos_phi * Bt
            self.B[:] /= np.linalg.norm(self.B)
            theta = np.arccos(np.dot(self.k, self.B))
            rhop = self.rhop_spl(s[i])
            Te = np.exp(self.Te_spl(rhop))
            ne = np.exp(self.ne_spl(rhop))
            freq_2X = self.freq_2X_spl(s[i])
            omega_p = cnst.e * np.sqrt(ne / (cnst.epsilon_0 * cnst.m_e))
            X = omega_p ** 2 / (2.0 * np.pi * self.freq) ** 2
            N, e = N_with_pol_vec(X, freq_2X / (2.0 * self.freq), np.sin(theta), np.cos(theta), 1)
            N_par = self.N_par_spl(s[i])
            print("N_abs, N_par in situ, N_par Gray", N, np.cos(theta) * N, N_par)
            if(self.em_abs_Alb_obj.is_resonant(rhop, Te, ne, \
                                     freq_2X, theta, self.freq, self.m)):
                x, y, spline = self.f_inter.get_spline(rhop, Te)
                dist_inter_slice = distribution_interpolator(x, y, spline)
                if(self.f_inter.B_min_spline(rhop).item() == 0.0):
                    self.zeta.append(1.0)
                else:
                    self.zeta.append(np.pi * freq_2X * cnst.m_e / (cnst.e * self.f_inter.B_min_spline(rhop).item()))
                    if(self.zeta[-1] < 1.0):
                        self.zeta[-1] = 1.0
                if(self.dist == "Ge"):
                    self.em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                         freq_2X, theta, self.freq, dist_inter_slice, self.B0, m=self.m)
                elif("Re" == self.dist or "Lu" == self.dist):
                    self.em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                         freq_2X, theta, self.freq, dist_inter_slice, \
                                         self.f_inter.B_min_spline(rhop).item(), m=self.m)
                else:
                    self.em_abs_Alb_obj.j_abs_Alb(rhop, Te, ne, \
                                         freq_2X, theta, self.freq)
                if(np.any(np.isnan(self.em_abs_Alb_obj.j))):
                    print("nan in em")
                    print(self.em_abs_Alb_obj.u_par, self.em_abs_Alb_obj.u_perp)
                    # plt.plot(em_abs_Alb_obj.u_par, em_abs_Alb_obj.j)
                    # plt.show()
                elif(len(self.em_abs_Alb_obj.u_par[np.abs(self.em_abs_Alb_obj.u_par) < self.u_par_limit]) < 2):
                    continue
                else:
                    if(self.u_len == 0):
                        self.u_len = len(self.em_abs_Alb_obj.u_par)
                    s_cur = np.zeros(self.u_len)
                    s_cur[:] = s[i]
                    self.s.append(s_cur)
                    R_cur = np.zeros(self.u_len)
                    R_cur[:] = R
                    self.R.append(R_cur[:])
                    rho_cur = np.zeros(self.u_len)
                    rho_cur[:] = rhop
                    if(self.B_ax < freq_2X * np.pi * cnst.m_e / cnst.e):
                        self.rho.append(-rho_cur)
                    else:
                        self.rho.append(rho_cur)
                    u_par = np.copy(self.em_abs_Alb_obj.u_par)
                    u_perp = np.copy(self.em_abs_Alb_obj.u_perp)
                    cur_val = np.copy(self.em_abs_Alb_obj.c_abs * ds * self.P)
                    cur_abs = np.copy(self.em_abs_Alb_obj.c_abs)
                    if(np.any(np.abs(u_par) > self.u_par_limit)):
                        u_par_cut = np.linspace(max(np.min(u_par), -self.u_par_limit), min(np.max(u_par), self.u_par_limit), len(u_par))
                        u_perp_spl = InterpolatedUnivariateSpline(u_par, u_perp)
                        cur_val_spl = InterpolatedUnivariateSpline(u_par, cur_val)
                        cur_abs_spl = InterpolatedUnivariateSpline(u_par, cur_abs)
                        self.u_par.append(u_par_cut)
                        self.u_perp.append(u_perp_spl(u_par_cut))
                        self.val.append(cur_val_spl(u_par_cut))
                        self.abs.append(cur_abs_spl(u_par_cut))
                    else:
                        self.u_par.append(u_par)
                        self.u_perp.append(u_perp)
                        self.val.append(cur_val)
                        self.abs.append(cur_abs)
                    if(np.min(self.u_par[-1]) < self.u_par_range[0]):
                        self.u_par_range[0] = np.min(self.u_par[-1])
                    if(np.max(self.u_par[-1]) > self.u_par_range[1]):
                        self.u_par_range[1] = np.max(self.u_par[-1])
                    if(np.max(self.u_perp[-1]) > self.u_perp_range[1]):
                        self.u_perp_range[1] = np.max(self.u_perp[-1])
                    cur_svec = s_vec(rhop, Te, ne, \
                                 freq_2X, theta)
                    mu = cnst.c ** 2 * cnst.m_e / (Te * cnst.e)
                    c_abs, j = self.em_abs_Alb_obj.abs_Albajar(cur_svec, 2.0 * np.pi * self.freq, 1, n_max=3, n_min=2)
                    self.P -= c_abs * ds * self.P
                    self.f.append(self.em_abs_Alb_obj.dist(self.u_par[-1], self.u_perp[-1], mu, cur_svec))
                    u, pitch, spline = self.f_inter_scnd.get_spline(rhop, Te)
                    dist_inter_slice = distribution_interpolator(u, pitch, spline)
                    print("P", self.P, "c_abs", c_abs)
            else:
                print("Channel not resonant at rhop = ", rhop)
            # plt.plot(u_par[-1], val[-1])
            # plt.show()
            print("point ", i + 1, " of ", len(s), " finished!")
        self.s = np.array(self.s)
        self.R = np.array(self.R)
        self.rho = np.array(self.rho)
        self.u_par = np.array(self.u_par)
        self.u_perp = np.array(self.u_perp)
        self.val = np.array(self.val)
        self.val[self.val < 0] = 0.0
        self.f = np.array(self.f)
        self.log10_f = self.f
        self.log10_f[self.log10_f < 1.e-20] = 1.e-20
        self.log10_f = np.log10(self.log10_f)
        self.val
        self.abs = np.array(self.abs)


def make_3DBDOP(working_dir, shot, time, ch_list, m_list, dist, diag="EQH", title="GENE", include_ECRH=False, only_contribution=False, nocolor=False, \
                ece_alpha=1.0, save_only=False, only_ECRH=False, flat=False, single_Beam=False, ECRH_freq=140.e9):
    fig = plt.figure(figsize=(12.5, 8.5))
#    fig.text(0.05, 0.95, "a)", fontsize=28)
    BDOP_list = []
#    BPD_max = -np.inf
    only_LFS = False
    rho = True
    EQObj = EQData(shot, EQ_diag=diag)
    EqSlice = EQObj.GetSlice(time)
    B_ax = EQObj.get_B_on_axis(time)
    freq = ECRH_freq
    cmaps = []
    alphas = []
    u_par_range = [np.inf, -np.inf]
    u_perp_range = [0, -np.inf]
    mdict = {}
    is_ecrh_list = []
    f_inter = make_f_inter(dist, working_dir=working_dir)
    for ich in range(len(ch_list)):
        svec, freq, Trad, T, = load_data_for_3DBDOP(time, dist, ich, working_dir=working_dir)
        BDOP_list.append(BDOP_3D(svec, freq, Trad, T, f_inter, dist, B_ax, m=m_list[ich], only_contribution=only_contribution))
        m = cm.ScalarMappable(cmap=plt.cm.get_cmap("viridis"))
        m.set_array(np.linspace(0.0, 1.0, 20))
        cmaps.append(m)
        alphas.append(ece_alpha)
        is_ecrh_list.append(0)
    if(include_ECRH):
        waves_mat = loadmat(os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/DistData", "GRAY_rays_{0:d}_{1:1.2f}.mat".format(shot, time)))
        linear_beam = read_waves_mat_to_beam(waves_mat, EqSlice)
        if(single_Beam):
            linear_beam.rays = linear_beam.rays[1:2]
        for i in range(len(linear_beam.rays)):
            BDOP_list.append(PowerDepo_3D(freq, linear_beam.rays[i][0], working_dir, shot, time, dist, B_ax, EqSlice, m=m_list[0], only_contribution=only_contribution))
            m = cm.ScalarMappable(cmap=plt.cm.get_cmap("hot"))
            m.set_array(np.linspace(0.0, 1.0, 20))
            cmaps.append(m)
            alphas.append(0.3)
            is_ecrh_list.append(1)
#        if(BPD_max < np.max(BDOP_list[-1].val)):
#            i_max = np.argmax(BDOP_list[-1].val)
#            i_x_max, i_u_par_max = np.unravel_index(i_max, BDOP_list[-1].val.shape)
#            x_max = BDOP_list[-1].rho[i_x_max, i_u_par_max]
#            u_perp_max = BDOP_list[-1].u_perp[i_x_max, i_u_par_max]
#            u_par_max = BDOP_list[-1].u_par[i_x_max, i_u_par_max]
#            BPD_max = np.max(BDOP_list[-1].val)
#    if(dist == "Ge" or dist == "ReComp"):
#        BPD_max = np.max(BDOP_val_list - np.min(BDOP_val_list))
#        BDOP_val_list = (BDOP_val_list - np.min(BDOP_val_list)) / BPD_max
#    else:
#        BPD_max = np.max(BDOP_val_list)
#        BDOP_val_list /= BPD_max
    # plt.figure(figsize=(12.5, 8.5))
    if(not flat):
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()
    if(title is not None):
        if("Comp" in dist):
            fig.suptitle(r"BPD of perturbations (" + title + ")")
        else:
            fig.suptitle(r"BPD for " + title + " distribution")
    mdict["BPD_vals"] = []
    mdict["BPD_rho"] = []
    mdict["BPD_u_par"] = []
    mdict["BPD_u_perp"] = []
    mdict["BPD_facecolors"] = []
    mdict["is_ecrh_list"] = []
    for BDOP, cmap, alpha, ECRH_select in zip(BDOP_list, cmaps, alphas, is_ecrh_list):
    #        fig.suptitle(r"BDOP for " + title + " distribution")
    #        surf = ax.plot_surface(BDOP.rho, BDOP.u_par, BDOP.u_perp, rstride=1, cstride=1, \
    #                           facecolors=plt.cm.get_cmap("plasma")(BDOP.val / np.max(BDOP.val)), \
    #                           linewidth=0, antialiased=True, shade=True)
#            print(BDOP.rho.shape, BDOP.rho[BDOP.rho > 0].shape)
        if(only_ECRH and not ECRH_select):
            continue
        if(rho):
            LFS_mask = BDOP.rho.T[0] > 0
            HFS_mask = BDOP.rho.T[0] < 0
            if(len(BDOP.rho[BDOP.rho.T[0] < 0]) > 2 and not only_LFS):
                BPD_max_ch = np.max(BDOP.val)
                if(nocolor):
                    facecolors = cmap.to_rgba(np.zeros(BDOP.val[HFS_mask].shape), alpha)
                elif(dist == "Ge" or dist == "ReComp"):
                    facecolors = cmap.to_rgba((BDOP.val[HFS_mask] - BDOP.val_back[HFS_mask] - np.min(BDOP.val[HFS_mask] - BDOP.val_back[HFS_mask])) / \
                                            np.max(BDOP.val[HFS_mask] - BDOP.val_back[HFS_mask] - np.min(BDOP.val[HFS_mask] - BDOP.val_back[HFS_mask])), alpha)
                else:
                    facecolors = cmap.to_rgba(BDOP.val[HFS_mask] / BPD_max_ch, alpha)
                if(save_only):
                    mdict["BPD_vals"].append(BDOP.val[HFS_mask] / BPD_max_ch)
                    mdict["BPD_rho"].append(BDOP.rho[HFS_mask])
                    mdict["BPD_u_par"].append(BDOP.u_par[HFS_mask])
                    mdict["BPD_u_perp"].append(BDOP.u_perp[HFS_mask])
                    mdict["BPD_facecolors"].append(facecolors)
                    mdict["is_ecrh_list"].append(ECRH_select)
                else:
                    if(not flat):
                        surf_HFS = ax.plot_surface(BDOP.rho[HFS_mask], BDOP.u_par[HFS_mask], BDOP.u_perp[HFS_mask], rstride=1, cstride=1, \
                                       facecolors=facecolors, \
                                       linewidth=0, antialiased=True, shade=True)
                    else:
                        surf_HFS = ax.contourf(BDOP.rho[HFS_mask], BDOP.u_par[HFS_mask], BDOP.u_perp[HFS_mask], cmap="viridis")
                        cont_HFS = ax.contour(BDOP.rho[HFS_mask], BDOP.u_par[HFS_mask], BDOP.val[HFS_mask] / BPD_max_ch, cmap="inferno", levels=np.linspace(0.0, 1.0, 11))
            else:
                BPD_max_ch = np.max(BDOP.val[LFS_mask])
                print("Skipping HFS -- not enough resonant points or not selected for plotting:", len(BDOP.rho[BDOP.rho < 0]))
            if(len(BDOP.rho[BDOP.rho.T[0] > 0]) > 2):
                if(nocolor):
                    facecolors = cmap.to_rgba(np.zeros(BDOP.val[LFS_mask].shape), alpha)
                elif("Comp" in dist):
                    facecolors = cmap.to_rgba((BDOP.val[LFS_mask] - BDOP.val_back[LFS_mask] - np.min(BDOP.val[LFS_mask] - BDOP.val_back[LFS_mask])) / \
                                            np.max(BDOP.val[LFS_mask] - BDOP.val_back[LFS_mask] - np.min(BDOP.val[LFS_mask] - BDOP.val_back[LFS_mask])), alpha)
                else:
                    facecolors = cmap.to_rgba(BDOP.val[LFS_mask] / BPD_max_ch, alpha)
                if(save_only):
                    mdict["BPD_vals"].append(BDOP.val[LFS_mask] / BPD_max_ch)
                    mdict["BPD_rho"].append(BDOP.rho[LFS_mask])
                    mdict["BPD_u_par"].append(BDOP.u_par[LFS_mask])
                    mdict["BPD_u_perp"].append(BDOP.u_perp[LFS_mask])
                    mdict["BPD_facecolors"].append(facecolors)
                    mdict["is_ecrh_list"].append(ECRH_select)
                else:
                    if(not flat):
                        surf = ax.plot_surface(BDOP.rho[LFS_mask], BDOP.u_par[LFS_mask], BDOP.u_perp[LFS_mask], rstride=1, cstride=1, \
                                   facecolors=facecolors, \
                                   linewidth=0, antialiased=True, shade=True)
                    else:
                        surf_LFS = ax.contourf(BDOP.rho[LFS_mask], BDOP.u_par[LFS_mask], BDOP.u_perp[LFS_mask], cmap="viridis")
                        cont_LFS = ax.contour(BDOP.rho[LFS_mask], BDOP.u_par[LFS_mask], BDOP.val[LFS_mask] / BPD_max_ch, cmap="inferno", levels=np.linspace(0.1, 1.0, 10))
            else:
                print("Skipping LFS -- not enough resonant points:", len(BDOP.rho[BDOP.rho < 0]))
        else:
#            for i in range(len(BDOP.R)):
#                fig2 = plt.figure()
#                plt.plot(BDOP.u_par[i], BDOP.val[i])
#                plt.show()
            BPD_max_ch = np.max(BDOP.val)
            if(nocolor):
                facecolors = cmap.to_rgba(np.zeros(BDOP.val.shape), alpha)
            elif("Comp" in dist):
                facecolors = cmap.to_rgba((BDOP.val - BDOP.val_back - np.min(BDOP.val - BDOP.val_back)) / \
                                        np.max(BDOP.val - BDOP.val_back - np.min(BDOP.val - BDOP.val_back)), alpha)
            else:
                facecolors = cmap.to_rgba(BDOP.val / BPD_max_ch, alpha)
            if(save_only):
                mdict["BPD_vals"].append(BDOP.val / BPD_max_ch)
                mdict["BPD_rho"].append(BDOP.R)
                mdict["BPD_u_par"].append(BDOP.u_par)
                mdict["BPD_u_perp"].append(BDOP.u_perp)
                mdict["BPD_facecolors"].append(facecolors)
                mdict["is_ecrh_list"].append(ECRH_select)
            else:
                if(not flat):
                    surf = ax.plot_surface(BDOP.R, BDOP.u_par, BDOP.u_perp, rstride=1, cstride=1, \
                                   facecolors=facecolors, \
                                   linewidth=0, antialiased=True, shade=True)
                else:
                    surf = ax.contourf(BDOP.R, BDOP.u_par, BDOP.u_perp, cmap="viridis")
                    cont = ax.contour(BDOP.R, BDOP.u_par, BDOP.val / BPD_max_ch, cmap="inferno", levels=np.linspace(0.1, 1.0, 10))
        if(BDOP.u_par_range[0] < u_par_range[0]):
            u_par_range[0] = BDOP.u_par_range[0]
        if(BDOP.u_par_range[1] > u_par_range[1]):
            u_par_range[1] = BDOP.u_par_range[1]
        if(BDOP.u_perp_range[1] > u_perp_range[1]):
            u_perp_range[1] = BDOP.u_perp_range[1]
    if(save_only):
        mdict["is_ecrh_list"] = np.array(mdict["is_ecrh_list"])
        color_axis = np.linspace(0.0, 1.0, 200)
        mdict["color_axis"] = color_axis
        if(include_ECRH):
            iECRH = np.where(mdict["is_ecrh_list"] == 1)[0][0]
            mdict["ECRH_cmap"] = cmaps[iECRH].to_rgba(color_axis)
        iBPD = np.where(mdict["is_ecrh_list"] == 0)[0][0]
        mdict["BPD_color_map"] = cmaps[iBPD].to_rgba(color_axis)
        filename = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/BPD", "BPD2D_{0:d}_{1:1.2f}_{2:d}.mat".format(shot, time, ch_list[0]))
        print(filename)
        savemat(filename, mdict)
        return
    if(not flat):
        ax.w_xaxis.set_major_locator(NLocator(nbins=7, prune='lower'))
        ax.w_xaxis.set_minor_locator(NLocator(nbins=14))
        ax.w_yaxis.set_major_locator(NLocator(nbins=5))
        ax.w_yaxis.set_minor_locator(NLocator(nbins=10))
        ax.w_zaxis.set_major_locator(NLocator(nbins=5))
        ax.w_zaxis.set_minor_locator(NLocator(nbins=10))
    else:
        steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
        ax.get_xaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps))
        ax.get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 2.0))
        ax.get_yaxis().set_major_locator(MaxNLocator(nbins=4, steps=steps))
        ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=8, steps=steps / 2.0))
    if(rho):
        ax.set_xlabel(r"$\rho_\mathrm{pol}$")  #
    else:
        ax.set_xlabel(r"$R$ [m]")  #
    ax.set_ylabel(r"$u_\parallel$")
    if(not flat):
        ax.set_zlabel(r"$u_\perp$")
    ax.set_ylim(u_par_range[0], u_par_range[1])
    if(not flat):
        ax.set_zlim(0, u_perp_range[1])
        ax.view_init(elev=30., azim=10)
    if(not nocolor and not only_ECRH):
        cb = fig.colorbar(cmaps[0])
        if("Comp" in dist):
    #        if(rho):
    #            cb.set_label(r"$D_\omega$")
    #        else:
    #            cb.set_label(r"$D_\omega [\si{{[a.u.]}}]$")
            cb.set_label(r"$D_\omega - D_{\omega,0}[\si{{a.u.}}]$")
    #        cb.set_label(r"$D_\omega - D_{\omega,0}[\si{\per\metre}]$")
        else:
    #        if(rho):
    #            cb.set_label(r"$D_\omega$ [\si{{[a.u.]}}]")
    #        else:
            if(not flat):
                cb.set_label(r"$D_\omega [\si{{a.u.}}]$")
            else:
                cb.set_label(r"$u_\perp$")
                m = cm.ScalarMappable(cmap=plt.cm.get_cmap("inferno"))
                m.set_array(np.linspace(0.0, 1.0, 11))
                cb2 = fig.colorbar(m)
                cb2.set_label(r"$D_\omega [\si{{a.u.}}]$")
    if(include_ECRH):
        if(not flat):
            cb_ECRH = fig.colorbar(cmaps[-1])
            cb_ECRH.set_label(r"$\mathrm{d}P_\mathrm{ECRH}/d\mathrm{s} [\si{{a.u.}}]$")
        else:
            m = cm.ScalarMappable(cmap=plt.cm.get_cmap("viridis"))
            m.set_array(np.linspace(0.0, 1.0, 11))
            cb_ECRH = fig.colorbar(m)
            cb_ECRH.set_label(r"$u_\perp$")
            m = cm.ScalarMappable(cmap=plt.cm.get_cmap("inferno"))
            m.set_array(np.linspace(0.0, 1.0, 11))
            cb_ECRH_2 = fig.colorbar(m)
            cb_ECRH_2.set_label(r"$\mathrm{d}P/\mathrm{d}R [\si{{a.u.}}]$")
    plt.show()
#        cb.set_label(r"$D_\omega$")
#    if(dist == "Ge"):
#        fig2 = plt.figure(figsize=(12.5, 8.5))
#        fig2.suptitle(r"Distribution perturbations (" + title + ")")
#        ax2 = fig2.gca(projection='3d')
#        surf2 = ax2.plot_surface(BDOP.rho, BDOP.u_par, BDOP.u_perp, rstride=1, cstride=1, \
#                               facecolors=plt.cm.get_cmap("plasma")((BDOP.f - BDOP.f_back - np.min(BDOP.f - BDOP.f_back)) / \
#                                                                    np.max(BDOP.f - BDOP.f_back - np.min(BDOP.f - BDOP.f_back))),
#                               linewidth=0, antialiased=True, shade=True)
#        ax2.w_xaxis.set_major_locator(NLocator(nbins=5, prune='lower'))
#        ax2.w_xaxis.set_minor_locator(NLocator(nbins=10))
#        ax2.w_yaxis.set_major_locator(NLocator(nbins=5))
#        ax2.w_yaxis.set_minor_locator(NLocator(nbins=10))
#        ax2.w_zaxis.set_major_locator(NLocator(nbins=5))
#        ax2.w_zaxis.set_minor_locator(NLocator(nbins=10))
#        ax2.set_xlabel(r"$\rho_\mathrm{pol}$")  #
#        ax2.set_ylabel(r"$u_\parallel$")
#        ax2.set_zlabel(r"$u_\perp$")
#        ax2.view_init(elev=30., azim=10)
#        m2 = cm.ScalarMappable(cmap=plt.cm.get_cmap("plasma"))
#        m2.set_array(BDOP.f - BDOP.f_back)
#        cb2 = fig2.colorbar(m2)
#        cb2.set_label(r"$f - f_0$")
#    else:
#        fig2 = plt.figure(figsize=(12.5, 8.5))
#        fig2.suptitle(r"f for " + title + " distribution")
#        ax2 = fig2.gca(projection='3d')
#        surf2 = ax2.plot_surface(BDOP.rho, BDOP.u_par, BDOP.u_perp, rstride=1, cstride=1, \
#                           facecolors=plt.cm.get_cmap("plasma")((BDOP.log10_f - np.min(BDOP.log10_f)) / (np.max(BDOP.log10_f) - np.min(BDOP.log10_f))), \
#                           linewidth=0, antialiased=True, shade=True)
#        ax2.w_xaxis.set_major_locator(NLocator(nbins=5, prune='lower'))
#        ax2.w_xaxis.set_minor_locator(NLocator(nbins=10))
#        ax2.w_yaxis.set_major_locator(NLocator(nbins=5))
#        ax2.w_yaxis.set_minor_locator(NLocator(nbins=10))
#        ax2.w_zaxis.set_major_locator(NLocator(nbins=5))
#        ax2.w_zaxis.set_minor_locator(NLocator(nbins=10))
#        ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
#        ax2.set_ylabel(r"$u_\parallel$")
#        ax2.set_zlabel(r"$u_\perp$")
#        ax2.view_init(elev=30., azim=10)
#        m2 = cm.ScalarMappable(cmap=plt.cm.get_cmap("plasma"))
#        m2.set_array(BDOP.log10_f)
#        cb2 = fig2.colorbar(m2)
#        cb2.set_label(r"$log_{10}(f_\mathrm{" + title + r"})$")
#    print("Maximum of BPD at", x_max, u_perp_max, u_par_max)
#    N = 50
#    u_par_grid = np.linspace(u_par_min, u_par_max, 2 * N)
#    u_perp_grid = np.linspace(0.0, u_perp_max, N)
#    binned_val = np.zeros((2 * N, N))
#    for i in range(len(rho)):
#        val_spl = InterpolatedUnivariateSpline(u_par[i], val[i])
#        for j in range(len(u_perp_grid)):
#            root_spl = InterpolatedUnivariateSpline(u_par[i], u_perp[i] - u_perp_grid[j])
#            for root in root_spl.roots():
#                i_u_par_next = np.argmin(np.abs(u_par_grid - root))
#                binned_val[i_u_par_next, j] += val_spl(root)
#    fig2 = plt.figure(figsize=(12.5, 8.5))
#    ax2 = fig2.add_subplot(111)
#    levels = np.linspace(0, np.max(binned_val), 30)
#    cmap = plt.cm.get_cmap("plasma")  # gnuplot
#    cont1 = ax2.contourf(u_par_grid, u_perp_grid, binned_val.T , levels=levels, cmap=cmap)  # ,norm = LogNorm()
#    cont2 = ax2.contour(u_par_grid, u_perp_grid, binned_val.T, levels=levels, colors='k',
#                        hold='on', alpha=0.25, linewidths=1)
#    ax2.set_xlabel(r"$u_\parallel$")
#    ax2.set_ylabel(r"$u_\perp$")
#    cb = fig2.colorbar(cont1, ax=ax2, ticks=[0, np.max(binned_val), 5])  # ticks = levels[::6] #,
#    cb.set_label(r"$\mathrm{Log}_\mathrm{10}\left(f_\mathrm{" + dist + r"}\right)$")  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
#    # cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
#    cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([1.0, 5.0, 10.0])))
#    cb.ax.minorticks_on()
# #    except ValueError:
#        m = cm.ScalarMappable(cmap=plt.cm.get_cmap("jet"))

def make_3DBDOP_cut(matfilename, shot, time, ch_list, m_list, dist, diag="EQH", include_ECRH=False, \
                    single_Beam=False, m_ECRH_list=[2], only_contribution=False, nocolor=False, \
                    ece_alpha=1.0, save_only=False, use_rhop=True, single_ray_BPD=False, recalc_BPD=False, \
                    preserve_original_BPD=False, Teweight=False, ECRH_freq=140.e9, usemat=True, alternative_mat_for_waves=None):
    if(not usemat):
        working_dir = matfilename
    else:
        working_dir = None
    fig = plt.figure(figsize=(16.5, 8.5))
    fig.text(0.025, 0.95, "a)")
    fig.text(0.55, 0.95, "b)")
    BDOP_list = []
#    BPD_max = -np.inf
    if(usemat):
        Results = ECRadResults()
        if(not Results.from_mat_file(matfilename)):
            print("NO FILE")
            return
        itime = np.argmin(np.abs(Results.Scenario.plasma_dict["time"] - time))
        rhop_Te = Results.Scenario.plasma_dict["rhop_prof"][itime] * Results.Scenario.Te_rhop_scale
        Te = np.log(Results.Scenario.plasma_dict["Te"][itime] * Results.Scenario.Te_scale)  # from IDA always positive definite
        rhop_ne = Results.Scenario.plasma_dict["rhop_prof"][itime] * Results.Scenario.ne_rhop_scale
        ne = np.log(Results.Scenario.plasma_dict["Te"][itime] * Results.Scenario.ne_scale)  # from IDA always positive definite
        EqSlice = Results.Scenario.plasma_dict["eq_data"][itime]
        EQObj = EQDataExt(shot, bt_vac_correction=1.0, Ext_data=True)
        EQObj.insert_slices_from_ext(Results.Scenario.plasma_dict["time"], Results.Scenario.plasma_dict["eq_data"])
        B_ax = EQObj.get_B_on_axis(time)
        R_ax, z_ax = EQObj.get_axis(time)
    else:
        rhop_Te, Te = np.loadtxt(os.path.join(working_dir, "ecfm_data", "Te_file.dat"), skiprows=1, unpack=True)
        Te[Te < 1.e-2] = 1.e-2
        rhop_ne, ne = np.loadtxt(os.path.join(working_dir, "ecfm_data", "Te_file.dat"), skiprows=1, unpack=True)
        ne[ne < 1.e6] = 1.e6
        EQObj = EQData(shot, EQ_diag=diag)
        B_ax = EQObj.get_B_on_axis(time)
        R_ax, z_ax = EQObj.get_axis(time)
    Te_spline = InterpolatedUnivariateSpline(rhop_Te, Te)
    ne_spline = InterpolatedUnivariateSpline(rhop_ne, ne)
    print("Position of magn. axus", R_ax, z_ax)
    if(usemat):
        if(alternative_mat_for_waves is not None):
            waves_dist_mat_filename = alternative_mat_for_waves
        else:
            waves_dist_mat_filename = matfilename
        mat = loadmat(waves_dist_mat_filename, squeeze_me=True)
        if(include_ECRH):
            linear_beam = read_waves_mat_to_beam(mat, EqSlice, use_wave_prefix=True)
            quasi_linear_beam = read_dist_mat_to_beam(mat, use_dist_prefix=True)
        dist_obj = load_f_from_mat(waves_dist_mat_filename, use_dist_prefix=True)
    else:
        dist_obj = load_f_from_mat(os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/DistData", "Dist_{0:d}_{1:1.2f}.mat".format(shot, time)), use_dist_prefix=False)
        if(include_ECRH):
            dist_mat = loadmat(os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/DistData", "Dist_{0:d}_{1:1.2f}.mat".format(shot, time)), squeeze_me=True)
            waves_mat = loadmat(os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/DistData", "GRAY_rays_{0:d}_{1:1.2f}.mat".format(shot, time)), squeeze_me=True)
            linear_beam = read_waves_mat_to_beam(waves_mat, EqSlice)            
            quasi_linear_beam = read_dist_mat_to_beam(dist_mat)
    if(single_Beam):
            linear_beam.rays = linear_beam.rays[1:2]
    freq = ECRH_freq
    cmaps = []
    alphas = []
    u_par_range = [np.inf, -np.inf]
    u_perp_range = [0, -np.inf]
    mdict = {}
    is_ecrh_list = []
    ax_depo = fig.add_subplot(121)
    ch_done_list = []
    R_BPD_dict = {}
    R_BPD_list = []
    s_BPD_list = []
    s_BPD_dict = {}
    rhop_BPD_dict = {}
    distribution_rhop = None
    ECRH_colors = ["magenta", "red"]
    for ich, m_ch in zip(ch_list, m_list):
        if(ich in ch_done_list):
            R_BPD_list.append(R_BPD_dict[str(ich)])
            s_BPD_list.append(s_BPD_dict[str(ich)])
            continue
        else:
            ch_done_list.append(ich)
        ray_list = []
        ray_BPD_spl_list = []
        iray = 2
        if(recalc_BPD):
            if(usemat):
                N_max = Results.Config.N_ray
            else:
                N_max = 1000  # Stop only when no files left
        else:
            N_max = 1
        if(usemat):
            for iray in range(N_max):
                ray_dict = {}
                ray_dict["s"] = Results.ray["sX"][itime][ich][iray]
                ray_dict["x"] = Results.ray["xX"][itime][ich][iray]
                ray_dict["y"] = Results.ray["yX"][itime][ich][iray]
                ray_dict["z"] = Results.ray["zX"][itime][ich][iray]
                ray_dict["rhop"] = Results.ray["rhopX"][itime][ich][iray]
                ray_dict["BPD"] = Results.ray["BPDX"][itime][ich][iray]
                ray_dict["BPD_second"] = Results.ray["BPD_secondX"][itime][ich][iray]
                ray_dict["N_ray"] = Results.ray["NX"][itime][ich][iray]
                ray_dict["N_cold"] = Results.ray["NcX"][itime][ich][iray]
                ray_dict["theta"] = Results.ray["thetaX"][itime][ich][iray]
                spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
                ray_dict["Nx"] = spl.derivative(1)(ray_dict["s"])
                spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
                ray_dict["Ny"] = spl.derivative(1)(ray_dict["s"])
                spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
                ray_dict["Nz"] = spl.derivative(1)(ray_dict["s"])
                norm = ray_dict["N_ray"] / np.sqrt(ray_dict["Nx"] ** 2 + ray_dict["Ny"] ** 2 + ray_dict["Nz"] ** 2)
                ray_dict["Nx"] *= norm
                ray_dict["Ny"] *= norm
                ray_dict["Nz"] *= norm
                ray_list.append(dict(ray_dict))
                ray_BPD_spl_list.append(InterpolatedUnivariateSpline(ray_list[-1]["s"], ray_list[-1]["BPD"]))
        else:
            iray = 1
            while True and iray < N_max + 1:
                try:
                    ray_list.append(read_ray_dict_from_file(os.path.join(working_dir, "ecfm_data"), dist, ich, mode="X", iray=iray))
                    iray += 1
                except IOError:
                    break
                ray_BPD_spl_list.append(InterpolatedUnivariateSpline(ray_list[-1]["s"], ray_list[-1]["BPD"]))
        # recalc_BPD does not seem to be fully implemented. Also the weights of the individual rays is not stored atm
        if(usemat):
            BPD_ch_rhop = Results.BPD["BPDX"][itime][ich]
            BPD_ch = Results.BPD["rhopX"][itime][ich]
        else:
            BPD_ch_data = np.loadtxt(os.path.join(working_dir, "ecfm_data", "Ich" + dist, "BPDX{0:03d}.dat".format(ich)))
            BPD_ch_rhop = np.abs(BPD_ch_data.T[0])
            BPD_ch = BPD_ch_data.T[1]
        s_helper = np.linspace(0.0, 1.0, len(BPD_ch_rhop))
        BPD_ch_spl = InterpolatedUnivariateSpline(s_helper, BPD_ch)
        BPD_ray_dict = ray_list[0]  # Central ray
        rhop_BPD_ray = BPD_ray_dict["rhop"]
        BPD_ray = BPD_ray_dict["BPD"]
        s_ray = BPD_ray_dict["s"]
        R_BPD_ray = np.sqrt(BPD_ray_dict["x"] ** 2 + BPD_ray_dict["y"] ** 2)
        R_spl = InterpolatedUnivariateSpline(s_ray, R_BPD_ray)
        BPD_spl = InterpolatedUnivariateSpline(s_ray, BPD_ray)
        rhop_spl = InterpolatedUnivariateSpline(s_ray, rhop_BPD_ray)
        n_rhop = 120
        rhop_binned = np.linspace(0.0, 1.0, n_rhop)
        BPD_binned = np.zeros(n_rhop)
        BPD_ch_binned = np.zeros(n_rhop)
        for i in range(len(rhop_binned)):
            if(recalc_BPD):
                for ray, ray_BPD_spl, ray_weight in zip(ray_list, ray_BPD_spl_list, Results.weights["ray"][itime][ich]):
                    root_spl_ray = InterpolatedUnivariateSpline(ray["s"], ray["rhop"] - rhop_binned[i])
                    for root in root_spl_ray.roots():
                        BPD_ch_binned[i] += ray_BPD_spl(root) * ray_weight
            else:
                root_spl = InterpolatedUnivariateSpline(s_ray, rhop_BPD_ray - rhop_binned[i])
                for root in root_spl.roots():
                    BPD_binned[i] += BPD_spl(root)
                root_spl_ch = InterpolatedUnivariateSpline(s_helper, BPD_ch_rhop - rhop_binned[i])
                for root in root_spl_ch.roots():
                    BPD_ch_binned[i] += BPD_ch_spl(root)
        s_BPD_max = s_ray[np.argmax(BPD_ray)]
        BPD_ray_spl = InterpolatedUnivariateSpline(s_ray, BPD_ray)
        sigma_BPD_ray_spl = InterpolatedUnivariateSpline(s_ray, BPD_ray * (s_ray - s_BPD_max) ** 2)
        sigma_guess = np.sqrt(sigma_BPD_ray_spl.integral(s_ray[0], s_ray[-1]) / BPD_ray_spl.integral(s_ray[0], s_ray[-1]))
        beta0 = np.array([np.max(BPD_ray), s_BPD_max, sigma_guess])
        data = odr.Data(s_ray, BPD_ray)
        mdl = odr.Model(func)
        ODR = odr.ODR(data, mdl, beta0)
        output = ODR.run()
        beta = output.beta
#        ax_depo.plot(s_ray, func(beta, s_ray), label="Gaussian BPD for channel {0:d}".format(ich))
#        ax_depo.plot(s_ray, func(beta0, s_ray), "+", label="Gaussian BPD for channel {0:d}".format(ich))
#        ax_depo.plot(s_ray, BPD_ray, label="BPD for channel {0:d}".format(ich), linestyle="--")
#        plt.show()
        max_shift = np.abs(s_BPD_max - beta[1]) / sigma_guess
        print(max_shift)
        if(max_shift > 0.2):
            print("Discarding fit results cause BPD seems very skewed -> initial guess most likely more accurate")
            beta = beta0
        BPD_POI = []
        BPD_POI_rhop = []
        s_important = [s_BPD_max, s_BPD_max + beta[2], s_BPD_max - beta[2]]
        for s in s_important:
            BPD_POI.append(R_spl(s))
            BPD_POI_rhop.append(rhop_spl(s))
            if(distribution_rhop is None):
                distribution_rhop = np.abs(rhop_spl(s_BPD_max))
        R_BPD_list.append(BPD_POI)
        R_BPD_dict[str(ich)] = BPD_POI
        rhop_BPD_dict[str(ich)] = BPD_POI_rhop
        s_BPD_list.append(s_important)
        s_BPD_dict[str(ich)] = s_important
        label = "BPD"
        if(use_rhop):
            if(preserve_original_BPD):
                ax_depo.plot(rhop_BPD_ray, BPD_ray / np.max(BPD_ray), label="BPD", linestyle="-")  # for channel {0:d}".format(ich)
            elif(single_ray_BPD):
                ax_depo.plot(rhop_binned, BPD_binned / np.max(BPD_binned), label="BPD", linestyle="-", color="blue")  # for channel {0:d}".format(ich)
                ax_depo.plot(rhop_binned, BPD_ch_binned / np.max(BPD_ch_binned), label=r"BPD $5 \times 5$ rays", marker="+", linestyle="None", color="blue")  # for channel {0:d}".format(ich)
            else:
                ax_depo.plot(rhop_binned, BPD_ch_binned / np.max(BPD_ch_binned), label="BPD", linestyle="-", color="blue")  # for channel {0:d}".format(ich)
        else:
            ax_depo.plot(R_BPD_ray, BPD_ray, label="BPD", linestyle="-")  # for channel {0:d}".format(ich)
#        if(len(rhop_BPD_ray[rhop_BPD_ray > 0]) > 0):
#            p = ax_depo.plot(rhop_BPD_ray[rhop_BPD_ray > 0], BPD_ray[rhop_BPD_ray > 0], label="BPD for channel {0:d}".format(ich), linestyle="-")
#        if(len(rhop_BPD_ray[rhop_BPD_ray < 0]) > 0 and p is None):
#            ax_depo.plot(rhop_BPD_ray[rhop_BPD_ray < 0], BPD_ray[rhop_BPD_ray < 0], label="BPD for channel {0:d}".format(ich), linestyle="-")
#        elif(len(rhop_BPD_ray[rhop_BPD_ray < 0]) > 0):
#            ax_depo.plot(rhop_BPD_ray[rhop_BPD_ray < 0], BPD_ray[rhop_BPD_ray < 0], linestyle="-", color=p[-1].get_color()
    if(usemat):
        f_inter = make_f_inter(dist, dist_obj=dist_obj, EQObj=EQObj, time=time)
    else:
        f_inter = make_f_inter(dist, working_dir=working_dir)
    for ich, m_ch, s_important in zip(ch_list, m_list, s_BPD_list):
        if(usemat):
            svec, freq, Trad, T, = load_data_for_3DBDOP(time, dist, ich, Results=Results)
        else:
            svec, freq, Trad, T, = load_data_for_3DBDOP(time, dist, ich, working_dir=working_dir)
        BDOP_list.append(BDOP_3D(svec, freq, Trad, T, f_inter, dist, B_ax, m=m_ch, only_contribution=only_contribution, steps=500, s_important=s_important))
        m = cm.ScalarMappable(cmap=plt.cm.get_cmap("winter"))
        m.set_array(np.linspace(0.0, 1.0, 20))
        cmaps.append(m)
        alphas.append(ece_alpha)
        is_ecrh_list.append(0)
    if(include_ECRH):
        ibeam = 1
        Beam_max = np.max(linear_beam.PW_beam.flatten())
        for beam, PW_beam, color in zip(linear_beam.rays, linear_beam.PW_beam, ECRH_colors):
            beam_rhop = beam[0]["rhop"]
            R_spl = InterpolatedUnivariateSpline(beam[0]["s"], beam[0]["R"])
            P_spl = InterpolatedUnivariateSpline(beam[0]["s"], beam[0]["PW"])
            rhop_spl = InterpolatedUnivariateSpline(beam[0]["s"], beam_rhop)
            dP = P_spl(beam[0]["s"], nu=1)
            beta0 = np.array([np.max(dP), beam[0]["s"][np.argmax(dP)], 0.05])
            data = odr.Data(beam[0]["s"], dP)
            mdl = odr.Model(func)
            ODR = odr.ODR(data, mdl, beta0)
            output = ODR.run()
            beta = output.beta
    #        beta[2] *= 10.0
            label = "ECRH (Gray)"
            if(not single_Beam):
                label += " beam no. " + str(ibeam)
            else:
                label += "linear damping"
            if(use_rhop):
                ax_depo.plot(linear_beam.rhop, PW_beam / Beam_max, label=label, linestyle="--", color=color)
#                ax_depo.plot(beam_rhop[beam_rhop < 0], dP[beam_rhop < 0] / np.max(beam[0]["PW"]), label=label, linestyle="--", color=color)
#                ax_depo.plot(beam_rhop[beam_rhop > 0], dP[beam_rhop > 0] / np.max(beam[0]["PW"]), linestyle="--", color=color)
            else:
                ax_depo.plot(beam[0]["R"], dP / Beam_max, label=label, linestyle="--", color=color)
#            ax_depo.plot(rhop_ray_signed, dP / np.max(beam[0]["PW"]), label="Central ray Gray beam no. " + str(ibeam), linestyle="--", color="black")
#            ax_depo.plot(quasi_linear_beam.rhop, quasi_linear_beam.PW, "+", label="RELAX", color=(0.4, 0.4, 0.0))
#            ax_depo.plot(rhop_ray_signed, func(beta, beam[0]["s"]), label="Gaussian BPD for channel {0:d}".format(ich))
#            ax_depo.plot(rhop_ray_signed, dP, "--", label="Gaussian BPD for channel {0:d}".format(ich))
            s_max = beam[0]["s"][np.argmax(dP)]
            s_important = [s_max, s_max + beta[2], s_max - beta[2]]
            PDP_POI = []
            PDP_POI_rhop = []
            for s in s_important:
                PDP_POI.append(R_spl(s))
                PDP_POI_rhop.append(rhop_spl(s))
            R_BPD_dict["ECRH" + "_" + str(ibeam)] = PDP_POI
            rhop_BPD_dict["ECRH" + "_" + str(ibeam)] = PDP_POI_rhop
            s_BPD_dict["ECRH" + "_" + str(ibeam)] = s_important
            ibeam += 1
            for m_ECRH in m_ECRH_list:
                BDOP_list.append(PowerDepo_3D(freq, beam[0], f_inter, shot, time, dist, B_ax, EqSlice, Te_spline, ne_spline, m=m_ECRH, only_contribution=only_contribution, steps=500, s_important=s_important))
                m = cm.ScalarMappable(cmap=plt.cm.get_cmap("spring"))
                m.set_array(np.linspace(0.0, 1.0, 20))
                cmaps.append(m)
                alphas.append(0.3)
                is_ecrh_list.append(1)
                R_BPD_list.append(PDP_POI)
                s_BPD_list.append(s_important)
        if(use_rhop):
            ax_depo.plot(quasi_linear_beam.rhop, quasi_linear_beam.PW / np.max(quasi_linear_beam.PW), label="ECRH (RELAX)", linestyle="None", marker="^", color="black")
#        if(BPD_max < np.max(BDOP_list[-1].val)):
#            i_max = np.argmax(BDOP_list[-1].val)
#            i_x_max, i_u_par_max = np.unravel_index(i_max, BDOP_list[-1].val.shape)
#            x_max = BDOP_list[-1].rho[i_x_max, i_u_par_max]
#            u_perp_max = BDOP_list[-1].u_perp[i_x_max, i_u_par_max]
#            u_par_max = BDOP_list[-1].u_par[i_x_max, i_u_par_max]
#            BPD_max = np.max(BDOP_list[-1].val)
#    if(dist == "Ge" or dist == "ReComp"):
#        BPD_max = np.max(BDOP_val_list - np.min(BDOP_val_list))
#        BDOP_val_list = (BDOP_val_list - np.min(BDOP_val_list)) / BPD_max
#    else:
#        BPD_max = np.max(BDOP_val_list)
#        BDOP_val_list /= BPD_max
    if(len(BDOP_list) > 1):
        leg = ax_depo.legend()
        leg.draggable()
    if(use_rhop):
        te_ax = ax_depo.twinx()
        te_ax.plot(rhop_Te, np.exp(Te) * 1.e-3, "--", label="$T_\mathrm{e}$")
        te_ax.set_ylabel(r"$T_\mathrm{e}$ [\si{\kilo\electronvolt}]")
        ax_depo.set_xlabel(r"$\rho_\mathrm{pol}$")

    else:
        ax_depo.set_xlabel(r"$R\,[\si{\metre}]$")
    if(include_ECRH):
        ax_depo.set_ylabel(r"$D_\omega\, \mathrm{and} \, \mathrm{d}P/\mathrm{d}R\,[\si{{a.u.}}]$")
    else:
        ax_depo.set_ylabel(r"$D_\omega\,[\si{{a.u.}}]$")
    steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    ax_depo.get_xaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps, prune="lower"))
    ax_depo.get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 2.0))
    plt.autoscale(False)
    if(use_rhop):
        for key in R_BPD_dict.keys():
            if("2" in  key):
                color = ECRH_colors[1]
            else:
                color = ECRH_colors[0]
            if("ECRH" not in key):
                ax_depo.vlines(rhop_BPD_dict[key], -300, 1000, linestyle="-.", color="blue")
            else:
                ax_depo.vlines(rhop_BPD_dict[key], -300, 1000, linestyle=":", color=color)
    else:
        for key in R_BPD_dict.keys():
            if("2" in  key):
                color = ECRH_colors[1]
            else:
                color = ECRH_colors[0]
            if("ECRH" not in key):
                ax_depo.vlines(R_BPD_dict[key], -300, 1000, linestyle="-.", color="blue")
            else:
                ax_depo.vlines(R_BPD_dict[key], -300, 1000, linestyle=":", color=color)
    ax_depo.set_ylim(-0.01, 1.2)
    ax_reso = fig.add_subplot(122)
    mdict["BPD_vals"] = []
    mdict["BPD_rho"] = []
    mdict["BPD_u_par"] = []
    mdict["BPD_u_perp"] = []
    mdict["BPD_facecolors"] = []
    mdict["is_ecrh_list"] = []
    got_f = False
    for BDOP, cmap, alpha, s_list in zip(BDOP_list, cmaps, alphas, s_BPD_list):
        BDOP_s = np.mean(BDOP.s, axis=1)
#        if(np.all((np.abs(BDOP_rhop) - np.abs(rhop_max)) < 0.0) or \
#                  np.all((np.abs(BDOP_rhop) - np.abs(rhop_max)) > 0.0)):
#            print("Skipping resonance line - no points available for currently selected rhop")
#            print("Rhop range of current BPD", np.min(np.abs(BDOP_rhop)), np.max(np.abs(BDOP_rhop)))
#            continue
        for s_max in s_list:
            i_s = np.argmin(np.abs(BDOP_s - s_max))
            print("s found vs. inquired s", BDOP_s[i_s], s_max)
            BDOP_val_norm = BDOP.val[i_s] / np.max(BDOP.val[i_s])
            linecolor = cmap.to_rgba(BDOP_val_norm, alpha)
            u_par = BDOP.u_par[i_s]
            u_perp = BDOP.u_perp[i_s]
            if(np.min(u_par) < u_par_range[0]):
                u_par_range[0] = np.min(u_par)
            if(np.max(u_par) > u_par_range[1]):
                u_par_range[1] = np.max(u_par)
            if(np.max(u_perp) > u_perp_range[1]):
                u_perp_range[1] = np.max(u_perp)
            i_start = 0
            i_end = 1
            while(i_end < len(u_par)):
                while(np.abs(BDOP_val_norm[i_start] - BDOP_val_norm[i_end]) < 0.01 and i_end < len(u_par) - 1):
                    i_end += 1
                i_end += 1
                ax_reso.plot(u_perp[i_start:i_end], u_par[i_start:i_end], linestyle="solid", color=linecolor[i_start])
                i_start = i_end - 2
            if(not got_f):
                x, y, spline = BDOP.f_inter.get_spline(distribution_rhop, np.exp(Te_spline(distribution_rhop)))
                dist_inter = distribution_interpolator(x, y, spline)
                zeta = BDOP.zeta[i_s]
                got_f = True
    N_f = 300
    f = np.zeros((N_f, 2 * N_f))
    print("U_perp /U_par range", u_perp_range, u_par_range)
    u_par_extend = u_par_range[1] - u_par_range[0]
    u_par_dist = np.linspace(u_par_range[0] - 0.5 * u_par_extend, u_par_range[1] + u_par_extend * 0.5, 2 * N_f)
    u_perp_dist = np.linspace(u_perp_range[0], u_perp_range[1] * 2.0, N_f)
    for i in range(len(u_perp_dist)):
        if(dist not in ["Ge", "GeComp"]):
            f[i] = dist_inter.eval_dist(u_perp_dist[i], u_par_dist, Te, "spline", zeta)
        else:
            f[i] = Juettner2D(u_perp_dist[i], u_par_dist, np.exp(Te_spline(distribution_rhop)))
    if(Teweight):
        for i in range(len(u_perp_dist)):
            f[i] *= u_perp_dist[i] ** 2 / np.sqrt(1.0 + u_perp_dist[i] ** 2 * u_par_dist ** 2)
        f /= np.max(f.flatten())
        levels = np.linspace(0, 1.0, 20)
        print(np.max(f))
        cont2 = ax_reso.contourf(u_perp_dist, u_par_dist, f.T, levels=levels,
                                 hold='on', cmap=cm.get_cmap("plasma"))
    else:
        f = np.log10(f)
        levels = np.linspace(-13.0, 5.0, 20)
        cont2 = ax_reso.contour(u_perp_dist, u_par_dist, f.T, levels=levels, colors='k',
                                hold='on', alpha=1.0, linewidths=1)
#        cont2 = ax_reso.contourf(u_perp_dist, u_par_dist, f.T, levels=levels, \
#                                hold='on', cmap=cm.get_cmap("plasma"))
    if(include_ECRH):
        cb_ECRH = fig.colorbar(cmaps[-1], pad=0.15, ticks=[0.0, 0.5, 1.0])
        cb_ECRH.set_label(r"$\mathrm{d}P_\mathrm{ECRH}/d\mathrm{s} [\si{{a.u.}}]$")
    if(Teweight):
        cb_dist = fig.colorbar(cont2, pad=0.15, ticks=[0.0, 0.5, 1.0])
        cb_dist.set_label(r"$f_\mathrm{MJ} u_\perp^2 / \gamma f_0$")
#    else:
#        cb_dist = fig.colorbar(cont2, pad=0.15, ticks=[0.0, 0.5, 1.0])
#        cb_dist.set_label(r"$\mathrm{log}_10\left(f\right)$")
    cb = fig.colorbar(cmaps[0], ticks=[0.0, 0.5, 1.0])
    cb.set_label(r"$D_\omega [\si{{a.u.}}]$")
    ax_reso.set_xlim(u_perp_range[0], u_perp_range[1])
    ax_reso.set_ylim(u_par_range[0], u_par_range[1])
    ax_reso.set_ylabel(r"$u_\parallel$")
    ax_reso.set_xlabel(r"$u_\perp$")
    ax_reso.get_xaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps))
    ax_reso.get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 2.0))
    ax_reso.get_yaxis().set_major_locator(MaxNLocator(nbins=4, steps=steps))
    ax_reso.get_yaxis().set_minor_locator(MaxNLocator(nbins=8, steps=steps / 2.0))
    ax_reso.set_aspect("equal")
    plt.tight_layout()
    plt.show()
# make_3DBDOP("/ptmp1/work/sdenk/nssf/33697/4.80/OERT/ed_39/", 33697, 4.80, 43, "Re", diag="IDE")  # RELAX
if(__name__ == "__main__"):
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_156/", 33697, 4.80, [38], [2], "Re", diag="IDE", ece_alpha=0.7, include_ECRH=True, single_Beam=True, only_contribution=True, save_only=True)  # RELAX
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_3/", 33697, 4.80, [38], [2], "Re", diag="IDE", ece_alpha=0.7, include_ECRH=False, single_Beam=True, only_contribution=True, save_only=False)  # RELAX
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_147/", 33697, 4.80, [2], [2], "Re", diag="IDE", ece_alpha=0.7, include_ECRH=True, single_Beam=True, only_contribution=True, save_only=False)  # RELAX
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/34663/3.60/OERT/ed_8/", 34663, 34663, [93], [2], "Re", diag="IDE", ece_alpha=0.7, include_ECRH=True, single_Beam=False, only_contribution=True, save_only=False, single_ray_BPD=True)  # RELAX
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_8/", 33697, 4.80, [2], [2], "Re", diag="IDE", m_ECRH_list=[2], include_ECRH=True, single_Beam=False, only_contribution=True, save_only=False, single_ray_BPD=False)  # RELAX
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/33705/4.90/OERT/ed_8/", 33705, 4.90, [95], [2], "Re", diag="IDE", ece_alpha=0.7, include_ECRH=True, single_Beam=False, only_contribution=True, save_only=False, single_ray_BPD=False)  # RELAX
#    make_3DBDOP_cut("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_11/", 34663, 3.60, [15, 95], [2, 2], "Re", diag="IDE", m_ECRH_list=[2], ece_alpha=0.7, include_ECRH=True, single_Beam=False, only_contribution=True, save_only=False, single_ray_BPD=False)  # RELAX
    make_3DBDOP_cut("/tokp/work/sdenk/ECRad/ECRad_35662_ECECTACTC_ed7.mat", 35662, 4.4, [81], [2], "Re", diag="IDE", \
                    m_ECRH_list=[2], ece_alpha=0.7, include_ECRH=True, single_Beam=False, only_contribution=True, save_only=False, \
                    single_ray_BPD=False, preserve_original_BPD=False, ECRH_freq=105.e9, Teweight=False, alternative_mat_for_waves = \
                    "/tokp/work/sdenk/ECRad/ECRad_35662_ECECTACTC_ed9.mat", recalc_BPD=True)  # 
#    make_3DBDOP_cut("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/", 33585, 3.0, [5], [2], "Ge", diag="EQH", m_ECRH_list=[], ece_alpha=0.7, include_ECRH=False, single_Beam=False, only_contribution=True, save_only=False, single_ray_BPD=True, preserve_original_BPD=True, Teweight=True)  # RELAX
#    make_3DBDOP("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/", 33585, 3.00, [5], [2], "Ge", diag="EQH", title=None, ece_alpha=0.7, include_ECRH=False, only_ECRH=False, only_contribution=True, flat=True, single_Beam=True)  # RELAX
#    make_3DBDOP("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_3/", 33697, 4.80, [38], [2], "Re", diag="IDE", title=None, ece_alpha=0.7, include_ECRH=False, only_ECRH=False, only_contribution=True, flat=True, single_Beam=True)  # RELAX
#    make_3DBDOP("/tokp/work/sdenk/nssf/34663/3.60/OERT/ed_11/", 34663, 3.60, [15, 66], [2, 2], "Re", diag="IDE", title=None, ece_alpha=0.7, include_ECRH=True, only_ECRH=False, only_contribution=True, flat=True, single_Beam=True, save_only=False)  # RELAX
#    make_3DBDOP("/ptmp1/work/sdenk/nssf/33134/3.16/OERT/", 33134, 3.16, [17, 27], [2, 2], "TB", diag="EQH", title=None, nocolor=True, ece_alpha=0.7, save_only=True)
#    make_3DBDOP("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_147/", 33697, 4.80, [38], [2], "Re", diag="IDE", title="RELAX", include_ECRH=True, only_contribution=True, ece_alpha=0.7, save_only=False, only_ECRH=True)  # RELAX.0,
#    make_3DBDOP("/ptmp1/work/sdenk/nssf/33697/4.80/OERT/ed_73", 33697, 4.80, [38], [2], "Re", diag="IDE", title="RELAX", include_ECRH=True, only_contribution=True, ece_alpha=0.7, save_only=True, only_ECRH=False)  # RELAX.0,

#    make_3DBDOP("/tokp/work/sdenk/nssf/33705/4.90/OERT/ed_95/", 33705, 4.90, [60], [2], "Re", diag="IDE", title="RELAX", include_ECRH=True, only_contribution=True)  # RELAX
#    make_3DBDOP("/tokp/work/sdenk/nssf/34663/3.60/OERT/", 34663, 3.60, [65], [2], "Re", diag="IDE", title="RELAX", only_contribution=True, include_ECRH=True)  # RELAX
#    make_3DBDOP("/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_97/", 34663, 3.60, [86], [2], "Re", diag="IDE", title="RELAX")
#    make_3DBDOP("/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_97/", 34663, 3.60, [86], [2], "ReTh", diag="IDE", title="RELAX")
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33697/4.80/OERT/ed_125/", 33697, 4.80, 38, "Re", diag="IDE", title="thermal", m=3, BPD_max_ext=BPD_max)  # RELAX
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/", 33585, 3.00, 4, "Ge", diag="EQH", title="GENE ch. 4")
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_7/", 33585, 3.00, 4, "Ge", diag="EQH", title="rel. BiMaxwellian ch. 4")  # thermal
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/", 33585, 3.00, 6, "Ge", diag="EQH", title="GENE ch. 6")
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_7/", 33585, 3.00, 6, "Ge", diag="EQH", title="rel. BiMaxwellian ch. 6")  # thermal
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/", 33585, 3.00, 7, "Ge", diag="EQH", title="GENE ch. 7")
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_24/", 33585, 3.00, 12, "Ge", diag="EQH", title="rel. BiMaxwellian ch. 7")  # thermal
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_1/", 34663, 3.60, 70, "Re", diag="EQH", title="RELAX", m=2)
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_1/", 33697, 3.60, 3, "Re", diag="EQH", title="RELAX", m=3)  # RELAX
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_3/", 31539, 2.81, 11, "TB", diag="EQH", title="Thermal", m=2)  # RELAX
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_3/", 31539, 2.81, 17, "TB", diag="EQH", title="Thermal", m=2)  # RELAX
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_3/", 31539, 2.81, 26, "TB", diag="EQH", title="Thermal", m=2)  # RELAX
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/31594/1.30/OERT/ed_13/", 31594, 1.30, 4, "Re", diag="EQH", title="RELAX", m=2)
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_5/", 34663, 3.60, 49, "Re", diag="EQH", title="RELAX", m=2)
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/34663/3.60/OERT/ed_2/", 34663, 3.60, 38, "Re", diag="EQH", title="RELAX", m=2)
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_11/", 33705, 4.90, 50, "Re", diag="IDE", title="RELAX", m=2)  # High frequency
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_12/", 34663, 4.90, 5, "Re", diag="IDE", title="RELAX", m=2)  # Low frequency
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_8/", 31539, 2.81, 15, "TB", diag="EQH", title="Thermal", m=2)
#    make_3DBDOP("/ptmp1/work/sdenk/nssf/33134/3.16/OERT/", 33134, 3.16, [17, 27], [2, 2], "TB", diag="EQH", title="thermal")
#    make_3DBDOP(fig, "/ptmp1/work/sdenk/nssf/33134/3.16/OERT/", 33134, 3.16, 27, "TB", diag="EQH", title="Thermal", m=2)

