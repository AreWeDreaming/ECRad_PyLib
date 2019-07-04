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
from Distribution import f_interpolator
from distribution_io import read_waves_mat_to_beam, read_dist_mat_to_beam, \
                            load_f_from_mat
from distribution_functions import Juettner2D                           
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

def make_f_inter(dist, EQObj, working_dir=None, dist_obj=None, time=None):
    if(dist in ["Re", "ReComp", "Ge", "GeComp"]):
        res_dist = dist.replace("Comp", "")
    else:
        res_dist = "thermal"
    EqSlice = EQObj.GetSlice(time)
    if(res_dist == "thermal"):
        f_inter = f_interpolator(dist=res_dist)
    elif(working_dir is not None):
        f_inter = f_interpolator(working_dir=working_dir, dist=res_dist, EqSlice=EqSlice)
    else:
        rhop_Bmin, Bmin = EQObj.get_B_min(time, dist_obj.rhop, append_B_ax=True)
        f_inter = f_interpolator(dist_obj=dist_obj, dist=res_dist, rhop_Bmin=rhop_Bmin, Bmin=Bmin)
    if(dist == "Ge"):
        f_inter_scnd = f_interpolator(working_dir, dist="Ge0", EqSlice=EqSlice)
        return f_inter, f_inter_scnd # Gene f0 distribution
    else:
        return f_inter, None # None will be replaced with f_inter based on thermal distribution


def make_3DBDOP_for_ray(result, time, ch, ir, m, B_ax, f_inter=None, N_pnts=100):
    # Currently only supported for non-Gene distributions
    dist = result.Config.dstf
    svec, freq, Trad, T, BPD = load_data_for_3DBDOP(result, time, dist, ch, ir=ir, get_BPD=True)
    s = distribute_points(svec["s"], BPD, N_pnts)
    return BDOP_3D(s, svec, freq, Trad, T, f_inter, dist, B_ax, m=m)
    

def load_data_for_3DBDOP(Results, time, dist, ch, ir=1, get_BPD=False):
    ich = ch - 1
    itime_Scenario = np.argmin(np.abs(Results.Scenario.plasma_dict["time"] - time))
    freq = Results.Scenario.ray_launch[itime_Scenario]["f"][ich]
    itime = np.argmin(np.abs(Results.time - time))
    svec = {}
    if(Results.Config.N_ray == 1):
        svec["rhop"] = Results.ray["rhopX"][itime][ich]
        svec["s"] = Results.ray["sX"][itime][ich][svec["rhop"] != -1.0]
        svec["R"] = np.sqrt(Results.ray["xX"][itime][ich][svec["rhop"] != -1.0] ** 2 + \
                            Results.ray["yX"][itime][ich][svec["rhop"] != -1.0] ** 2)
        svec["z"] = Results.ray["zX"][itime][ich][svec["rhop"] != -1.0]
        svec["Te"] = Results.ray["TeX"][itime][ich][svec["rhop"] != -1.0]
        svec["theta"] = Results.ray["thetaX"][itime][ich][svec["rhop"] != -1.0]
        svec["freq_2X"] = Results.ray["YX"][itime][ich][svec["rhop"] != -1.0] * 2.0 * freq
        svec["N_abs"] = Results.ray["NcX"][itime][ich][svec["rhop"] != -1.0]
    else:
        ray_index = ir - 1
        svec["rhop"] = Results.ray["rhopX"][itime][ich][ray_index]
        svec["s"] = Results.ray["sX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["R"] = np.sqrt(Results.ray["xX"][itime][ich][ray_index][svec["rhop"] != -1.0] ** 2 + \
                            Results.ray["yX"][itime][ich][ray_index][svec["rhop"] != -1.0] ** 2)
        svec["z"] = Results.ray["zX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["Te"] = Results.ray["TeX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["theta"] = Results.ray["thetaX"][itime][ich][ray_index][svec["rhop"] != -1.0]
        svec["freq_2X"] = Results.ray["YX"][itime][ich][ray_index][svec["rhop"] != -1.0] * 2.0 * freq
        svec["N_abs"] = Results.ray["NcX"][itime][ich][ray_index][svec["rhop"] != -1.0]
    ne_spl = InterpolatedUnivariateSpline(Results.Scenario.plasma_dict["rhop_prof"][itime_Scenario], \
                                          np.log(Results.Scenario.plasma_dict["ne"][itime_Scenario]), ext=3)
    svec["ne"] = np.exp(ne_spl(svec["rhop"][svec["rhop"] != -1.0]))
    if(dist != "ReTh"):
        Trad = Results.Trad[itime][ich]
        if(len(Results.ray["TX"]) == 0):
            raise ValueError("WARNING THERE IS NO TRANSMIVITY DATA AVAILABLE")
        elif(Results.Config.N_ray == 1):
            T = Results.ray["TX"][itime][ich]
            if(get_BPD):
                BPD = Results.ray["BPDX"][itime][ich]
        else:
            T = Results.ray["TX"][itime][ich][ray_index]
            if(get_BPD):
                BPD = Results.ray["BPDX"][itime][ich][ray_index]
    else:
        T = Results.ray["T_secondX"][itime][ich][ray_index]
        if(get_BPD):
                BPD = Results.ray["BPD_secondX"][itime][ich][ray_index]
    T = T[svec["rhop"] != -1.0]
    if(get_BPD):
        BPD = BPD[svec["rhop"] != -1.0]
    svec["rhop"] = svec["rhop"][svec["rhop"] != -1.0]
    svec["ne"][svec["ne"] < 1.e15] = 1.e15
    svec["Te"][svec["Te"] < 2.e-2] = 2.e-2
    if(get_BPD):
        return svec, freq, Trad, T, BPD
    else:
        return svec, freq, Trad, T

def distribute_points(x, weight, N_pnts):
    # Distributes N_pnts amounts of points such that the point density is 
    # weighted according to weight
    weight_spl = InterpolatedUnivariateSpline(x, weight)
    weight_internal = np.copy(weight)
    weight_internal /= weight_spl.integral(x[0], x[-1]) # Normalize
    cum_weight_spl = InterpolatedUnivariateSpline(x, weight_internal).antiderivative(1)
    cum_weight = cum_weight_spl(x)
    x_weighted = []
    func_points = np.linspace(cum_weight_spl(x[0]), cum_weight_spl(x[-1]), N_pnts)
    for val in func_points:
        if(val not in cum_weight):
            spl = InterpolatedUnivariateSpline(x, cum_weight - val)
            x_weighted.append(spl.roots()[0])
        else:
            x_weighted.append(x[val == cum_weight][0])
    return np.array(x_weighted)
#     plt.plot(x, weight, "^")
#     plt.plot(x_weighted, weight_spl(x_weighted), "+")
#     plt.show()

class BDOP_3D:
    def __init__(self, s, svec, freq, Trad, T, f_inter, dist, B_ax, m=2, f_inter_scnd=None):
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
        # Already did this msot likely, but doesnt hurt tp do it again
        for key in svec.keys():
            if(key != "rhop"):
                svec[key] = svec[key][svec["rhop"] != -1.0]
        T = T[svec["rhop"] != -1.0]
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
        if(f_inter_scnd is None):
            self.f_inter_scnd = f_interpolator(None, dist="thermal")
        else:
            self.f_inter_scnd = f_inter_scnd
        self.u_par_max = -np.inf
        self.u_perp_max = -np.inf
        R_spl = InterpolatedUnivariateSpline(svec["s"], svec["R"])
        rhop_spl = InterpolatedUnivariateSpline(svec["s"], svec["rhop"])
        Te_spl = InterpolatedUnivariateSpline(svec["s"], np.log(svec["Te"]))
        ne_spl = InterpolatedUnivariateSpline(svec["s"], np.log(svec["ne"]))
        freq_2X_spl = InterpolatedUnivariateSpline(svec["s"], svec["freq_2X"])
        theta_spl = InterpolatedUnivariateSpline(svec["s"], svec["theta"])
        T_spl = InterpolatedUnivariateSpline(svec["s"], T)
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
            if(em_abs_Alb_obj.is_resonant(rhop, Te, ne, freq_2X, theta, freq, m)):
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
                    u, pitch, spline = self.f_inter_scnd.get_spline(rhop, Te)
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
    def __init__(self, freq, ray, f_inter, dist, B_ax, EqSlice, Te_spl, ne_spl, m=2, only_s_important=False, only_contribution=False, steps=2000, s_important=[], f_inter_scnd=None):
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
        self.only_s_important = only_s_important
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
        if(f_inter_scnd is None):
            self.f_inter_scnd = f_interpolator(None, dist="thermal")
        else:
            self.f_inter_scnd = f_inter_scnd
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
        if(self.only_s_important):
            s = np.copy(s_important)
        else:
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


def make_3DBDOP_cut_GUI(Results, fig,  time, ch):
    return make_3DBDOP_cut(fig, Results, time, [ch], [2], "Th", only_contribution=True)

def make_3DBDOP_cut_standalone(matfilename, time, ch_list, m_list, dist, include_ECRH=False, \
                               single_Beam=False, m_ECRH_list=[2], only_contribution=False, \
                               single_ray_BPD=False, Teweight=False, ECRH_freq=140.e9, \
                               mat_for_waves_and_distribution=None):
    Results = ECRadResults()
    fig = plt.figure(figsize=(16.5, 8.5))
    Results.from_mat_file(matfilename)
    fig = make_3DBDOP_cut(fig, Results, time, ch_list, m_list, dist, include_ECRH=include_ECRH, \
                    single_Beam=single_Beam, m_ECRH_list=m_ECRH_list, only_contribution=only_contribution, \
                    single_ray_BPD=single_ray_BPD, Teweight=Teweight, ECRH_freq=ECRH_freq, \
                    mat_for_waves_and_distribution=mat_for_waves_and_distribution)
    plt.show()

def make_3DBDOP_cut(fig, Results, time, ch_list, m_list, dist, include_ECRH=False, \
                    single_Beam=False, m_ECRH_list=[2], only_contribution=False, \
                    single_ray_BPD=False, Teweight=False, ECRH_freq=140.e9, \
                    mat_for_waves_and_distribution=None):
    fig.text(0.025, 0.95, "a)")
    fig.text(0.55, 0.95, "b)")
    BDOP_list = []
    use_fit_for_s_important = False
    itime = np.argmin(np.abs(Results.Scenario.plasma_dict["time"] - time))
    rhop_Te = Results.Scenario.plasma_dict["rhop_prof"][itime] * Results.Scenario.Te_rhop_scale
    Te = np.log(Results.Scenario.plasma_dict["Te"][itime] * Results.Scenario.Te_scale)  # from IDA always positive definite
    rhop_ne = Results.Scenario.plasma_dict["rhop_prof"][itime] * Results.Scenario.ne_rhop_scale
    ne = np.log(Results.Scenario.plasma_dict["Te"][itime] * Results.Scenario.ne_scale)  # from IDA always positive definite
    EqSlice = Results.Scenario.plasma_dict["eq_data"][itime]
    EQObj = EQDataExt(Results.Scenario.shot, bt_vac_correction=1.0, Ext_data=True)
    EQObj.insert_slices_from_ext(Results.Scenario.plasma_dict["time"], Results.Scenario.plasma_dict["eq_data"])
    B_ax = EQObj.get_B_on_axis(time)
    R_ax, z_ax = EQObj.get_axis(time)
    Te_spline = InterpolatedUnivariateSpline(rhop_Te, Te)
    ne_spline = InterpolatedUnivariateSpline(rhop_ne, ne)
    print("Position of magn. axus", R_ax, z_ax)
    if(mat_for_waves_and_distribution is not None):
        mat = loadmat(mat_for_waves_and_distribution, squeeze_me=True)
        dist_obj = load_f_from_mat(mat_for_waves_and_distribution, use_dist_prefix=True)
        if(include_ECRH):
            linear_beam = read_waves_mat_to_beam(mat, EqSlice, use_wave_prefix=True)
            quasi_linear_beam = read_dist_mat_to_beam(mat, use_dist_prefix=True)
            if(single_Beam):
                linear_beam.rays = linear_beam.rays[1:2]
    else:
        dist_obj = None
    freq = ECRH_freq
    cmaps = []
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
    ECRH_colors = ["magenta", "red"]
    for ch, m_ch in zip(ch_list, m_list):
        ich = ch - 1
        if(ich in ch_done_list):
            R_BPD_list.append(R_BPD_dict[str(ich)])
            s_BPD_list.append(s_BPD_dict[str(ich)])
            continue
        else:
            ch_done_list.append(ich)
        ray_list = []
        ray_BPD_spl_list = []
        iray = 2
        steps_in_plasma = 0
        if(Results.Config.N_ray == 1):
            ray_dict = {}
            ray_dict["s"] = Results.ray["sX"][itime][ich]
            ray_dict["x"] = Results.ray["xX"][itime][ich]
            ray_dict["y"] = Results.ray["yX"][itime][ich]
            ray_dict["z"] = Results.ray["zX"][itime][ich]
            ray_dict["rhop"] = Results.ray["rhopX"][itime][ich]
            ray_dict["BPD"] = Results.ray["BPDX"][itime][ich]
            ray_dict["BPD_second"] = Results.ray["BPD_secondX"][itime][ich]
            ray_dict["N_ray"] = Results.ray["NX"][itime][ich]
            ray_dict["N_cold"] = Results.ray["NcX"][itime][ich]
            ray_dict["theta"] = Results.ray["thetaX"][itime][ich]
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
            steps_in_plasma = len(ray_dict["rhop"][ray_dict["rhop"] >= 0.0])
            ray_list.append(dict(ray_dict))
            ray_BPD_spl_list.append(InterpolatedUnivariateSpline(ray_list[-1]["s"], ray_list[-1]["BPD"]))
        else:
            for iray in range(Results.Config.N_ray):
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
                if(steps_in_plasma < len(ray_dict["rhop"][ray_dict["rhop"] >= 0.0])):
                    steps_in_plasma = len(ray_dict["rhop"][ray_dict["rhop"] >= 0.0])
                ray_list.append(dict(ray_dict))
                ray_BPD_spl_list.append(InterpolatedUnivariateSpline(ray_list[-1]["s"], ray_list[-1]["BPD"]))
        BPD_ray_dict = ray_list[0]  # Central ray
        rhop_BPD_ray = BPD_ray_dict["rhop"]
        s_ray = BPD_ray_dict["s"]
        BPD_ray = BPD_ray_dict["BPD"]
        BPD_ray_spl = InterpolatedUnivariateSpline(s_ray, BPD_ray)
        if(BPD_ray_spl.integral(s_ray[0], s_ray[-1]) == 0):
            raise ValueError("Error birthplace distribution is zero")
        R_BPD_ray = np.sqrt(BPD_ray_dict["x"] ** 2 + BPD_ray_dict["y"] ** 2)
        R_spl = InterpolatedUnivariateSpline(s_ray, R_BPD_ray)
        rhop_spl = InterpolatedUnivariateSpline(s_ray, rhop_BPD_ray)
        n_rhop = steps_in_plasma
        rhop_binned = np.linspace(0.0, 1.0, n_rhop)
        BPD_binned = np.zeros(n_rhop)
        BPD_ch_binned = np.zeros(n_rhop)
        for i in range(len(rhop_binned)):
            # Recalculate BDOP for an unsigned rhop grid
            for ray, ray_BPD_spl, ray_weight in zip(ray_list, ray_BPD_spl_list, Results.weights["ray"][itime][ich]):
                root_spl_ray = InterpolatedUnivariateSpline(ray["s"], ray["rhop"] - rhop_binned[i])
                for root in root_spl_ray.roots():
                    BPD_ch_binned[i] += ray_BPD_spl(root) * ray_weight
        BPD_POI = []
        BPD_POI_rhop = []
        s_BPD_max = s_ray[np.argmax(BPD_ray)]
        if(use_fit_for_s_important):
            # Fit a gaussian to get the 3 radial points for the 3D BPD cuts
            sigma_BPD_ray_spl = InterpolatedUnivariateSpline(s_ray, BPD_ray * (s_ray - s_BPD_max) ** 2)
            sigma_guess = np.sqrt(sigma_BPD_ray_spl.integral(s_ray[0], s_ray[-1]) / BPD_ray_spl.integral(s_ray[0], s_ray[-1]))
            beta0 = np.array([np.max(BPD_ray), s_BPD_max, sigma_guess])
            data = odr.Data(s_ray, BPD_ray)
            mdl = odr.Model(func)
            ODR = odr.ODR(data, mdl, beta0)
            output = ODR.run()
            beta = output.beta
            max_shift = np.abs(s_BPD_max - beta[1]) / sigma_guess
            print(max_shift)
            if(max_shift > 0.2):
                print("Discarding fit results cause BPD seems very skewed -> initial guess most likely more accurate")
                beta = beta0
            s_important = [s_BPD_max, s_BPD_max + beta[2], s_BPD_max - beta[2]]
        else:
            # Use the integral of the birthplace distribution to determine a pseudo sigma 
            # analogous to the normal distribution
            s_important =  []
            # Compute norm, since BPD is normalized only in s
            BPD_int_spl = InterpolatedUnivariateSpline(rhop_binned, BPD_ch_binned).antiderivative(1)
            BPD_norm = BPD_int_spl(rhop_binned[-1])    
            for cum_BPD_val in [0.5 - 0.31731, 0.5, 0.5 + 0.31731]:# Confidence interval
                root_spl = InterpolatedUnivariateSpline(rhop_binned, BPD_int_spl(rhop_binned)/BPD_norm - cum_BPD_val)
                roots_cum_BPD = root_spl.roots()
                if(len(roots_cum_BPD) != 1):
                    print("Found " + str(len(root_spl)) + " roots when looking for rhop where BPD at ", cum_BPD_val)
                    print("Discarding this value")
                else:
                    rhop_BPD_root_spl = InterpolatedUnivariateSpline(s_ray, rhop_BPD_ray - roots_cum_BPD)
                    s_roots = rhop_BPD_root_spl.roots()
                    i_s_closest = np.argmin(np.abs(s_roots - s_BPD_max))
                    s_important.append(s_roots[i_s_closest])
        for s in s_important:
            BPD_POI.append(R_spl(s))
            BPD_POI_rhop.append(rhop_spl(s))
        if(len(s_important) > 1):
            distribution_rhop = np.abs(rhop_spl(s_important[1])) # Always the second s important
        else:
            distribution_rhop = np.abs(rhop_spl(s_important[0]))
        R_BPD_list.append(BPD_POI)
        R_BPD_dict[str(ich)] = BPD_POI
        rhop_BPD_dict[str(ich)] = BPD_POI_rhop
        s_BPD_list.append(s_important)
        s_BPD_dict[str(ich)] = s_important
        label = "BPD"
        if(single_ray_BPD):
            ax_depo.plot(rhop_binned, BPD_binned / np.max(BPD_binned), label="BPD", linestyle="-", color="blue")  # for channel {0:d}".format(ich)
            ax_depo.plot(rhop_binned, BPD_ch_binned / np.max(BPD_ch_binned), label=r"BPD $5 \times 5$ rays", marker="+", linestyle="None", color="blue")  # for channel {0:d}".format(ich)
        else:
            ax_depo.plot(rhop_binned, BPD_ch_binned / np.max(BPD_ch_binned), label="BPD", linestyle="-", color="blue")  # for channel {0:d}".format(ich)
    f_inter, f_inter_scnd = make_f_inter(dist, EQObj, dist_obj=dist_obj, time=time)
    for ch, m_ch, s_important in zip(ch_list, m_list, s_BPD_list):
        svec, freq, Trad, T, = load_data_for_3DBDOP(Results, time, dist, ch) # expects channel number not channel index
        BDOP_list.append(BDOP_3D(s_important, svec, freq, Trad, T, f_inter, dist, B_ax, m=m_ch, f_inter_scnd=f_inter_scnd))
        m = cm.ScalarMappable(cmap=plt.cm.get_cmap("winter"))
        m.set_array(np.linspace(0.0, 1.0, 20))
        cmaps.append(m)
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
            ax_depo.plot(linear_beam.rhop, PW_beam / Beam_max, label=label, linestyle="--", color=color)
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
                BDOP_list.append(PowerDepo_3D(freq, beam[0], f_inter, dist, B_ax, EqSlice, Te_spline, ne_spline, m=m_ECRH, only_contribution=only_contribution, steps=500, s_important=s_important))
                m = cm.ScalarMappable(cmap=plt.cm.get_cmap("spring"))
                m.set_array(np.linspace(0.0, 1.0, 20))
                cmaps.append(m)
                is_ecrh_list.append(1)
                R_BPD_list.append(PDP_POI)
                s_BPD_list.append(s_important)
        ax_depo.plot(quasi_linear_beam.rhop, quasi_linear_beam.PW / np.max(quasi_linear_beam.PW), label="ECRH (RELAX)", linestyle="None", marker="^", color="black")
    if(len(BDOP_list) > 1):
        leg = ax_depo.legend()
        leg.draggable()
    te_ax = ax_depo.twinx()
    te_ax.plot(rhop_Te, np.exp(Te) * 1.e-3, "--", label="$T_\mathrm{e}$")
    te_ax.set_ylabel(r"$T_\mathrm{e}$ [\si{\kilo\electronvolt}]")
    ax_depo.set_xlabel(r"$\rho_\mathrm{pol}$")
    if(include_ECRH):
        ax_depo.set_ylabel(r"$D_\omega\, \mathrm{and} \, \mathrm{d}P/\mathrm{d}R\,[\si{{a.u.}}]$")
    else:
        ax_depo.set_ylabel(r"$D_\omega\,[\si{{a.u.}}]$")
    steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    ax_depo.get_xaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps, prune="lower"))
    ax_depo.get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 2.0))
    plt.autoscale(False)
    for key in R_BPD_dict.keys():
        if("2" in  key):
            color = ECRH_colors[1]
        else:
            color = ECRH_colors[0]
        if("ECRH" not in key):
            ax_depo.vlines(rhop_BPD_dict[key], -300, 1000, linestyle="-.", color="blue")
        else:
            ax_depo.vlines(rhop_BPD_dict[key], -300, 1000, linestyle=":", color=color)
    ax_depo.set_ylim(-0.01, 1.2)
    ax_reso = fig.add_subplot(122)
    mdict["BPD_vals"] = []
    mdict["BPD_rho"] = []
    mdict["BPD_u_par"] = []
    mdict["BPD_u_perp"] = []
    mdict["BPD_facecolors"] = []
    mdict["is_ecrh_list"] = []
    got_f = False
    found_one_cut = False
    for BDOP, cmap, s_list in zip(BDOP_list, cmaps, s_BPD_list):
        if(len(BDOP.s) == 0):
            print("No BPD values computed, this is not supposed to happen ..., skpping 3D BPD slices...")
            continue
        found_one_cut = True
        BDOP_s = np.mean(BDOP.s, axis=1)
        for s_max in s_list:
            i_s = np.argmin(np.abs(BDOP_s - s_max))
            print("s found vs. inquired s", BDOP_s[i_s], s_max)
            BDOP_val_norm = BDOP.val[i_s] / np.max(BDOP.val[i_s])
            linecolor = cmap.to_rgba(BDOP_val_norm, 1.0)
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
                if(dist not in ["Ge", "GeComp", "Th"]):
                    zeta = BDOP.zeta[i_s]
                got_f = True
    N_f = 300
    f = np.zeros((N_f, 2 * N_f))
    print("U_perp /U_par range", u_perp_range, u_par_range)
    u_par_extend = u_par_range[1] - u_par_range[0]
    u_par_dist = np.linspace(u_par_range[0] - 0.5 * u_par_extend, u_par_range[1] + u_par_extend * 0.5, 2 * N_f)
    u_perp_dist = np.linspace(u_perp_range[0], u_perp_range[1] * 2.0, N_f)
    for i in range(len(u_perp_dist)):
        if(dist not in ["Ge", "GeComp", "Th"]):
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
        f[f < 1.e-20] = 1.e-20
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
    if(found_one_cut):
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
    return fig
    
if(__name__ == "__main__"):
    x = np.linspace(0,1,30)
    distribute_points(x, 0.2 + 5 * np.exp(-(x-0.5)**2 / 0.05**2), 30)
#     make_3DBDOP_cut_standalone("/tokp/work/sdenk/ECRad/ECRad_35662_ECECTACTC_ed7.mat", 4.40, [81], [2], "Re", \
#                     include_ECRH=True, m_ECRH_list=[2], only_contribution=True, \
#                     ECRH_freq=105.e9, mat_for_waves_and_distribution = \
#                     "/tokp/work/sdenk/ECRad/ECRad_35662_ECECTACTC_ed9.mat")  # 
