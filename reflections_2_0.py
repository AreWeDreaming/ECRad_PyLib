'''
Created on Jan 18, 2018

@author: sdenk
'''
import os
import sys
from plotting_configuration import *
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from winding_number import wn_PnPoly
from equilibrium_utils_AUG import EQData
from electron_distribution_utils import find_cold_res
from shotfile_handling_AUG import get_data_calib, Diag

def I_0(shot, time, folder, ch, mode, eq_slice):
    x_w, y_w = np.loadtxt(os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Pylib", "ASDEX_Upgrade_vessel.txt"), skiprows=1, unpack=True)
    circ_nume = 0.0
    R_torus = 1.65
    for i in range(1, len(x_w)):
        circ_nume += np.sqrt((x_w[i] - x_w[i - 1]) ** 2 + (y_w[i] - y_w[i - 1]) ** 2)
    A_torus = circ_nume * R_torus * 2.0 * np.pi
    R_wall = 0.95
    Tradfile = np.loadtxt(os.path.join(folder, "O_TRadM_therm.dat"))
    tau_0 = Tradfile.T[2][ch - 1]
    z, Trad, T_e_0 = Trad_cord(shot, time, folder, ch, mode, eq_slice)
    mean_tau = tau_0 * Trad ** 2 / T_e_0 ** 2
    Trad_int_spl = InterpolatedUnivariateSpline(z, Trad * (1.0 - np.exp(-mean_tau)))
    tau_int_spl = InterpolatedUnivariateSpline(z, (1.0 - np.exp(-mean_tau)))
    I_0 = Trad_int_spl.integral(np.min(z), np.max(z)) / ((1.0 - R_wall) * A_torus / (4.0 * np.pi * R_torus) + tau_int_spl.integral(np.min(z), np.max(z)))
    return I_0, T_e_0


def compare_reflections(shot, time, folder, mode, eq_exp, eq_diag, eq_ed):
    eq_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    eq_slice = eq_obj.GetSlice(time)
    Tradfile = np.loadtxt(os.path.join(folder, "O_TRadM_therm.dat"))
    ECRad_inp = np.loadtxt(os.path.join(folder, "ECRad.inp"), dtype=np.str)
    R_wall = float(ECRad_inp[14])
    print("R_wall", R_wall)
    f_ECE = np.loadtxt(os.path.join(folder, "f_ECE.dat"))
    f_upper = 100.e9
    Trad_without_refl = Tradfile.T[1][f_ECE < f_upper] * (1.0 - R_wall * np.exp(-Tradfile.T[2][f_ECE < f_upper]))
    Trad_specular = []  # Tradfile.T[1][f_ECE < f_upper]
    channels = np.linspace(1, len(f_ECE), len(f_ECE), dtype=np.int)
    channels = channels[f_ECE < f_upper]
    Trad_0 = []
    for ich in channels:
        Trad_0_val, Te_0 = I_0(shot, time, folder, ich, mode, eq_slice)
        Trad_specular.append(Te_0 * (1.0 - np.exp(-Tradfile.T[2][ich])) / (1.0 - R_wall * np.exp(-Tradfile.T[2][ich])))
        Trad_0.append(Trad_0_val)
    Trad_0 = np.array(Trad_0) / 1.e3
    Trad_specular = np.array(Trad_specular) / 1.e3
    rho = Tradfile.T[0][f_ECE < f_upper]
    plt.plot(rho, Trad_0 * (1.0 - np.exp(-Tradfile.T[2][channels - 1])) + Trad_without_refl, "-")
    plt.plot(rho, Trad_0 , "d")
    plt.plot(rho, Trad_specular, "--")
    plt.plot(rho, Trad_without_refl, ":")
    plt.plot(rho, Tradfile.T[1][f_ECE < f_upper], "+")
    plt.show()

def Mode_conversion_contributions_simple(mode_conversion_ratio, shot, time, folder, mode, eq_exp, eq_diag, eq_ed):
    eq_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    eq_slice = eq_obj.GetSlice(time)
    Tradfile = np.loadtxt(os.path.join(folder, "TRadM_therm.dat"))
    Tradfile_X = np.loadtxt(os.path.join(folder, "X_TRadM_therm.dat"))
    X_frac = Tradfile_X.T[3]
    ECRad_inp = np.loadtxt(os.path.join(folder, "ECRad.inp"), dtype=np.str)
    R_wall = float(ECRad_inp[14])
    print("R_wall", R_wall)
    f_ECE = np.loadtxt(os.path.join(folder, "f_ECE.dat"))
    f_upper = 100.e9
    channels = np.linspace(1, len(f_ECE), len(f_ECE), dtype=np.int)
    channels = channels[f_ECE < f_upper]
    Trad_0 = []
    rho = Tradfile.T[0][f_ECE < f_upper]
    for ich in channels:
        Trad_0_val, Te_0 = I_0(shot, time, folder, ich, "O", eq_slice)
        Trad_0.append(Trad_0_val)
    Trad_0 = np.array(Trad_0) / 1.e3
    plt.plot(rho, Tradfile.T[1][f_ECE < f_upper], "+", label=r"Conventional reflection model")
    plt.plot(rho, Tradfile.T[1][f_ECE < f_upper] + mode_conversion_ratio * Trad_0 * X_frac[f_ECE < f_upper], "s", label=r"Reflection model with 20 \% mode conversion")
    ECE_diag = Diag("ECE", "AUGD", "RMD", 0)
    y_err, ECE_data = get_data_calib(diag=ECE_diag, shot=shot, time=time, ext_resonances=Tradfile.T[0])
    plt.errorbar(rho, ECE_data[1][f_ECE < f_upper], y_err[0][f_ECE < f_upper] + y_err[1][f_ECE < f_upper], label=r"Measured spectrum")
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.gca().set_ylabel(r"$T_\mathrm{rad}\,[\si{\kilo\electronvolt}]$")
    leg = plt.legend()
    leg.draggable()
    plt.tight_layout()
    plt.show()


def Trad_cord(shot, time, folder, ch, mode, eq_slice):
    rho_spl = RectBivariateSpline(eq_slice.R, eq_slice.z, eq_slice.rhop)
    sucess, s_res, R_res, z_res, rho_res = find_cold_res(folder, ch, mode="O")
    z_eval = eq_slice.z
    z_eval = z_eval[np.abs(eq_slice.z) < 1.0]  # Avoid private flux region
    R_cut = np.zeros(len(z_eval))
    R_cut[:] = R_res
    rhop_eval = rho_spl(R_cut, z_eval, grid=False)
    rhop, Te = np.loadtxt(os.path.join(folder, "Te_file.dat"), skiprows=1, unpack=True)
    Te_spl = InterpolatedUnivariateSpline(rhop, Te)
    z_eval = z_eval[rhop_eval < np.max(rhop)]
    rhop_eval = rhop_eval[rhop_eval < np.max(rhop)]
    return z_eval, Te_spl(rhop_eval), Te_spl(rho_res)

def ContourArea():
    x_w, y_w = np.loadtxt(os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Pylib", "ASDEX_Upgrade_vessel.txt"), skiprows=1, unpack=True)
    s = np.linspace(0, 1, len(x_w))
    x_w_spl = InterpolatedUnivariateSpline(s, x_w)
    a_spl = InterpolatedUnivariateSpline(s, y_w * x_w_spl(s, nu=1))
    Area = a_spl.integral(0.0, 1.0)
    print(Area)
    x_0 = np.min(x_w)
    x_1 = np.max(x_w)
    y_0 = np.min(y_w)
    y_1 = np.max(y_w)
    N_monte = 100000
    x_random = x_0 + np.random.random(N_monte) * (x_1 - x_0)
    y_random = y_0 + np.random.random(N_monte) * (y_1 - y_0)
    poly = []
    for i in range(len(x_w)):
        poly.append([x_w[i], y_w[i]])
    poly = np.array(poly)
    hits = 0
    for i in range(N_monte):
        if(wn_PnPoly(np.array([x_random[i], y_random[i]]), poly) != 0):
            hits += 1
    Area_monte = (x_1 - x_0) * (y_1 - y_0) * float(hits) / float(N_monte)
    print(Area_monte)

if __name__ == "__main__":
    Mode_conversion_contributions_simple(1.0 / 5.0, 32934, 3.30, "/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_8/ECRad_data/", "O", "AUGD", "EQH", 0)
#    compare_reflections(32934, 3.30, "/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_8/ECRad_data/", "O", "AUGD", "EQH", 0)
