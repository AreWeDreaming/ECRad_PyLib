'''
Created on Apr 12, 2016

@author: Severin Denk
'''
from ecrad_pylib.TB_Communication import read_topfile
import sys
# from kk_abock import kk as KKeqi
# from kk_extra import kk_extra
from ecrad_pylib.Diag_Types import ECRH_diag
from ecrad_pylib.Global_Settings import globalsettings
if(globalsettings.AUG):
    import dd
    from Equilibrium_Utils_AUG import EQData
    from Shotfile_Handling_AUG import get_data_calib
from ecrad_pylib.Plotting_Configuration import plt
import numpy as np
import os
from scipy import constants as cnst
from ecrad_pylib.Plotting_Core import PlottingCore
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.integrate import simps
from ecrad_pylib.Em_Albajar import EmAbsAlb, SVec, DistributionInterpolator, GeneDistributionInterpolator
from ecrad_pylib.ECRad_Interface import read_svec_dict_from_file
from ecrad_pylib.TB_Communication import make_topfile_from_ext_data
from ecrad_pylib.Equilibrium_Utils import EQDataExt
from scipy.io import loadmat
from ecrad_pylib.ECRad_Results import ECRadResults
from ecrad_pylib.Distribution_Functions import Juettner2D, Gauss_norm, Gauss_not_norm, \
                                   Juettner2D_bidrift, multi_slope, RunAway2D
from ecrad_pylib.Distribution_IO import load_f_from_ASCII, read_LUKE_data


def debug_append_ECRadResults(filename):
    results = ECRadResults()
    results.from_mat_file(filename) # Get scenario and launch
    results.reset()
    results.append_new_results(1.5)
    results.tidy_up(True)

def compare_ECRad_results(result_file_list, time, ch, ir=1):
    main_quant = "ray"
    subquantx = "sX"
    subquanty = "abX"
    factor = 1.0
    res = ECRadResults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for result_file,marker in zip(result_file_list,["-", "--"]):
        res.from_mat_file(result_file)
        itime = np.argmin(np.abs(time - res.time))
        if(res.Config["Physics"]["N_ray"] > 1):
            ax.plot(getattr(res, main_quant)[subquantx][itime][ch - 1], np.rad2deg(1) * getattr(res, main_quant)[subquanty][itime][ch - 1][ir-1] * factor, marker)
        else:
            ax.plot(getattr(res, main_quant)[subquantx][itime][ch - 1], np.rad2deg(1) * getattr(res, main_quant)[subquanty][itime][ch - 1] * factor, marker)
#         ax.vlines(res.resonance[subquantx[:-1] + "_cold"][itime][ch-1], 0 , 2000, linestyle=marker )
#     ax.hlines(0.5, 0, 4)
    plt.show()


def compare_eq_midplane(shot, time, eq_diag_1, eq_diag_2, eq_exp_1="AUGD", eq_exp_2="AUGD", \
                        eq_ed_1=0, eq_ed_2=0):
    eq_obj_1 = EQData(shot, EQ_exp=eq_exp_1, EQ_diag=eq_diag_1, EQ_ed=eq_ed_1)
    eq_obj_2 = EQData(shot, EQ_exp=eq_exp_2, EQ_diag=eq_diag_2, EQ_ed=eq_ed_2)
    eq_slice_1 = eq_obj_1.GetSlice(time)
    eq_slice_2 = eq_obj_2.GetSlice(time)
    i_z = np.argmin(np.abs(eq_slice_1.z - eq_slice_1.z_ax))
    plt.plot(eq_slice_1.rhop[...,i_z], np.abs(1 - eq_slice_1.Bt[...,i_z]/eq_slice_2.Bt[...,i_z]) * 1.e2, "--", \
            label="$z = " + "{0:1.1f}".format(eq_slice_1.z[i_z] * 1.e2) + r"$ [cm]")
    print(eq_slice_1.z[i_z], eq_slice_2.z[i_z])
    
def compare_eq_Rz(s1, R1, z1, val1, s2, R2, z2, val2, shot, time, eq_diag_1, eq_diag_2, eq_exp_1="AUGD", eq_exp_2="AUGD", \
                        eq_ed_1=0, eq_ed_2=0):
    eq_obj_1 = EQData(shot, EQ_exp=eq_exp_1, EQ_diag=eq_diag_1, EQ_ed=eq_ed_1)
    eq_obj_2 = EQData(shot, EQ_exp=eq_exp_2, EQ_diag=eq_diag_2, EQ_ed=eq_ed_2)
    eq_slice_1 = eq_obj_1.GetSlice(time,False)
    eq_slice_2 = eq_obj_2.GetSlice(time,False)
    B_tot_spl_1 = RectBivariateSpline(eq_slice_1.R, eq_slice_1.z, \
                                      np.sqrt(eq_slice_1.Br**2 + eq_slice_1.Bt**2 + \
                                              eq_slice_1.Bz**2))
    B_tot_spl_2 = RectBivariateSpline(eq_slice_2.R, eq_slice_2.z, \
                                      np.sqrt(eq_slice_2.Br**2 + eq_slice_2.Bt**2 + \
                                              eq_slice_2.Bz**2))
    mask1 = np.ones(len(R1), dtype=bool)
    mask1[R1 < np.min(eq_slice_1.R)] = False
    mask1[R1 > np.max(eq_slice_1.R)] = False
    mask2 = np.ones(len(R2), dtype=bool)
    mask2[R2 < np.min(eq_slice_2.R)] = False
    mask2[R2 > np.max(eq_slice_2.R)] = False
    s_max = min(np.max(s1[mask1]), np.max(s2[mask2]))
    s = np.linspace(0.0, s_max, 2000)
    R1_spl = InterpolatedUnivariateSpline(s1[mask1], R1[mask1])
    z1_spl = InterpolatedUnivariateSpline(s1[mask1], z1[mask1])
    val1_spl  = InterpolatedUnivariateSpline(s1[mask1], val1[mask1])
    R2_spl = InterpolatedUnivariateSpline(s2[mask2], R2[mask2])
    z2_spl = InterpolatedUnivariateSpline(s2[mask2], z2[mask2])
    val2_spl  = InterpolatedUnivariateSpline(s2[mask2], val2[mask2])
#     plt.plot(s, R1_spl(s))
#     plt.plot(s, R2_spl(s),"--")
    plt.plot(R1_spl(s), 1.e2*(1-B_tot_spl_1(R1_spl(s),z1_spl(s),grid=False) / \
                        B_tot_spl_2(R2_spl(s),z2_spl(s),grid=False)), "-",label="Before BTFABB correction")
#     eq_obj_1.ApplyBVacCorrectionToSlice(eq_slice_1)
#     eq_obj_2.ApplyBVacCorrectionToSlice(eq_slice_2)
#     B_tot_spl_1 = RectBivariateSpline(eq_slice_1.R, eq_slice_1.z, \
#                                       np.sqrt(eq_slice_1.Br**2 + eq_slice_1.Bt**2 + \
#                                               eq_slice_1.Bz**2))
#     B_tot_spl_2 = RectBivariateSpline(eq_slice_2.R, eq_slice_2.z, \
#                                       np.sqrt(eq_slice_2.Br**2 + eq_slice_2.Bt**2 + \
#                                               eq_slice_2.Bz**2))
#     plt.plot(s, 1.e2*(1-B_tot_spl_1(R1_spl(s),z1_spl(s),grid=False) / \
#                         B_tot_spl_2(R2_spl(s),z2_spl(s),grid=False)), "-",label="After BTFABB correction")
#     plt.plot(s, 1.e2*(1-val1_spl(s)/val2_spl(s)), "--", label="Ray")
#     plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
#     plt.gca().set_ylabel(r"1 - $B_\mathrm{t,EQH} / B_\mathrm{t,IDE}$ [\%]")
#     plt.suptitle(r"$z = " + "{0:1.1f}".format(eq_slice_1.z[i_z] * 1.e2) + r"$ [cm]")
# #     plt.plot(eq_slice_2.R, eq_slice_2.Bt[...,i_z], "--")
#     plt.tight_layout()
#     plt.show()
    
def compare_ECRad_results_diff(primary_result_file, result_file_list, time, ch, ir=1, label=None):
    main_quant = "ray"
    subquantx = "sX"
    subquanty = "YX"
    prim_res = ECRadResults()
    prim_res.from_mat_file(primary_result_file)
    print(prim_res.Scenario.bt_vac_correction)
    itime_prim = np.argmin(np.abs(time - prim_res.time))
    f_ECE = prim_res.Scenario.ray_launch[itime_prim]["f"][ch -1]
    scale = cnst.m_e* f_ECE*2*np.pi  / cnst.e
    if(prim_res.Config["Physics"]["N_ray"] > 1):
        val = getattr(prim_res, main_quant)[subquanty][itime_prim][ch - 1][ir-1]
        prim_spl = InterpolatedUnivariateSpline(getattr(prim_res, main_quant)["s" + subquanty[-1]][itime_prim][ch - 1][ir-1], \
                                                val)
        s = getattr(prim_res, main_quant)["s" + subquanty[-1]][itime_prim][ch - 1][ir-1]
        R = np.sqrt(getattr(prim_res, main_quant)["x" + subquanty[-1]][itime_prim][ch - 1][ir-1]**2 + \
                    getattr(prim_res, main_quant)["y" + subquanty[-1]][itime_prim][ch - 1][ir-1]**2)
        z = getattr(prim_res, main_quant)["z" + subquanty[-1]][itime_prim][ch - 1][ir-1]
    else:
        val = getattr(prim_res, main_quant)[subquanty][itime_prim][ch - 1]
        prim_spl = InterpolatedUnivariateSpline(getattr(prim_res, main_quant)["s" + subquanty[-1]][itime_prim][ch - 1], \
                                                val)
        s = getattr(prim_res, main_quant)["s" + subquanty[-1]][itime_prim][ch - 1]
        R = np.sqrt(getattr(prim_res, main_quant)["x" + subquanty[-1]][itime_prim][ch - 1]**2 + \
                    getattr(prim_res, main_quant)["y" + subquanty[-1]][itime_prim][ch - 1]**2)
        z = getattr(prim_res, main_quant)["z" + subquanty[-1]][itime_prim][ch - 1]
    res = ECRadResults()
    fig_ray = plt.figure()
    ax_ray = fig_ray.add_subplot(111)
    ax_ray.plot(R,z*1.e2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for result_file,marker in zip(result_file_list,["-", "--"]):
        res.from_mat_file(result_file)
        print(res.Scenario.bt_vac_correction)
        itime = np.argmin(np.abs(time - res.time))
        if(res.Config["Physics"]["N_ray"] > 1):
            s_res = getattr(res, main_quant)["s" + subquanty[-1]][itime_prim][ch - 1][ir-1]
            R_res = np.sqrt(getattr(res, main_quant)["x" + subquanty[-1]][itime_prim][ch - 1][ir-1]**2 + \
                            getattr(res, main_quant)["y" + subquanty[-1]][itime_prim][ch - 1][ir-1]**2)
            z_res = getattr(res, main_quant)["z" + subquanty[-1]][itime_prim][ch - 1][ir-1]
            ax_ray.plot(R_res, z_res*1.e2, "--")
            #getattr(res, main_quant)[subquantx][itime][ch - 1][ir-1]
            val = (1.0 - prim_spl(getattr(res, main_quant)["s" + subquanty[-1]][itime][ch - 1][ir-1])/\
                                        getattr(res, main_quant)[subquanty][itime][ch - 1][ir-1])
#             ax.plot(R_res, 1.e2* (1.0 - prim_spl(getattr(res, main_quant)["s" + subquanty[-1]][itime][ch - 1][ir-1])/\
#                                         getattr(res, main_quant)[subquanty][itime][ch - 1][ir-1]), marker, label=label)
        else:
            s_res = getattr(res, main_quant)["s" + subquanty[-1]][itime_prim][ch - 1]
            R_res = np.sqrt(getattr(res, main_quant)["x" + subquanty[-1]][itime_prim][ch - 1]**2 + \
                            getattr(res, main_quant)["y" + subquanty[-1]][itime_prim][ch - 1]**2)
            z_res = getattr(res, main_quant)["z" + subquanty[-1]][itime_prim][ch - 1]
            ax_ray.plot(R_res, z_res*1.e2, "--")
            #getattr(res, main_quant)[subquantx][itime][ch - 1]\
            
            val_res = getattr(res, main_quant)[subquanty][itime][ch - 1]
#             ax.plot(R_res, 1.e2* (1.0 - prim_spl(getattr(res, main_quant)["s" + subquanty[-1]][itime][ch - 1])/\
#                                               getattr(res, main_quant)[subquanty][itime][ch - 1]), marker, label=label)
#         ax.vlines(res.resonance[subquantx[:-1] + "_cold"][itime][ch-1], 0 , 2000, linestyle=marker )
#     ax.hlines(0.5, 0, 4)
    plt.gca().set_xlabel(r"$s$ [m]")
    plt.gca().set_ylabel(r"$ 1 - \vert\vec{B}_\mathrm{EQH} \vert /  \vert \vec{B}_\mathrm{IDE} \vert$ [\%]")
#     plt.suptitle(r"Channel no. " + "{0:d}".format(ch))
    plt.tight_layout()
    return s, R, z, val*scale, s_res, R_res, z_res, val_res*scale
#     plt.show()

def compare_ECRad_results_ds(result_file_list, time, ch, ir=1):
    res = ECRadResults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for result_file,marker in zip(result_file_list,["-", "--"]):
        res.from_mat_file(result_file)
        itime = np.argmin(np.abs(time - res.time))
        ax.plot(res.ray["sX"][itime][ch - 1][:-1], (res.ray["sX"][itime][ch - 1][1:] - res.ray["sX"][itime][ch - 1][:-1])/ np.max(res.ray["sX"][itime][ch - 1][1:] - res.ray["sX"][itime][ch - 1][:-1]), marker)    
        ax.plot(res.ray["sX"][itime][ch - 1], res.ray["BPDX"][itime][ch - 1] / np.max(res.ray["BPDX"][itime][ch - 1]), "+")
        ax.vlines(res.resonance["s_cold"][itime][ch-1], 0 , 1)
#     ax.hlines(0.5, 0, 4)
    plt.show()
    
def compare_ECRad_Trad(result_file_A, result_file_B, time):
    res_A = ECRadResults()
    res_B = ECRadResults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res_A.from_mat_file(result_file_A)
    res_B.from_mat_file(result_file_B)
    itime_A = np.argmin(np.abs(time - res_A.time))
    itime_B = np.argmin(np.abs(time - res_B.time))
    ax.plot(res_A.resonance["rhop_cold"][itime_A], res_A.Trad[itime_A] - res_A.Trad[itime_B])
    plt.show()
    
def compare_ECRad_results_R(result_file_list, time, ch, ir=1):
    res = ECRadResults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for result_file,marker in zip(result_file_list,["+", "*"]):
        res.from_mat_file(result_file)
        itime = np.argmin(np.abs(time - res.time))
        R = np.sqrt(res.ray["xX"][itime][ch - 1]**2 + res.ray["yX"][itime][ch - 1]**2)
        ax.plot(res.ray["sX"][itime][ch - 1], R, marker)
#         ax.vlines(res.resonance["s_cold"][itime][ch-1], 0 , 1.e-2)
#     ax.hlines(0.5, 0, 4)
    plt.show()



def debug_f_remap(working_dir):
    fig_list = []
    ax_list = []
    data_list = []
#    data_to_plot = ["test_f_int.txt", "test_f_int_du.txt", "test_f_int_dpitch.txt", \
#                    "test_f_int_duperp.txt", "test_f_int_dupar.txt", \
#                    "finite_diff_par.txt", "finite_diff_perp.txt", \
#                    "finite_diff_par.txt", "finite_diff_perp.txt"]
#    coloumns_to_plot = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [3, 4], [3, 4]]
#    markers = [["-", "--", "+"], ["-", "--"], ["-", "--"], ["-", "--"], \
#               ["-", "--"], ["-", "--"], ["-", "--"], ["-", "--"], ["-", "--"]]
    data_to_plot = [ "finite_difg_par.txt", "finite_difg_perp.txt", \
                    "finite_difg_par.txt", "finite_difg_perp.txt"]
    coloumns_to_plot = [[1, 2], [1, 2], [3, 4], [3, 4]]
    markers = [["-", "--"], ["-", "--"], ["-", "--"], ["-", "--"]]
    labels = [["dmu/dupar ana", "dmu/dupar num"], \
              ["dmu/duperp ana", "dmu/duperp num"], \
              ["dvpar/dupar ana", "dvpar/dupar num"], \
              ["dvpar/duperp ana", "dvpar/duperp num"]]
    print("Creating debug plots")
    for i in range(len(data_to_plot)):
        try:
            data_list.append(np.loadtxt(os.path.join(working_dir, data_to_plot[i])))
            fig_list.append(plt.figure())
            ax_list.append(fig_list[-1].add_subplot(111))
            print("Found " + data_to_plot[i])
            ilabel = 0
            for icol in coloumns_to_plot[i]:
                ax_list[-1].plot(data_list[-1].T[0], data_list[-1].T[icol], markers[i][ilabel], label=labels[i][ilabel])
                ilabel += 1
            lns = ax_list[-1].get_lines()
            labs = [l.get_label() for l in lns]
            ax_list[-1].legend(lns, labs)
        except IOError:
            print(data_to_plot[i] + " not available at " + os.path.join(working_dir, data_to_plot[i]))
    plt.show()

def debug_f_inter(path, shot, time, channelno, dstf, mode, rhop_in, HFS, beta, \
                                    eq_exp="AUGD", eq_diag="EQH", eq_ed=0):
        ecfm_data = os.path.join(path, "ecfm_data")
        svec_dict = read_svec_dict_from_file(ecfm_data, channelno, mode=mode)[0]
        flag_use_ASCII = True
        Te_ext = -1
        if(np.abs(mode) != 1):
            print("Mode has to be either +1 (X) or -1 (O). mode:", mode)
            return
        elif(mode == -1):
            print("O mode selected")
        elif(mode == 1):
            print("X mode selected")
        Te_filename = os.path.join(ecfm_data, "Te_file.dat")
        Te_data = np.loadtxt(Te_filename, skiprows=1)
        rhop_vec_Te = Te_data.T[0]
        Te_vec = Te_data.T[1]
        ne_filename = os.path.join(ecfm_data, "ne_file.dat")
        ne_data = np.loadtxt(ne_filename, skiprows=1)
        rhop_vec_ne = ne_data.T[0]
        ne_vec = ne_data.T[1]
        Te_spl = InterpolatedUnivariateSpline(rhop_vec_Te, Te_vec)
        ne_spl = InterpolatedUnivariateSpline(rhop_vec_ne, ne_vec)
        Alb = EmAbsAlb()
        rhop = rhop_in
        EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed))
        B0 = EQ_obj.get_B_min(time, rhop_in, append_B_ax=True)
        if(dstf == "Re" or dstf == "Lu"):
            if(flag_use_ASCII):
                f_folder = os.path.join(ecfm_data, "f" + dstf)
                dist_obj = load_f_from_ASCII(f_folder, rhop_in)
                x = dist_obj.u
                y = dist_obj.pitch
                Fe = np.log(dist_obj.f)
                rhop = dist_obj.rhop[0]
                Fe = Fe[0]
                # Fe = np.exp(Fe)
                print("Distribution shape", x.shape, y.shape, Fe.shape)
            elif(dstf == "Lu"):
                dist_obj = read_LUKE_data(ecfm_data)
                rhop = dist_obj.rhop
                x = dist_obj.u
                y = dist_obj.rhop
                Fe = dist_obj.f
            else:
                raise ValueError
            # bt_vac correction not necessary here since constant factor that cancels out for B/B_min
        elif(dstf != "Ge"):
            if(dstf == "DM"):
                beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
            elif(dstf == "BJ"):
                beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
                beta[5] = 0.0
                beta[6] = 0.0
            elif(dstf == "BM"):
                beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
                beta[5] = 0.0
                beta[6] = 0.0
            elif(dstf == "MS"):
                beta[0] = Gauss_not_norm(rhop, [1.0, 0.0, beta[0]])
            elif(dstf == "SH"):
                print("dstf = SH is not supported!")
                return
            elif(dstf == "RA"):
                beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
            elif(dstf == "GB"):
                beta[0] = 1.0
                beta[5] = 0.0
                beta[6] = 0.0
            else:
                beta[0] = 0.0
            x = np.linspace(0.0, 3.0, 200)
            y = np.linspace(0.0, np.pi, 200)
            Fe = np.zeros((len(x), len(y)))
            rhop = rhop_in
            if((dstf != "SH" and dstf != "MS") and beta[0] == 0.0):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = Juettner2D(u_par, u_perp, Te_spl(rhop_in))
            elif(dstf != "SH" and dstf != "MS" and dstf != "RA" and dstf != "GB" and beta[0] != 0.0):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = (1.e0 - beta[0]) * (Juettner2D(u_par, u_perp, Te_spl(rhop_in)) + \
                                    beta[0] * Juettner2D_bidrift(u_par, u_perp, beta[3], beta[4], beta[5], beta[6]))
            elif(dstf == "GB"):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = Juettner2D_bidrift(u_par, u_perp, beta[3], beta[4], beta[5], beta[6])
            elif(dstf == "MS"):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = multi_slope(u_par, u_perp, Te_spl(rhop_in), \
                                               beta[1], beta[0] * beta[2])
            elif(dstf == "RA"):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = (RunAway2D(u_par, u_perp, Te_spl(rhop_in), ne_spl(rhop_in), \
                                               beta[0], beta[3], beta[4]))
            else:
                print("The chosen dstf is not supported", dstf)
            Fe[Fe < 1.e-30] = 1.e-30
            Fe = np.log(Fe)
        else:
            f_folder = os.path.join(ecfm_data, "f" + dstf)
            rhop, beta_par, mu_norm, Fe, B0 = load_f_from_ASCII(f_folder, rhop_in, Gene=True)
            # np.abs(g)
            Alb.B_min = B0
        if(dstf != "Ge"):
            dist_obj = DistributionInterpolator(x, y, RectBivariateSpline(x, y, Fe))
            Alb.dist_mode = "ext"
            Alb.ext_dist = dist_obj
            Alb.B_min = B0
        else:
            dist_obj = GeneDistributionInterpolator(beta_par, mu_norm, RectBivariateSpline(beta_par, mu_norm, Fe))
            Alb.dist_mode = "gene"
            Alb.ext_dist = dist_obj
        if(np.min(svec_dict["rhop"]) > rhop):
            print("LOS does not intersect requested flux surface")
            print("Smallest rhop on LOS", np.min(svec_dict["rhop"]))
            return
        npts = 400
        intersect_spl = InterpolatedUnivariateSpline(svec_dict["s"], svec_dict["rhop"] - rhop)
        s_res_list = intersect_spl.roots()
        print("possible resonances", s_res_list)
        omega_c_spl = InterpolatedUnivariateSpline(svec_dict["s"], svec_dict["freq_2X"] * np.pi)
        B_ax = EQ_obj.get_B_on_axis(time)
        omega_c_ax = B_ax * cnst.e / cnst.m_e
        omega_c_max = np.max(svec_dict["freq_2X"]) * np.pi
        if(HFS and omega_c_ax > omega_c_max):
            print("LOS does not cross HFS")
            return
        elif(len(s_res_list > 2) and omega_c_ax > omega_c_max):
            print("More than one resonance found on LFS - the second one will be ignored")
        elif(len(s_res_list > 2)):
            if(HFS):
                FS_str = "HFS"
            else:
                FS_str = "LFS"
            print("More than two resonances found. Picking the one closest to the antenna on the " + FS_str + ".")
        omega_c_res = omega_c_spl(s_res_list)
        if(HFS):
            if(len(s_res_list[omega_c_ax < omega_c_res]) == 0):
                print("All found intersection with the chosen flux surface lie on the LFS")
                return
            s_res = s_res_list[omega_c_ax < omega_c_res][np.argmax(s_res_list[omega_c_ax < omega_c_res])]
        else:
            if(len(s_res_list[omega_c_ax >= omega_c_res]) == 0):
                print("All found intersection with the chosen flux surface lie on the HFS")
                return
            s_res = s_res_list[omega_c_ax >= omega_c_res][np.argmax(s_res_list[omega_c_ax >= omega_c_res])]
        omega_c = omega_c_spl(s_res)
        theta_spl = InterpolatedUnivariateSpline(svec_dict["s"], svec_dict["theta"])
        theta = theta_spl(s_res)
        if(Te_ext <= 0):
            Te = Te_spl(rhop)
        else:
            Te = Te_ext
        ne = ne_spl(rhop)
        svec = SVec(rhop, Te, ne, omega_c / np.pi, theta)
        print("svec", rhop, Te, ne, omega_c / np.pi, theta)
        for i_upar in range(len(beta_par)):
            u_par = np.zeros(npts)
            mu_smooth = np.linspace(np.min(mu_norm), np.max(mu_norm), npts)
            u_perp = np.sqrt(mu_smooth * 2.0 * B0) / np.sqrt(1.0 - beta_par[i_upar] ** 2 - mu_smooth * 2.0 * B0)
            u_par = beta_par[i_upar] / np.sqrt(1.0 - beta_par[i_upar] ** 2 - mu_smooth * 2.0 * B0)
            u_perp_grid = np.sqrt(mu_norm * 2.0 * B0) / np.sqrt(1.0 - beta_par[i_upar] ** 2 - mu_norm * 2.0 * B0)
            log10_f_along_res = []
            if(dstf != "Ge"):
                Alb.dist_mode = "ext"
            else:
                Alb.dist_mode = "gene"
            dist_vals = Alb.dist(u_par, u_perp, (cnst.m_e * cnst.c ** 2) / (svec.Te * cnst.e), svec)
            dist_vals[dist_vals < 1.e-20] = 1.e-20
            log10_f_along_res.append(np.log10(dist_vals))
    #        plt.plot(u_par, u_perp, "-")
    #        plt.plot(u_par_grid, u_perp_grid, "+")
            plt.plot(u_perp, log10_f_along_res[0])
            plt.plot(u_perp_grid, np.log10(np.exp(Fe[i_upar])), "+")
            plt.show()
        return



def u_thermal(Te):
    return np.sqrt(1 - \
            cnst.physical_constants["electron mass"][0] ** 2 * \
            cnst.physical_constants["speed of light in vacuum"][0] ** 4 / \
            (cnst.physical_constants["electron mass"][0] * \
            cnst.physical_constants["speed of light in vacuum"][0] ** 2 + \
            cnst.physical_constants["elementary charge"][0] * Te) ** 2)
# print(u_thermal(2000))

def check_ray_bundle(working_dir, shotno, time, N_ray, channel=1, mode="X", tor_view=False, R_focus=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    crossX = np.loadtxt(os.path.join(working_dir, "ecfm_data", "IchTB", "RayCrossX{0:03d}.dat".format(channel)))
    crossY = np.loadtxt(os.path.join(working_dir, "ecfm_data", "IchTB", "RayCrossY{0:03d}.dat".format(channel)))
    crossZ = np.loadtxt(os.path.join(working_dir, "ecfm_data", "IchTB", "RayCrossZ{0:03d}.dat".format(channel)))
    for i in range(N_ray):
        ray = np.loadtxt(os.path.join(working_dir, "ecfm_data", "ray", "Ray{0:03d}ch{1:03d}_{2:s}.dat".format(i + 1, channel, mode)))
        if(not tor_view):
            if(i == 0):
                ax.plot(np.sqrt(ray.T[1] ** 2 + ray.T[2] ** 2), ray.T[3], "-r")
            else:
                ax.plot(np.sqrt(ray.T[1] ** 2 + ray.T[2] ** 2), ray.T[3], "--b")
        else:
            if(i == 0):
                ax.plot(ray.T[1], ray.T[2], "-r")
            else:
                ax.plot(ray.T[1], ray.T[2], "-b")
    if(R_focus is not None):
        if(not tor_view):
            ax.plot(R_focus[0], R_focus[2], "+r")
        else:
            ax.plot(R_focus[0] * np.cos(R_focus[1]), R_focus[0] * np.sin(R_focus[1]), "+r")
    if(not tor_view):
        R_mat = np.sqrt(crossX ** 2 + crossY ** 2)
        for i in range(len(R_mat.T[::20])):
            ax.plot(R_mat.T[::20][i], crossZ.T[::20][i], "-k")
    else:
        for i in range(len(crossX.T[::20])):
            ax.plot(crossX.T[::20][i], crossY.T[::20][i], "-k")
    # ax.add_patch(patches.Rectangle(\
    #    (1, -1.27), 1.27, 2.54, fill=False, edgecolor="red"))
    # ax.add_patch(patches.Rectangle(\
    #    (1.23, -0.55), 0.7, 1.33, fill=False, edgecolor="blue"))
    plt.show()

def plot_quant_on_LOS(ECRad_result_file, it, ich, ir, x, quant):
    Results = ECRadResults()
    Results.from_mat_file(ECRad_result_file)
    plt.plot(Results.ray[x][it][ich], Results.ray[quant][it][ich])

def validate_theta_along_los(ida_working_dir, ed, ch):
    ida_ecfm_data = os.path.join(ida_working_dir, "ecfm_data")
    if(ed == 0):
        ecfm_data = os.path.join(ida_working_dir, "OERT", "ecfm_data")
    else:
        ecfm_data = os.path.join(ida_working_dir, "OERT", "ed_" + str(ed), "ecfm_data")
    ida_svec_dict = read_svec_dict_from_file(ida_ecfm_data, ch)[0]
    ecfm_svec_dict = read_svec_dict_from_file(ecfm_data, ch)[0]
    fig1 = plt.figure(figsize=(12.0, 8.5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(ida_svec_dict["R"], ida_svec_dict["z"], "-r", label=r"IDA")
    ax1.plot(ecfm_svec_dict["R"], ecfm_svec_dict["z"], "--b", label=r"New ECRad")
    ax1.set_xlabel(r"$R\,[\si{\metre}]$")
    ax1.set_ylabel(r"$z\,[\si{\metre}]$")
    ax1.legend()
    fig2 = plt.figure(figsize=(12.0, 8.5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(ida_svec_dict["R"], np.abs(ida_svec_dict["theta"] - np.pi / 2.e0) * 180.e0 / np.pi, "-r", label=r"IDA")
    ax2.plot(ecfm_svec_dict["R"], np.abs(ecfm_svec_dict["theta"] - np.pi / 2.e0) * 180.e0 / np.pi, "--b", label=r"New ECRad")
    ax2.set_xlabel(r"$R\,[\si{\metre}]$")
    ax2.set_ylabel(r"$\vert\theta - \ang{90}\vert\,[^\circ$]")
    ax2.legend()
    plt.show()

def compare_LOS_Rz(working_dir, ida_working_dir, chno):
    svec_ECRad = np.loadtxt(os.path.join(working_dir, "chdata{0:03d}.dat".format(chno)))
    svec_IDA = np.loadtxt(os.path.join(ida_working_dir, "chdata{0:03d}.dat".format(chno)))
    plt.plot(svec_ECRad.T[1][svec_ECRad.T[3] > 0], svec_ECRad.T[2][svec_ECRad.T[3] > 0], "-", label=r"$R_\mathrm{ECRad}(z)$")
    plt.plot(svec_IDA.T[1][svec_IDA.T[3] > 0], svec_IDA.T[2][svec_IDA.T[3] > 0], "--", label=r"$R_\mathrm{IDA}(z)$")
    plt.gca().set_xlabel(r"$R$ [m]")
    plt.gca().set_ylabel(r"$z$ [m]")
    plt.legend()
    plt.title(r"Rz IDA vs ECRad")
    plt.show()
    
def compare_quant_on_LOS(working_dir, ida_working_dir, chno):
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    svec_IDA = read_svec_dict_from_file(ida_working_dir, chno)[0]
    scale_dict = {}
    scale_dict["ne"] = 1.e19
    scale_dict["Te"] = 1.e3
    scale_dict["R"] = 1.0
    scale_dict["z"] = 1.e-2
    scale_dict["freq_2X"] = 100.e9
    scale_dict["rhop"] = 1.e0
    for quant in ["R", "z"]:
        if(quant not in ["R", "z"]):
            mask_ECRad = svec_ECRad["rhop"] >= 0
            mask_IDA = svec_IDA["rhop"] >= 0
        else:
            mask_ECRad = np.zeros(len(svec_ECRad["s"]), dtype=np.bool)
            mask_ECRad[:] = True
            mask_IDA = np.zeros(len(svec_IDA["s"]), dtype=np.bool)
            mask_IDA[:] = True
        ECRadx = svec_ECRad["R"][mask_ECRad]
        ECRadquant = svec_ECRad[quant][mask_ECRad]
        IDAx = svec_IDA["R"][mask_IDA]
        IDAquant =  svec_IDA[quant][mask_IDA]
        print("ECRad: {0:1.8e}\t IDA: {1:1.8e}".format(ECRadquant[-1], IDAquant[-1]))
        plt.plot(ECRadx, ECRadquant/scale_dict[quant], "-", label=quant.replace("_"," ") + " ECRad")
        plt.plot(IDAx, IDAquant/scale_dict[quant], "--", label=quant.replace("_"," ") + " IDA")
#     plt.gca().set_xlabel(r"$s$ [m]")
#     plt.gca().set_ylabel(r"$\rho_\mathrm{pol}$")
    plt.legend()
#     plt.title(r"$\rho_\mathrm{pol}$ IDA vs ECRad")
    plt.show()
    
def compare_quant_on_LOS_rel(working_dir, ida_working_dir, chno):
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    svec_IDA = read_svec_dict_from_file(ida_working_dir, chno)[0]
    scale_dict = {}
    x_quant = "R"
    scale_dict["ne"] = 1.e19
    scale_dict["Te"] = 1.e3
    scale_dict["R"] = 1.0
    scale_dict["z"] = 1.e-2
    scale_dict["freq_2X"] = 100.e9
    scale_dict["rhop"] = 1.e0
    scale_dict["theta"] = 2.0 * np.pi
    for quant, deriv in zip(["freq_2X","ne", "z"], [0,0,0,0]):
        if(False):#quant not in ["R", "z"]
            mask_ECRad = svec_ECRad["rhop"] >= 0
            mask_IDA = svec_IDA["rhop"] >= 0
        else:
            mask_ECRad = np.zeros(len(svec_ECRad["s"]), dtype=np.bool)
            mask_ECRad[:] = True
            mask_IDA = np.zeros(len(svec_IDA["s"]), dtype=np.bool)
            mask_IDA[:] = True
        ECRadx = svec_ECRad[x_quant][mask_ECRad]
        ECRadquant = svec_ECRad[quant][mask_ECRad]
        ECRad_spl = InterpolatedUnivariateSpline(ECRadx, ECRadquant)
        IDAx = svec_IDA[x_quant][mask_IDA]
        IDAquant =  svec_IDA[quant][mask_IDA]
        IDA_spl = InterpolatedUnivariateSpline(IDAx, IDAquant)
        s = np.linspace(max(np.min(ECRadx),np.min(IDAx)), min(np.max(ECRadx),np.max(IDAx)), 2000)
        plt.plot(s, (IDA_spl(s, nu=deriv)-ECRad_spl(s, nu=deriv)) / scale_dict[quant], label=quant.replace("_"," "))
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.legend()
    plt.show() 

def compare_res_pos(working_dir, ida_working_dir, channel_num_as_x=False):
    res_ECRad = np.loadtxt(os.path.join(working_dir, "sres.dat"))
    res_IDA = np.loadtxt(os.path.join(ida_working_dir, "sres.dat"))
    if(channel_num_as_x):
        plt.plot(np.array(range(len(res_ECRad.T[0])))  + 1, 1.0 - res_ECRad.T[3]/res_IDA.T[3], "s")#, label="ECRad"
#         plt.plot(np.array(range(len(res_IDA.T[0]))) + 1, res_IDA.T[0], "+", label="IDA")
    else:
        plt.plot(res_ECRad.T[3], res_ECRad.T[3]-res_IDA.T[3], "s", label="ECRad")
#         plt.plot(res_IDA.T[0], res_IDA.T[-1], "+", label="IDA")
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol,res}$")
    plt.gca().set_ylabel(r"$\Delta\rho_\mathrm{pol,res}$")
#     plt.legend()
    plt.title(r"$\rho_\mathrm{pol,res}$ [ECRad] $-$ $\rho_\mathrm{pol,res}$ [IDA]")
    plt.show()
    
def compare_ds(working_dir, ida_working_dir, chno):
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    svec_IDA = read_svec_dict_from_file(ida_working_dir, chno)[0]
    res_ECRad = np.loadtxt(os.path.join(working_dir, "sres.dat"))
    res_IDA = np.loadtxt(os.path.join(ida_working_dir, "sres.dat"))
    ds_ECRad = svec_ECRad["s"][1:] - svec_ECRad["s"][:-1]
    ds_IDA = svec_IDA["s"][1:] - svec_IDA["s"][:-1]
    plt.plot(svec_ECRad["s"][:-1], ds_ECRad, "-", label="ECRad")
    plt.vlines(res_ECRad.T[0][chno-1], 0, np.max(ds_ECRad))
    plt.plot(svec_IDA["s"][:-1], ds_IDA, "--", label="IDA")
    plt.vlines(res_IDA.T[0][chno-1], 0, np.max(ds_IDA), linestyles="--", color="red")
#     plt.gca().set_xlabel(r"$s$ [m]")
#     plt.gca().set_ylabel(r"$\rho_\mathrm{pol}$")
    plt.legend()
#     plt.title(r"$\rho_\mathrm{pol}$ IDA vs ECRad")
    plt.show()   
    
def compare_ds_rel(working_dir, ida_working_dir, chno):
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    svec_IDA = read_svec_dict_from_file(ida_working_dir, chno)[0]
    res_IDA = np.loadtxt(os.path.join(ida_working_dir, "sres.dat"))
    ds_ECRad = svec_ECRad["s"][1:] - svec_ECRad["s"][:-1]
    ds_spl_ECRad=InterpolatedUnivariateSpline(svec_ECRad["s"][:-1], ds_ECRad)
    ds_IDA = svec_IDA["s"][1:] - svec_IDA["s"][:-1]
    ds_spl_IDA=InterpolatedUnivariateSpline(svec_IDA["s"][:-1], ds_IDA)
    s = np.linspace(max([np.min(svec_IDA["s"][:-1]), np.min(svec_ECRad["s"][:-1])]), min([np.max(svec_IDA["s"][:-1]), np.max(svec_ECRad["s"][:-1])]), 500)
    plt.plot(s, ds_spl_IDA(s) - ds_spl_ECRad(s), "--")
    plt.vlines(res_IDA.T[0][chno-1], 0, np.max(ds_IDA), linestyles="--", color="red")
#     plt.gca().set_xlabel(r"$s$ [m]")
#     plt.gca().set_ylabel(r"$\rho_\mathrm{pol}$")
#     plt.title(r"$\rho_\mathrm{pol}$ IDA vs ECRad")
    plt.show()    
  
def compare_rhop(working_dir, ida_working_dir, chno):
    ECRad_topdata = read_topfile(working_dir)
    IDA_topdata = read_topfile(ida_working_dir)
    ECRad_rhop = np.sqrt(ECRad_topdata["Psi"])
    rhop_spl = RectBivariateSpline(ECRad_topdata["R"], ECRad_topdata["z"], ECRad_rhop)
    IDA_rhop = np.sqrt(IDA_topdata["Psi"])
    levels_rhop = np.linspace(0.0, 1.2, 13)
    plt_range = np.array([np.min(ECRad_rhop - IDA_rhop), np.max(ECRad_rhop - IDA_rhop)])
    quant_av = np.sqrt(np.mean(np.abs(ECRad_rhop).flatten()))
    print(plt_range)
    print(plt_range/quant_av)
    levels = np.linspace(plt_range[0],plt_range[1], 30)
    plt.contourf(IDA_topdata["R"], IDA_topdata["z"], IDA_rhop.T - ECRad_rhop.T, levels=levels, cmap=plt.cm.get_cmap("plasma"))
    plt.contour(ECRad_topdata["R"], ECRad_topdata["z"], ECRad_rhop.T, levels=levels_rhop, colors="k", linestyles="-")
    plt.figure()
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    svec_IDA = read_svec_dict_from_file(ida_working_dir, chno)[0]
    quant = "rhop"
    if(quant not in ["R", "z"]):
        mask_ECRad = svec_ECRad["rhop"] >= 0
        mask_IDA = svec_IDA["rhop"] >= 0
    else:
        mask_ECRad = np.zeros(len(svec_ECRad["s"]), dtype=np.bool)
        mask_ECRad[:] = True
        mask_IDA = np.zeros(len(svec_IDA["s"]), dtype=np.bool)
        mask_IDA[:] = True
    plt.plot(svec_ECRad["s"][mask_ECRad], svec_ECRad[quant][mask_ECRad] - \
             rhop_spl( svec_ECRad["R"][mask_ECRad],svec_ECRad["z"][mask_ECRad], grid=False), "-k", label="ECRad")
    plt.plot(svec_IDA["s"][mask_IDA], svec_IDA[quant][mask_IDA] - \
             rhop_spl( svec_IDA["R"][mask_IDA],svec_IDA["z"][mask_IDA], grid=False), "--r", label="IDA")
    plt.legend()
    plt.show()
    
def compare_LOS_dev(working_dir, ida_working_dir, chno):
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    svec_IDA = read_svec_dict_from_file(ida_working_dir, chno)[0]
    mask_ECRad = svec_ECRad["rhop"] >= 0
    mask_IDA = svec_IDA["rhop"] >= 0
    for quant, deriv in zip(["ne", "Te"], [0,0]):
        ECRadx = svec_ECRad["s"][mask_ECRad]
        ECRadquant = svec_ECRad[quant][mask_ECRad]
        ECRadquantspl = InterpolatedUnivariateSpline(ECRadx, ECRadquant)
        IDAx = svec_IDA["s"][mask_IDA]
        IDAquant =  svec_IDA[quant][mask_IDA]
        IDAquantspl = InterpolatedUnivariateSpline(IDAx, IDAquant)
        s = np.linspace(max([np.min(IDAx), np.min(ECRadx)]), min([np.max(IDAx), np.max(ECRadx)]), 500)
        plt.plot(s, (ECRadquantspl(s,nu=deriv) - IDAquantspl(s,nu=deriv)) / ECRadquantspl(s,nu=deriv), label=quant)
#     plt.gca().set_xlabel(r"$s$ [m]")
#     plt.gca().set_ylabel(r"$\rho_\mathrm{pol}$")
    plt.legend()
#     plt.title(r"$\rho_\mathrm{pol}$ IDA vs ECRad")
    plt.show()

def compare_topfiles(working_dir, ida_working_dir, ref_path, shot=None, time=None, EQ_exp=None, EQ_diag=None, EQ_ed=None, btf_cor=None):
    ECRad_topdata = read_topfile(working_dir)
    IDA_topdata = read_topfile(ida_working_dir)
    if(shot is not None):
        raise ValueError("This routine needs to be fixed before usage")
        make_topfile_from_ext_data(ref_path, shot, time, EQ_exp, EQ_diag, EQ_ed, copy_Te_ne=False)
        IDA_topdata = read_topfile(ref_path)
    quant = "Bt"
    level_dict = {}
    level_dict["Psi"] = np.linspace(0.0, 1.2, 13)
    level_dict["Bt"] = np.linspace(-3.0, -1.2, 13)
    level_dict["Br"] = np.linspace(-200.e-3, 200.e-3, 13)
    level_dict["Bz"] = np.linspace(-200.e-3, 200.e-3, 13)
#     plt.contour(ECRad_topdata["R"], ECRad_topdata["z"], ECRad_topdata["Psi"].T, levels=level_dict["Psi"], colors="k", linestyles="-")
#     plt.contour(IDA_topdata["R"], IDA_topdata["z"], IDA_topdata["Psi"].T, levels=level_dict["Psi"], colors="r", linestyles="--")
    plt_range = np.array([np.min(IDA_topdata[quant].T - ECRad_topdata[quant].T), np.max(IDA_topdata[quant].T - ECRad_topdata[quant].T)])
    quant_av = np.mean(np.abs(ECRad_topdata[quant]).flatten())
    print(plt_range)
    print(plt_range/quant_av)
    levels = np.linspace(plt_range[0],plt_range[1], 30)
    plt.contourf(IDA_topdata["R"], IDA_topdata["z"], IDA_topdata[quant].T - ECRad_topdata[quant].T, levels=levels, cmap=plt.cm.get_cmap("plasma"))
    plt.contour(ECRad_topdata["R"], ECRad_topdata["z"],ECRad_topdata[quant].T, levels=level_dict[quant], colors="k", linestyles="-")
    plt.show()
    
def compare_topfiles_cut(working_dir, ida_working_dir, ref_path, shot=None, time=None, EQ_exp=None, EQ_diag=None, EQ_ed=None, btf_cor=None):
    ECRad_topdata = read_topfile(working_dir)
    IDA_topdata = read_topfile(ida_working_dir)
    if(shot is not None):
        raise ValueError("This routine needs to be fixed before usage")
        make_topfile_from_ext_data(ref_path, shot, time, EQ_exp, EQ_diag, EQ_ed, copy_Te_ne=False)
        ECRad_topdata = read_topfile(ref_path)
    quant = "Psi"
    level_dict = {}
    level_dict["Psi"] = np.linspace(0.0, 1.2, 13)
    level_dict["Bt"] = np.linspace(-3.0, -1.2, 13)
    level_dict["Br"] = np.linspace(-200.e-3, 200.e-3, 13)
    level_dict["Bz"] = np.linspace(-200.e-3, 200.e-3, 13)
#     plt.contour(ECRad_topdata["R"], ECRad_topdata["z"], ECRad_topdata["Psi"].T, levels=level_dict["Psi"], colors="k", linestyles="-")
#     plt.contour(IDA_topdata["R"], IDA_topdata["z"], IDA_topdata["Psi"].T, levels=level_dict["Psi"], colors="r", linestyles="--")
    plt.plot(ECRad_topdata["R"], ECRad_topdata[quant][:,len(ECRad_topdata["z"]) / 2], "-k")
    plt.plot(IDA_topdata["R"], IDA_topdata[quant][:,len(IDA_topdata["z"]) / 2], "--r")
    plt.show()

def compare_topfiles_B_tot(working_dir, ida_working_dir):
    ECRad_topdata = read_topfile(working_dir,)
    IDA_topdata = read_topfile(ida_working_dir)
    plt.figure()
    level_dict = {}
    level_dict["Btot"] = np.linspace(-3.0, -1.2, 13)
    Btot_ECRad = np.zeros(ECRad_topdata["Bt"].shape)
    Btot_IDA = np.zeros(ECRad_topdata["Bt"].shape)
    for quant  in ["Br", "Bz"]: # "Bt", 
        Btot_ECRad += ECRad_topdata[quant]**2
        Btot_IDA += IDA_topdata[quant]**2
    Btot_ECRad = np.sqrt(Btot_ECRad)
    Btot_IDA = np.sqrt(Btot_IDA)
#     plt.contour(ECRad_topdata["R"], ECRad_topdata["z"], ECRad_topdata["Psi"].T, levels=rhop_levels, colors="k", linestyles="-")
#     plt.contour(IDA_topdata["R"], IDA_topdata["z"], IDA_topdata["Psi"].T, levels=rhop_levels, colors="r", linestyles="--")
    plt_range = np.array([np.min(Btot_IDA-Btot_ECRad), np.max(Btot_IDA-Btot_ECRad)])
    Btot_av = np.mean(np.abs(Btot_ECRad).flatten())
    print(plt_range) 
    print(plt_range / Btot_av)
    levels = np.linspace(plt_range[0],plt_range[1], 30)
    plt.contourf(IDA_topdata["R"], IDA_topdata["z"], Btot_IDA.T-Btot_ECRad.T, levels=levels, cmap=plt.cm.get_cmap("plasma"))
#     plt.imshow(1.0 - Btot_IDA/Btot_ECRad, interpolation=None, cmap=plt.cm.get_cmap("plasma"))
#     plt.contour(ECRad_topdata["R"], ECRad_topdata["z"],Btot_ECRad.T, levels=level_dict[quant], colors="k", linestyles="-")
    plt.show()

def inspect_svec(working_dir, chno):
    svec_ECRad = read_svec_dict_from_file(working_dir, chno)[0]
    mask_ECRad = svec_ECRad["rhop"] >= 0
    scale_dict= {}
    scale_dict["ne"] = 1.e19
    scale_dict["Te"] = 1.e3
    scale_dict["freq_2X"] = 1.e9
    scale_dict["theta"] = np.deg2rad(1)
    scale_dict["R"] = 1.0
    scale_dict["z"] = 1.0
    scale_dict["ds"] = 1.e-3
    svec_ECRad["ds"] = svec_ECRad["s"][1:] - svec_ECRad["s"][:-1]
    ECRadx = svec_ECRad["s"][mask_ECRad][:-1]
    for quant in ["ds"]:
        ECRadquant = svec_ECRad[quant][mask_ECRad[:-1]]
        plt.plot(ECRadx, ECRadquant / scale_dict[quant], label=quant.replace("_", " "))
    plt.legend()
    plt.show()

def calculate_coupling(path, channel):
    svec_X = read_svec_dict_from_file(os.path.join(path, "ecfm_data"), channel)[0]
    svec_O, freq = read_svec_dict_from_file(os.path.join(path, "ecfm_data"), channel, "O")
    N_X_spl = InterpolatedUnivariateSpline(svec_X["s"][svec_X["rhop"] != -1], svec_X["N_abs"][svec_X["rhop"] != -1])
    N_O_spl = InterpolatedUnivariateSpline(svec_O["s"][svec_O["rhop"] != -1], svec_O["N_abs"][svec_O["rhop"] != -1])
    s = np.linspace(np.max([np.min(svec_X["s"][svec_X["rhop"] != -1]), np.min(svec_O["s"][svec_O["rhop"] != -1])]), \
                    np.min([np.max(svec_X["s"][svec_X["rhop"] != -1]), np.max(svec_O["s"][svec_O["rhop"] != -1])]), 1000)
    N_diff = np.abs(N_X_spl(s) - N_O_spl(s))
    Psi = simps(N_diff, s) * freq * 2.0 * np.pi / cnst.c
    print("Psi", Psi)
    plt.plot(s, N_diff * 1.e6)
    plt.show()

def plot_resonance_line(freq_2X, freq, theta):
    abs_a = EmAbsAlb()
    svec = SVec(0.2, 8.e3, 1.5e19, freq_2X, np.pi / 180.0 * theta)
    u_par, u_perp = abs_a.abs_Albajar_resonance_line(svec, freq * 2.e0 * np.pi, 1, 2)
    plt.plot(u_par, u_perp)

def R_wall_behavior(folder, R_init):
    Trad_data = np.loadtxt(os.path.join(folder, "ecfm_data", "O_TRadM_therm.dat"))
    Trad_data = Trad_data[Trad_data.T[2] > 0.0]
    Trad_data = Trad_data[Trad_data.T[2] < 2.0]
    for R in np.concatenate([[0.0], np.linspace(0.9, 1.0, 5)]):
        plt.plot(Trad_data.T[0], Trad_data.T[1] * (1.0 - R_init * np.exp(-Trad_data.T[2])) / (1.0 - R * np.exp(-Trad_data.T[2])), \
                 "+", label=r"$R_\mathrm{wall}$ = " + r"{0:1.3e}".format(R))
    plt.legend()
    R = 1.0
    plt.figure()
    plt.plot(Trad_data.T[0], Trad_data.T[1] * (1.0 - R_init * np.exp(-Trad_data.T[2])) - (1.0 - R * np.exp(-Trad_data.T[2])) / Trad_data.T[1], \
             "+", label=r"$T_\mathrm{rad}[R_\mathrm{wall} = " + r"{0:1.2f}".format(R) + r"] - " + r"T_\mathrm{rad}[R_\mathrm{wall} = " + r"{0:1.2f}".format(R_init) + r"]$")
    plt.legend()
    plt.show()


def double_check_alpha_integration(folder, ch):
    ichdata = np.loadtxt(os.path.join(folder, "ecfm_data", "IchTB", "Irhopch" + "{0:03d}".format(ch) + ".dat"))
    Tdata = np.loadtxt(os.path.join(folder, "ecfm_data", "IchTB", "Trhopch" + "{0:03d}".format(ch) + ".dat"))
    abs_spl = InterpolatedUnivariateSpline(ichdata.T[0], ichdata.T[5])
    T = np.zeros(len(ichdata.T[0]))
    for i in range(len(ichdata.T[0])):
        T[i] = abs_spl.integral(ichdata.T[0][i], ichdata.T[-1])
    plt.plot(Tdata.T[0], Tdata.T[1])
    plt.plot(ichdata.T[0], np.exp(-T), "--")
    plt.show()

def test_get_diag_data(shot, times, diag):
    if(diag == "CTA"):
        N = 50
        calib = np.zeros(N)
        calib[:] = 1.0
        std_dev_calib = np.zeros(N)
        res = []
        for i in range(len(times)):
            res.append(np.linspace(1, N, N))
        data = get_data_calib(diag_id=diag, shot=shot, time=times, exp="AUGD", ed=0, calib=calib, std_dev_calib=std_dev_calib, \
                       ext_resonances=res, name="", t_smooth=1.e-3)[1]
    elif(diag == "ECN"):
        N = 160
        calib = np.zeros(N)
        calib[:] = 1.0
        std_dev_calib = np.zeros(N)
        res = []
        for i in range(len(times)):
            res.append(np.linspace(1, N, N))
        data = get_data_calib(diag_id=diag, shot=shot, time=times, exp="AUGD", ed=0, calib=calib, std_dev_calib=std_dev_calib, \
                       ext_resonances=res, name="", t_smooth=1.e-3)[1]
    elif(diag == "ECO"):
        N = 120
        calib = np.zeros(N)
        calib[:] = 1.0
        std_dev_calib = np.zeros(N)
        res = []
        for i in range(len(times)):
            res.append(np.linspace(1, N, N))
        data = get_data_calib(diag_id=diag, shot=shot, time=times, exp="AUGD", ed=0, calib=calib, std_dev_calib=std_dev_calib, \
                       ext_resonances=res, name="", t_smooth=1.e-3)[1]
    print(data[1].shape)
    plt.plot(data[0][0], data[1][0])
    plt.show()

def debug_EQ(path, working_dir):
    time = 1.0
    at_least_1d_keys = ["t", "R", "z", "Psi_sep", "Psi_ax"]
    at_least_2d_keys = ["rhop", "Te", "ne"]
    at_least_3d_keys = ["Psi", "Br", "Bt", "Bz"]
    variable_names = at_least_1d_keys + at_least_2d_keys + at_least_3d_keys + ["shotnum"]
    # print(variable_names)
    try:
        mdict = loadmat(path, chars_as_strings=True, squeeze_me=True, variable_names=variable_names)
    except IOError:
        print("Error: " + path + " does not exist")
        raise IOError
    print(mdict)
    increase_diag_dim = False
    increase_time_dim = False
    if(np.isscalar(mdict["t"])):
        times = np.array([mdict["t"]])
        increase_time_dim = True
    else:
        times = mdict["t"]
    for key in mdict:
        if(not key.startswith("_")):  # throw out the .mat specific information
            try:
                if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                    mdict[key] = np.array([mdict[key]])
                elif(key in at_least_2d_keys):
                    if(increase_time_dim):
                        mdict[key] = np.array([mdict[key]])
                    elif(increase_time_dim):
                        for i in range(len(mdict[key])):
                            mdict[key][i] = np.array([mdict[key][i]])
                    if(increase_diag_dim):
                        mdict[key] = np.array([mdict[key]])
                elif(key in at_least_3d_keys):
                    if(increase_time_dim):
                        mdict[key] = np.array([mdict[key]])
            except Exception as e:
                print(key)
                print(e)
    for key in at_least_3d_keys:
        mdict[key] = np.swapaxes(mdict[key], 2, 0)
        mdict[key] = np.swapaxes(mdict[key], 1, 2)
    for key in at_least_2d_keys:
        mdict[key] = np.swapaxes(mdict[key], 0, 1)
    EQ_obj = EQDataExt(0, external_folder=os.path.dirname(path), Ext_data=True)
    shot = int(mdict["shotnum"])
    EQ_obj.load_slices_from_mat(times, mdict)
    itime = np.argmin(np.abs(times - time))
    print(np.min(EQ_obj.slices[itime].rhop), np.max(EQ_obj.slices[itime].rhop))
    plt.contour(EQ_obj.slices[itime].R, EQ_obj.slices[itime].z, EQ_obj.slices[itime].rhop, levels=[0.1, 0.4, 1.0, 1.2])
    plt.show()
    make_topfile_from_ext_data(working_dir, shot, times[itime], EQ_obj.slices[itime], mdict["rhop"][itime], \
                               mdict["Te"][itime] * 1.e3, mdict["ne"][itime], grid=False)


def debug_calib(resultfile):
    result = ECRadResults()
    result.from_mat_file(resultfile)
    CTA = ECRH_diag("CTA", "AUGD", "CTA", 0, 7, 1.0, False, t_smooth=1.e-3)
    res = result.resonance["rhop_cold"][0][result.Scenario.ray_launch[0]["diag_name"] == CTA.name]
    calib = np.zeros(len(res))
    calib[:] = 1.0
    std_dev_calib = np.zeros(len(res))
    sys_dev_calib = np.zeros(len(res))
    err, data = get_data_calib(CTA, 35662, 1.5,calib=calib, std_dev_calib=std_dev_calib, sys_dev_calib=sys_dev_calib, ext_resonances=res)
    plt.plot(res, err[0] / data[1], "+")
    # Gets the data from al )
#     plt.plot(result.resonance["rhop_cold"][0][result.Scenario.ray_launch[0]["diag_name"] == diag], np.abs(result.sys_dev["CTA"]/result.calib["CTA"]), "+")
#     plt.plot(result.resonance["rhop_cold"][0][result.Scenario.ray_launch[0]["diag_name"] == diag], np.abs(result.rel_dev["CTA"]), "^")
#     plt.errorbar(result.resonance["rhop_cold"][0][result.Scenario.ray_launch[0]["diag_name"] == diag], result.calib["CTA"], result.sys_dev["CTA"])
    plt.show()


def debug_ray(results_file, itime, ich, ir):
    result = ECRadResults()
    result.from_mat_file(results_file)
    plt.plot(result.ray["rhopX"][itime][ich])
#     plt.plot(result.ray["rhopX"][itime][ich], result.ray["TeX"][itime][ich])
#     plt.gca().twinx()
#     plt.plot(result.ray["rhopX"][itime][ich], result.ray["neX"][itime][ich], "--r")
    plt.show()
    
def debug_fitpack(x_file=None, y_file=None, xy_file=None, log=False):
    if(x_file is not None):
        x = np.loadtxt(x_file, skiprows=2, delimiter=",").T[1]
        y = np.loadtxt(y_file, skiprows=2, delimiter=",").T[1]
    else:
        x,y = np.loadtxt(xy_file, skiprows=1, unpack=True)
    if(log):
        spl = InterpolatedUnivariateSpline(x,np.log(y))
    else:
        spl = InterpolatedUnivariateSpline(x,y)
    x_int = np.linspace(np.min(x), np.max(x), 10000)
    plt.plot(x,y,"+")
    if(log):
        plt.plot(x_int,np.exp(spl(x_int)), "--")
    else:
        plt.plot(x_int,spl(x_int), "--")
    plt.show()

if(__name__ == "__main__"):
    pass