'''
Created on Oct 11, 2016

@author: sdenk
'''
import os
import numpy as np
import sys
vessel_file = '/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Pylib/ASDEX_Upgrade_vessel.txt'
sys.path.append("../ECRad_Pylib")
from subprocess import call
from scipy.io import savemat
from GlobalSettings import globalsettings
if(globalsettings.AUG):
    from shotfile_handling_AUG import load_IDA_data, get_Vloop, get_RELAX_target_current, get_total_current, make_ext_data_for_testing_from_data
    from equilibrium_utils_AUG import EQData
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
import scipy.constants as cnst
from plotting_configuration import *
import shutil
import wx
from wxEvents import *
from shutil import copyfile

tb_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/"

tb_path_itm = "/marconi_work/eufus_gw/work/g2sdenk/torbeam/lib-OUT/"
# "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam/"
#


def read_topfile(working_dir):
    topfile = open((os.path.join(working_dir, "topfile")), "r")
    order = ["dummy", "dim", "flxsep", "R", "z", "Br", "Bt", "Bz", "Psi"]
    data_dict = {}
    for key in order:
        data_dict[key] = []
    iorder = 0
    for line in topfile.readlines():
        parsed_line = np.fromstring(line, sep=" ")
        if(len(parsed_line) == 0):
            if(iorder > 0):
                data_dict[order[iorder]] = np.concatenate(data_dict[order[iorder]]).flatten()
            iorder += 1
        else:
            data_dict[order[iorder]].append(parsed_line)
    data_dict[order[iorder]] = np.array(data_dict[order[iorder]]).flatten()
    m, n = data_dict["dim"]
    m = int(m)
    n = int(n)
    data_dict["Psi_sep"] = data_dict["flxsep"][-1]
    data_dict["R"] = data_dict["R"]
    data_dict["z"] = data_dict["z"]
    data_dict["Br"] = data_dict["Br"].reshape((n, m)).T
    data_dict["Bt"] = data_dict["Bt"].reshape((n, m)).T
    data_dict["Bz"] = data_dict["Bz"].reshape((n, m)).T
    data_dict["Psi"] = data_dict["Psi"].reshape((n, m)).T
    return data_dict

def make_topfile(working_dir, shot, time, eq_exp, eq_diag, eq_ed, bt_vac_correction=1.005, copy_Te_ne=True):
    # Note this routine uses MBI-BTFABB for a correction of the toroidal magnetic field
    # Furthermore, an empirical vacuum BTF correction factor of bt_vac_correction is applied
    # This creates a topfile that is consistent with the magnetic field used in OERT
    print("Creating topfile for #{0:d} t = {1:1.2f}".format(shot, time))
    columns = 3  # number of coloumns
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed, bt_vac_correction=bt_vac_correction)
    EQ_t = EQ_obj.GetSlice(time)
    print("Magnetic axis position: ", "{0:1.3f}".format(EQ_t.R_ax))
    topfile = open(os.path.join(working_dir, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5n}: {1:1.4f}\n'.format(shot, time))
    topfile.write('   {0: 8n} {1: 8n}\n'.format(len(EQ_t.R), len(EQ_t.z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.8E}  {1: 1.8E}  {2: 1.8E}'.format(EQ_t.R[0], EQ_t.R[-1], \
                                                              1.0))  # 1.0000
    # Normalize PSI
    topfile.write('\n')
    topfile.write('Radial grid coordinates\n')
    cnt = 0
    for i in range(len(EQ_t.R)):
        topfile.write("  {0: 1.8E}".format(EQ_t.R[i]))
        if(cnt == columns - 1):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for i in range(len(EQ_t.z)):
        topfile.write("  {0: 1.8E}".format(EQ_t.z[i]))
        if(cnt == columns - 1):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    # ivR = np.argmin(np.abs(pfm_dict["Ri"] - rv))
    # jvz = np.argmin(np.abs(pfm_dict["zj"] - vz))
    # plt.plot(pfm_dict["Ri"],B_t[0], "^", label = "EQH B")
    # print("BTFABB correction",Btf0, Btf0_eq )
    # print("R,z",pfm_dict["Ri"][ivR],pfm_dict["zj"][jvz])
    B_r = EQ_t.Br.T  # in topfile R is the small index (i.e. second index in C) and z the large index (i.e. first index in C)
    B_t = EQ_t.Bt.T  # in topfile z comes first regardless of the arrangement
    B_z = EQ_t.Bz.T  # in topfile z comes first regardless of the arrangement
    Psi = EQ_t.rhop.T**2  # in topfile z comes first regardless of the arrangement
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(B_r)):
        for j in range(len(B_r[i])):
            topfile.write("  {0: 1.8E}".format(B_r[i][j]))
            if(cnt == columns - 1):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(B_t)):
        for j in range(len(B_t[i])):
            topfile.write("  {0: 1.8E}".format(B_t[i][j]))
            if(cnt == columns - 1):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(B_z)):
        for j in range(len(B_z[i])):
            topfile.write("  {0: 1.8E}".format(B_z[i][j]))
            if(cnt == columns - 1):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(Psi)):
        for j in range(len(Psi[i])):
            topfile.write("  {0: 1.8E}".format(Psi[i][j]))
            if(cnt == columns - 1):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.flush()
    topfile.close()
    print("topfile successfully written to", os.path.join(working_dir, "topfile"))
    if(copy_Te_ne):
        print("Copying Te and ne")
        copyfile(os.path.join(working_dir, "Te_file.dat"), \
                 os.path.join(working_dir, "Te.dat"))
        copyfile(os.path.join(working_dir, "ne_file.dat"), \
                 os.path.join(working_dir, "ne.dat"))
    return 0

def make_topfile_no_data_load(working_dir, shot, time, R, z, Psi, Br, Bt, Bz, Psi_ax, Psi_sep, ITM=False):
    # Note this routine uses MBI-BTFABB for a correction of the toroidal magentic field
    # Furthermore, an empirical vacuum BTF correction factor of bt_vac_correction is applied
    # This creates a topfile that is consistent with the magnetic field used in OERT
    print("Creating topfile for #{0:n} t = {1:1.2f}".format(shot, time))
    if(ITM):
        columns = len(z) - 1
    else:
        columns = 8  # number of coloumns
        columns -= 1
    topfile = open(os.path.join(working_dir, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5n}: {1:1.4f}\n'.format(shot, time))
    topfile.write('   {0: 8n} {1: 8n}\n'.format(len(R), len(z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.8E}  {1: 1.8E}  {2: 1.8E}'.format(R[0], R[-1], 1.0))
    # Normalize PSI
    topfile.write('\n')
    topfile.write('Radial grid coordinates\n')
    cnt = 0
    for i in range(len(R)):
        topfile.write("  {0: 1.8E}".format(R[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    topfile.write('\n')
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for i in range(len(z)):
        topfile.write("  {0: 1.8E}".format(z[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    # ivR = np.argmin(np.abs(pfm_dict["Ri"] - rv))
    # jvz = np.argmin(np.abs(pfm_dict["zj"] - vz))
    # plt.plot(pfm_dict["Ri"],B_t[0], "^", label = "EQH B")
    # print("BTFABB correction",Btf0, Btf0_eq )
    # print("R,z",pfm_dict["Ri"][ivR],pfm_dict["zj"][jvz])
    B_r = Br.T  # in topfile R is the small index (i.e. second index in C) and z the large index (i.e. first index in C)
    B_t = Bt.T  # in topfile z comes first regardless of the arrangement
    B_z = Bz.T  # in topfile z comes first regardless of the arrangement
    Psi = Psi.T  # in topfile z comes first regardless of the arrangement
    Psi = (Psi - Psi_ax) / (Psi_sep - Psi_ax)
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(B_r)):
        for j in range(len(B_r[i])):
            topfile.write("  {0: 1.8E}".format(B_r[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt is not columns and cnt is not 0):
        topfile.write('\n')
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(B_t)):
        for j in range(len(B_t[i])):
            topfile.write("  {0: 1.8E}".format(B_t[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt is not columns and cnt is not 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(B_z)):
        for j in range(len(B_z[i])):
            topfile.write("  {0: 1.8E}".format(B_z[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt is not columns and cnt is not 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(Psi)):
        for j in range(len(Psi[i])):
            topfile.write("  {0: 1.8E}".format(Psi[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.flush()
    topfile.close()
    print("topfile successfully written to", os.path.join(working_dir, "topfile"))
    return 0

def make_topfile_from_ext_data(working_dir, shot, time, EQ, rhop, Te, ne, grid=False):
    columns = 5  # number of coloumns
    columns -= 1
    print("Magnetic axis position: ", "{0:1.3f}".format(EQ.special[0]))
    topfile = open(os.path.join(working_dir, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5n}: {1:1.4f}\n'.format(shot, time))
    topfile.write('   {0: 8n} {1: 8n}\n'.format(len(EQ.R), len(EQ.z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.8E}  {1: 1.8E}  {2: 1.8E}'.format(EQ.R[0], EQ.R[-1], \
        EQ.special[1]))
    topfile.write('\n')
    topfile.write('Radial grid coordinates\n')
    cnt = 0
    for i in range(len(EQ.R)):
        topfile.write("  {0: 1.8E}".format(EQ.R[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt is not 0):
        topfile.write('\n')
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for j in range(len(EQ.z)):
        topfile.write("  {0: 1.8E}".format(EQ.z[j]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt is not 0):
        topfile.write('\n')
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(EQ.Br)):
        for j in range(len(EQ.Br[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Br[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt is not 0):
        topfile.write('\n')
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(EQ.Bt)):
        for j in range(len(EQ.Bt[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Bt[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt is not 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(EQ.Bz)):
        for j in range(len(EQ.Bz[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Bz[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt is not 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(EQ.Psi)):
        for j in range(len(EQ.Psi[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Psi[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.flush()
    topfile.close()
    print("topfile successfully written to", os.path.join(working_dir, "topfile"))
    if(not grid):
        print("Copying Te and ne profile")
        Te_file = open(os.path.join(working_dir, "Te_file.dat"), "w")
        Te_tb_file = open(os.path.join(working_dir, "Te.dat"), "w")
        lines = 150
        Te_file.write("{0: 7d}".format(lines) + "\n")
        Te_tb_file.write("{0: 7d}".format(lines) + "\n")
        Te_spline = InterpolatedUnivariateSpline(rhop, Te, k=1)
        rhop_short = np.linspace(np.min(rhop), np.max(rhop), lines)
        for i in range(len(rhop_short)):
            try:
                Te_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], Te_spline(rhop_short[i]).item()) + "\n")
                Te_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], Te_spline(rhop_short[i]).item() / 1.e03) + "\n")
            except ValueError:
                print(rhop_short[i], Te_spline(rhop_short[i]))
                raise(ValueError)
        Te_file.flush()
        Te_file.close()
        Te_tb_file.flush()
        Te_tb_file.close()
        ne_file = open(os.path.join(working_dir, "ne_file.dat"), "w")
        ne_tb_file = open(os.path.join(working_dir, "ne.dat"), "w")
        lines = 150
        ne_file.write("{0: 7n}".format(lines) + "\n")
        ne_tb_file.write("{0: 7n}".format(lines) + "\n")
        ne_spline = InterpolatedUnivariateSpline(rhop, ne, k=1)
        rhop_short = np.linspace(np.min(rhop), np.max(rhop), lines)
        for i in range(len(rhop_short)):
            ne_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], ne_spline(rhop_short[i]).item()) + "\n")
            ne_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], ne_spline(rhop_short[i]).item() / 1.e19) + "\n")
        ne_file.flush()
        ne_file.close()
        ne_tb_file.flush()
        ne_tb_file.close()
    else:
        print("Copying Te and ne matrix")
        Te_ne_matfile = open(os.path.join(working_dir, "Te_ne_matfile"), "w")
        Te_ne_matfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5n}: {1:1.4f}\n'.format(shot, time))
        Te_ne_matfile.write('   {0: 8n} {1: 8n}\n'.format(len(EQ.R), len(EQ.z)))
        Te_ne_matfile.write('Radial grid coordinates\n')
        cnt = 0
        for i in range(len(EQ.R)):
            Te_ne_matfile.write("  {0: 1.8E}".format(EQ.R[i]))
            if(cnt == columns):
                Te_ne_matfile.write("\n")
                cnt = 0
            else:
                cnt += 1
        if(cnt is not 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('Vertical grid coordinates\n')
        cnt = 0
        for j in range(len(EQ.z)):
            Te_ne_matfile.write("  {0: 1.8E}".format(EQ.z[j]))
            if(cnt == columns):
                Te_ne_matfile.write("\n")
                cnt = 0
            else:
                cnt += 1
        if(cnt is not 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('Te on grid\n')
        cnt = 0
        print("EQ.Bz shape", EQ.Bz.shape)
        print("Te shape", Te.shape)
        for i in range(len(Te)):
            for j in range(len(Te[i])):
                Te_ne_matfile.write("  {0: 1.8E}".format(Te[i][j]))
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
        if(cnt is not 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('ne on grid\n')
        cnt = 0
        for i in range(len(ne)):
            for j in range(len(ne[i])):
                Te_ne_matfile.write("  {0: 1.8E}".format(ne[i][j]))
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
    return 0

def make_WKBEAM_file(filename, data_dict, key):
    WKBEAM_file = open(filename, "w")
    WKBEAM_file.write("{0:d} {1:d}\n".format(len(data_dict["R_new"]), len(data_dict["z_new"])))
    for i in range(len(data_dict["R_new"])):
        WKBEAM_file.write("{0:3.7e} ".format(data_dict["R_new"][i]))
    WKBEAM_file.write("\n")
    for i in range(len(data_dict["z_new"])):
        WKBEAM_file.write("{0:3.7e} ".format(data_dict["z_new"][i]))
    WKBEAM_file.write("\n")
    for i in range(len(data_dict["z_new"])):
        for j in range(len(data_dict["R_new"])):
            WKBEAM_file.write("{0:3.7e} ".format(data_dict[key].T[i][j]))
        WKBEAM_file.write("\n")
    WKBEAM_file.flush()
    WKBEAM_file.close()

# def smooth_2D_profile(folder, filename, index):
#    if("Te" in filename):
#        level = 100
#    else:
#        level = np.log(1.e18)
#    keys = ["R", "z", filename]
#    data_dict = {}
#    for key in keys:
#        data_dict[key] = np.loadtxt(os.path.join(folder, key + "{0:n}".format(index))).T
#    cs = plt.contour(data_dict["R"], data_dict["z"], data_dict[filename], levels=level)
#    p = cs.collections[0].get_paths()[0]
#    v = p.vertices
#    R_c = v[:,0]
#    z_c = v[:,1]
#    for i in range(len(z_c)):


def make_input_data_for_WKBEAM(folder, index):
    keys = ["R", "z", "Br", "Bt", "Bz", "ne", "Te", "Psi"]
    data_dict = {}
    for key in keys:
        data_dict[key] = np.loadtxt(os.path.join(folder, key + "{0:n}".format(index))).T
    data_dict["R"] *= 1.e2
    data_dict["z"] *= 1.e2
    data_dict["R_new"] = np.linspace(np.min(data_dict["R"]), 370, 200)
    data_dict["z_new"] = np.linspace(np.min(data_dict["z"]), np.max(data_dict["z"]), 400)
    for key in keys:
        if(key == "Br"):
            Br_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], data_dict["Br"].T)
            data_dict["Br"] = Br_spl(data_dict["R_new"], data_dict["z_new"])
            make_WKBEAM_file(os.path.join(folder, "B_V_x.txt"), data_dict, key)
        elif(key == "Bt"):
            Bt_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], data_dict["Bt"].T)
            data_dict["Bt"] = Bt_spl(data_dict["R_new"], data_dict["z_new"])
            make_WKBEAM_file(os.path.join(folder, "B_Tbc_y.txt"), data_dict, key)
        elif(key == "Bz"):
            Bz_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], data_dict["Bz"].T)
            data_dict["Bz"] = Bz_spl(data_dict["R_new"], data_dict["z_new"])
        elif(key == "Psi"):
            Psi_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], data_dict["Psi"].T)
            data_dict["Psi"] = Psi_spl(data_dict["R_new"], data_dict["z_new"])
            make_WKBEAM_file(os.path.join(folder, "B_V_z.txt"), data_dict, key)
        elif(key == "ne"):
            ne_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], np.log(data_dict["ne"].T))
            data_dict["ne"] = np.exp(ne_spl(data_dict["R_new"], data_dict["z_new"]))
            make_WKBEAM_file(os.path.join(folder, "ne.txt"), data_dict, key)
        elif(key == "Te"):
            Te_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], np.log(data_dict["Te"]).T)
            data_dict["Te"] = np.exp(Te_spl(data_dict["R_new"], data_dict["z_new"]))
            make_WKBEAM_file(os.path.join(folder, "Te.txt"), data_dict, key)
    np.savetxt(os.path.join(folder, "R{0:n}".format(index)), data_dict["R_new"] / 100)
    np.savetxt(os.path.join(folder, "z{0:n}".format(index)), data_dict["z_new"] / 100)
    np.savetxt(os.path.join(folder, "Br{0:n}".format(index)), data_dict["Br"])
    np.savetxt(os.path.join(folder, "Bt{0:n}".format(index)), data_dict["Bt"])
    np.savetxt(os.path.join(folder, "Bz{0:n}".format(index)), data_dict["Bz"])
    np.savetxt(os.path.join(folder, "ne{0:n}".format(index)), data_dict["ne"])
    np.savetxt(os.path.join(folder, "Te{0:n}".format(index)), data_dict["Te"])
    np.savetxt(os.path.join(folder, "Psi{0:n}".format(index)), data_dict["Psi"])
    fig1 = plt.figure()
    plt.contourf(data_dict["R_new"] / 100, data_dict["z_new"] / 100, data_dict["ne"].T * 1.e-19, levels=np.linspace(0, 8.0, 30))
    fig2 = plt.figure()
    plt.contourf(data_dict["R_new"] / 100, data_dict["z_new"] / 100, data_dict["Te"].T * 1.e-3, levels=np.linspace(0, 1.5, 30))
    plt.show()
# make_input_data_for_WKBEAM("/ptmp1/work/sdenk/ECRad4/Ext_data/", 0)


# read_EQ_from_ext_data("/ptmp1/work/sdenk/ECRad/Ext_data/", 6.00000048, 0, 1.005)


def make_all_TORBEAM_rays_thread(args):
    working_dir = str(args[0])
    ECRad_data_dir = os.path.join(working_dir, "ECRad_data")
    shot = int(args[1])
    time = float(args[2])
    eq_exp = args[3]
    eq_diag = args[4]
    eq_ed = args[5]
    ray_launch = args[6]
    t_index = int(args[7])
    mode = int(args[8])
    plasma = args[9]
    host = args[10]
    bt_vac_correction = args[11]
    N_ray = args[12]
    prepare_TB_data(ECRad_data_dir, shot, time, eq_exp, eq_diag, eq_ed, t_index, mode, plasma, copy_Te_ne=False, bt_vac_correction=bt_vac_correction)
#    copyfile(os.path.join(ECRad_data_dir, "Te.dat"), \
#                 os.path.join(tb_path, "Te.dat"))
#    copyfile(os.path.join(ECRad_data_dir, "ne.dat"), \
#             os.path.join(tb_path, "ne.dat"))
#    copyfile(os.path.join(ECRad_data_dir, "topfile"), \
#             os.path.join(tb_path, "topfile"))
    make_all_TORBEAM_rays(ECRad_data_dir, shot, time, eq_exp, eq_diag, ray_launch, t_index, mode, plasma, N_ray)
    evt_out_2 = ThreadFinishedEvt(Unbound_EVT_THREAD_FINISHED, host.GetId())
    wx.PostEvent(host, evt_out_2)

def make_LUKE_data_no_data_load(working_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep, launches):
    LUKE_dir = os.path.join(working_dir, "{0:d}_{1:1.2f}_gy21_input".format(shot, time))
    if(not os.path.isdir(LUKE_dir)):
        os.mkdir(LUKE_dir)
    mode = -1  # hard coded to X-mode
    prepare_TB_data_no_data_load(LUKE_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep)
    inbeam_index = 0
    for launch in launches:
        make_inbeam(LUKE_dir, launch, mode, time, inbeam_index, cyl=True)
        inbeam_index += 1

def make_TORBEAM_no_data_load(working_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep, launches, ITM=False, ITER=False, Z_eff=None):
    TB_out_dir = os.path.join(working_dir, "{0:d}_{1:1.3f}_rays".format(shot, time))
    if(not os.path.isdir(TB_out_dir)):
        os.mkdir(TB_out_dir)
    org_path = os.getcwd()
    tb_lib_path = globalsettings.TB_path
    os.chdir(TB_out_dir)
    mode = -1  # hard coded to X-mode
    prepare_TB_data_no_data_load(TB_out_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep, ITM)
    beam_index = 0
    for launch in launches:
        make_inbeam(TB_out_dir, launch, mode, time, 0, cyl=False, ITM=ITM, ITER=ITER, Z_eff=Z_eff)
        try:
            call([os.path.join(tb_lib_path, "a.out"), ""])
        except OSError:
            print("Weird OS error")
            os.chdir(org_path)
            return
        copyfile(os.path.join(TB_out_dir, "t1_LIB.dat"), \
             os.path.join(TB_out_dir, "Rz_beam_{0:1d}.dat".format(beam_index + 1).replace(",", "")))
        copyfile(os.path.join(TB_out_dir, "inbeam.dat"), \
             os.path.join(TB_out_dir, "inbeam{0:1d}.dat".format(beam_index + 1).replace(",", "")))
        copyfile(os.path.join(TB_out_dir, "t1tor_LIB.dat"), \
             os.path.join(TB_out_dir, "xy_beam_{0:1d}.dat".format(beam_index + 1).replace(",", "")))
        copyfile(os.path.join(TB_out_dir, "t2_new_LIB.dat"), \
             os.path.join(TB_out_dir, "pw_j_beam_{0:1d}.dat".format(beam_index + 1).replace(",", "")))
        try:
            infile = open(os.path.join(TB_out_dir, "fort.41"))
            lines = infile.readlines()
            infile.close()
            outfile = open(os.path.join(TB_out_dir, "beam_{0:d}_detail_tb.dat".format(beam_index + 1)), "w")
            header = " "
            quants = ["s [m]", "R [m]", "z [m]", "phi [deg]", "rho_p", \
                      "n_e [1.e19 m^-3]", "T_e [keV]", "B_r [T]", \
                      "B_t [T]", "B_z [T]", "N", "N_par", "k_im [m-1]", \
                      "P/P_0", "dIds [A/m]", "(dP/ds)/P"]
            for quant in quants:
                header += "{0:11s} ".format(quant)
            outfile.write(header + " \n")
            for line in lines:
                outfile.write(line)
            outfile.flush()
            outfile.close()
        except IOError:
            print("No detailed output from torbeam")
        print("TORBEAM: Beam " + str(beam_index + 1) + "/" + str(len(launches)) + " complete")
        beam_index += 1
    os.chdir(org_path)

def make_LUKE_data(working_dir, shot, time, t_index, plasma, eq_exp, eq_diag, eq_ed, bt_vac_correction):
    LUKE_dir = os.path.join(working_dir, "{0:d}_{1:1.2f}_gy21_input".format(shot, time[t_index]))
    if(not os.path.isdir(LUKE_dir)):
        os.mkdir(LUKE_dir)
    mode = -1  # hard coded to X-mode
    prepare_TB_data(LUKE_dir, shot, time[t_index], eq_exp, eq_diag, eq_ed, t_index, mode, plasma, copy_Te_ne=False, bt_vac_correction=bt_vac_correction)
    gy_list = load_all_ECRH(shot)
    inbeam_index = 0
    for gy in gy_list:
        if(gy.avail):
            if(gy.PW[np.argmin(np.abs(gy.time - time[t_index]))] > 1.0):
                launch_set = launch()
                launch_set.parse_gy(gy, time[t_index])
                make_inbeam(LUKE_dir, launch_set, mode, time[t_index], inbeam_index, cyl=True)
                inbeam_index += 1

def prepare_TB_data(working_dir, shot, time, eq_exp, eq_diag, eq_ed, t_index, mode, plasma, copy_Te_ne=False, bt_vac_correction=1.005):
    if(eq_exp == "Ext" or eq_exp == "ext" or eq_exp == "EXT"):
        make_topfile_from_ext_data(working_dir, shot, time, \
                                   plasma["eq_data"][t_index], plasma["rhop"][t_index], \
                                   plasma["Te"][t_index], plasma["ne"][t_index], bt_vac_correction)
        return
    if(make_topfile(working_dir, shot, time, eq_exp, eq_diag, eq_ed, copy_Te_ne=copy_Te_ne, bt_vac_correction=bt_vac_correction)) is not 0:
        return
    TeSpline = InterpolatedUnivariateSpline(plasma["rhop"][t_index], plasma["Te"][t_index] / 1.e3, k=1)  # linear interpolation to avoid overshoot
    neSpline = InterpolatedUnivariateSpline(plasma["rhop"][t_index], plasma["ne"][t_index] / 1.e19, k=1)  # linear interpolation to avoid overshoot
    npts = 150  # Maximum TORBEAM can handle
    rhop = np.linspace(0.0, 1.06, npts)
    Te_file = open(os.path.join(working_dir, "Te.dat"), "w")
    Te_file.write("{0: 7n}\n".format(npts))
    Te_remapped = TeSpline(rhop)
    for i in range(npts):
        Te_file.write("{0: 1.12E} {1: 1.12E}\n".format(rhop[i], Te_remapped[i]))
    Te_file.close()
    ne_file = open(os.path.join(working_dir, "ne.dat"), "w")
    ne_file.write("{0: 7n}\n".format(npts))
    ne_remapped = neSpline(rhop)
    for i in range(npts):
        ne_file.write("{0: 1.12E} {1: 1.12E}\n".format(rhop[i], ne_remapped[i]))
    ne_file.close()

def prepare_TB_data_no_data_load(working_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep, ITM=False):
    if(make_topfile_no_data_load(working_dir, shot, time, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep, ITM) is not 0):
        return
    Te_file = open(os.path.join(working_dir, "Te.dat"), "w")
    if(len(rho_prof) > 150):
        new_rho_prof = np.linspace(0.0, 1.1, 150)
        Te_spl = InterpolatedUnivariateSpline(rho_prof, Te_prof / 1.e3, k=1)
        ne_spl = InterpolatedUnivariateSpline(rho_prof, ne_prof / 1.e19, k=1)
        Te_new = Te_spl(new_rho_prof)
        ne_new = ne_spl(new_rho_prof)
    else:
        new_rho_prof = rho_prof
        Te_new = Te_prof / 1.e3
        ne_new = ne_prof / 1.e19
    Te_file.write("{0: 7n}\n".format(len(new_rho_prof)))
    for i in range(len(new_rho_prof)):
        Te_file.write("{0: 1.12E} {1: 1.12E}\n".format(new_rho_prof[i], Te_new[i]))
    Te_file.close()
    ne_file = open(os.path.join(working_dir, "ne.dat"), "w")
    ne_file.write("{0: 7n}\n".format(len(new_rho_prof)))
    for i in range(len(new_rho_prof)):
        ne_file.write("{0: 1.12E} {1: 1.12E}\n".format(new_rho_prof[i], ne_new[i]))
    ne_file.close()

def make_inbeam(working_dir, launch, mode, time, inbeam_no=0, cyl=False, ITM=False, ITER=False, Z_eff=None):
    tb_lib_path = globalsettings.TB_path
    inbeam_file = open(os.path.join(tb_lib_path, "inbeam.dat"))
    inbeam_lines = inbeam_file.readlines()
    inbeam_file.close()
    double_check_dict = {}
    double_check_dict["freq_set"] = False
    double_check_dict["mode_set"] = False
    double_check_dict["pol_angle_set"] = False
    double_check_dict["tor_angle_set"] = False
    double_check_dict["x_set"] = False
    double_check_dict["y_set"] = False
    double_check_dict["z_set"] = False
    double_check_dict["curv_y_set"] = False
    double_check_dict["curv_z_set"] = False
    double_check_dict["width_y_set"] = False
    double_check_dict["width_z_set"] = False
    double_check_dict["PW_set"] = False
    if(Z_eff is not None):
        double_check_dict["xzeff_set"] = False
    for i in range(len(inbeam_lines)):
        try:
            if("xf" in inbeam_lines[i]):
                inbeam_lines[i] = " xf =           {0:1.6e},\n".format(launch.f)
                double_check_dict["freq_set"] = True
        except ValueError as e:
            print(e)
            print(launch.f)
            return
        if("nmod" in inbeam_lines[i]):
            inbeam_lines[i] = " nmod = {0: n},\n".format(mode)
            double_check_dict["mode_set"] = True
        elif("xtordeg" in inbeam_lines[i]):
            if(not ITER):
                if(cyl):
                    inbeam_lines[i] = " xtordeg =           {0:1.6f},\n".format(launch.phi_tor)
                else:
                    inbeam_lines[i] = " xtordeg =           {0:1.6f},\n".format(launch.phi_tor + launch.phi)
            elif(launch.alpha is not None and launch.beta is not None):
                if(cyl):
                    inbeam_lines[i] = " xtordeg =           {0:1.6f},\n".format(np.rad2deg(launch.beta))
                else:
                    inbeam_lines[i] = " xtordeg =           {0:1.6f},\n".format(np.rad2deg(launch.beta))
            else:
                print("Error ITER selcted but neither alpha nor beta provided")
                raise(AttributeError)
            double_check_dict["tor_angle_set"] = True
        elif("xpoldeg" in inbeam_lines[i]):
            if(not ITER):
                inbeam_lines[i] = " xpoldeg =           {0:1.6f},\n".format(launch.theta_pol)
            elif(launch.alpha is not None and launch.beta is not None):
                inbeam_lines[i] = " xpoldeg =           {0:1.6f},\n".format(np.rad2deg(launch.alpha))
            else:
                print("Error ITER selcted but neither alpha nor beta provided")
                raise(AttributeError)
            double_check_dict["pol_angle_set"] = True
        elif("xxb" in inbeam_lines[i]):
            if(cyl):
                inbeam_lines[i] = " xxb =           {0:1.6f},\n".format(launch.R * 100.0)
            else:
                inbeam_lines[i] = " xxb =           {0:1.6f},\n".format(launch.x * 100.0)
            double_check_dict["x_set"] = True
        elif("xyb" in inbeam_lines[i]):
            if(cyl):
                inbeam_lines[i] = " xyb =           {0:1.6f},\n".format(0.0)
            else:
                inbeam_lines[i] = " xyb =           {0:1.6f},\n".format(launch.y * 100.0)  #
            double_check_dict["y_set"] = True
        elif("xzb" in inbeam_lines[i]):
            inbeam_lines[i] = " xzb =           {0:1.6f},\n".format(launch.z * 100.0)
            double_check_dict["z_set"] = True
        elif(Z_eff is not None and "xzeff" in inbeam_lines[i]):
            Z_eff_av = np.mean(Z_eff)
            inbeam_lines[i] = " xzeff =           {0:1.6f},\n".format(Z_eff_av)
            double_check_dict["xzeff_set"] = True
        if(launch.gy_launch):
            if(launch.curv_y == 0.e0):
                print("Zero encountered in curvature")
                print("Error!: Gyrotron data not properly read")
                return
            if("xryyb" in inbeam_lines[i]):
                inbeam_lines[i] = " xryyb =           {0:1.6f},\n".format(launch.curv_y * 100.0)
                double_check_dict["curv_y_set"] = True
            elif("xrzzb" in inbeam_lines[i]):
                inbeam_lines[i] = " xrzzb =           {0:1.6f},\n".format(launch.curv_z * 100.0)
                double_check_dict["curv_z_set"] = True
            elif("xwyyb" in inbeam_lines[i]):
                inbeam_lines[i] = " xwyyb =           {0:1.6f},\n".format(launch.width_y * 100.0)
                double_check_dict["width_y_set"] = True
            elif("xwzzb" in inbeam_lines[i]):
                inbeam_lines[i] = " xwzzb =           {0:1.6f},\n".format(launch.width_z * 100.0)
                double_check_dict["width_z_set"] = True
            elif("xpw0" in inbeam_lines[i]):
                inbeam_lines[i] = " xpw0 =           {0:1.6f},\n".format(launch.PW / 1.e6)
                double_check_dict["PW_set"] = True
        else:
            if("xryyb" in inbeam_lines[i]):
                if(launch.curv_y is not None):
                    inbeam_lines[i] = " xryyb =           {0:1.6f},\n".format(launch.curv_y * 100.0)
                else:
                    inbeam_lines[i] = " xryyb =           {0:1.6f},\n".format(0.8 * 100.0)
                double_check_dict["curv_y_set"] = True
            elif("xrzzb" in inbeam_lines[i]):
                if(launch.curv_z is not None):
                    inbeam_lines[i] = " xrzzb =           {0:1.6f},\n".format(launch.curv_z * 100.0)
                else:
                    inbeam_lines[i] = " xrzzb =           {0:1.6f},\n".format(0.8 * 100.0)
                double_check_dict["curv_z_set"] = True
            elif("xwyyb" in inbeam_lines[i]):
                if(launch.width_y is not None):
                    inbeam_lines[i] = " xwyyb =           {0:1.6f},\n".format(launch.width_y * 100.0)
                else:
                    inbeam_lines[i] = " xwyyb =           {0:1.6f},\n".format(0.02 * 100.0)
                double_check_dict["width_y_set"] = True
            elif("xwzzb" in inbeam_lines[i]):
                if(launch.width_z is not None):
                    inbeam_lines[i] = " xwzzb =           {0:1.6f},\n".format(launch.width_z * 100.0)
                else:
                    inbeam_lines[i] = " xwzzb =           {0:1.6f},\n".format(0.02 * 100.0)
                double_check_dict["width_z_set"] = True
            elif("xpw0" in inbeam_lines[i]):
                inbeam_lines[i] = " xpw0 =           {0:1.6f},\n".format(500.e3 / 1.e6)
                double_check_dict["PW_set"] = True
    for checked_quant in double_check_dict.keys():
        if(not double_check_dict[checked_quant]):
            print("ERROR!! " + checked_quant + " was not put into the inbeam file")
            raise IOError
    print('Inbeam file successfully parsed!')
    if(inbeam_no == 0):
        inbeam_file = open(os.path.join(working_dir, "inbeam.dat"), "w")
    else:
        inbeam_file = open(os.path.join(working_dir, "inbeam{0:d}.dat".format(inbeam_no + 1)), "w")
    for line in inbeam_lines:
        inbeam_file.write(line)
    inbeam_file.flush()
    inbeam_file.close()

def make_all_TORBEAM_rays(working_dir, shot, time, eq_exp, eq_diag, ray_launch, t_index, mode, plasma, N_ray=1):
    # shot = int(sys.argv[1])
    # time = float(sys.argv[2])
#    diag = sys.argv[3]
#    mode = sys.argv[4]
#    job = sys.argv[5]
#    scope = sys.argv[6]
    org_path = os.getcwd()
    os.chdir(working_dir)
    ray_out_path = os.path.join(working_dir, "ray")
    if(not os.path.isdir(ray_out_path)):
        os.mkdir(ray_out_path)
    for ich in range(len(np.squeeze(ray_launch["f"][t_index][::N_ray]))):
        launch_set = launch()
        if(ray_launch["width"] is None):
            launch_set.parse_custom(ray_launch["f"][t_index][::N_ray][ich], ray_launch["x"][t_index][::N_ray][ich], \
                                    ray_launch["y"][t_index][::N_ray][ich], ray_launch["z"][t_index][::N_ray][ich],
                                    ray_launch["tor_ang"][t_index][::N_ray][ich], ray_launch["pol_ang"][t_index][::N_ray][ich])
        else:
            launch_set.parse_custom(ray_launch["f"][t_index][::N_ray][ich], ray_launch["x"][t_index][::N_ray][ich], \
                                    ray_launch["y"][t_index][::N_ray][ich], ray_launch["z"][t_index][::N_ray][ich],
                                    ray_launch["tor_ang"][t_index][::N_ray][ich], ray_launch["pol_ang"][t_index][::N_ray][ich],
                                    ray_launch["width"][t_index][::N_ray][ich], ray_launch["dist_focus"][t_index][::N_ray][ich])
        make_inbeam(working_dir, launch_set, mode, time, cyl=True)
        try:
            call([os.path.join(tb_path, "a.out"), ""])
        except OSError:
            print("Weird OS error")
            os.chdir(org_path)
            return
        copyfile(os.path.join(working_dir, "t1_LIB.dat"), \
             os.path.join(ray_out_path, "ray_ch_R{0:04d}tb.dat".format(ich + 1)))
        copyfile(os.path.join(working_dir, "t1tor_LIB.dat"), \
             os.path.join(ray_out_path, "ray_ch_x{0:04d}tb.dat".format(ich + 1)))
        try:
            infile = os.path.join(working_dir, "fort.41")
            lines = infile.readlines()
            infile.close()
            outfile = open(os.path.join(ray_out_path, "beam_{0:d}_detail_tb.dat".format(ich + 1)), "w")
            header = " "
            quants = ["s [m]", "R [m]", "z [m]", "phi [deg]", "rho_p", \
                      "n_e [1.e19 m^-3]", "T_e [keV]", "B_r [T]", \
                      "B_t [T]", "B_z [T]", "N", "N_par", "k_im [m-1]", \
                      "P/P_0", "dIds [A/m]", "(dP/ds)/P"]
            for quant in quants:
                header += "{0:12s} ".format(quant)
            outfile.write(header)
            for line in lines:
                outfile.write(line)
            outfile.flush()
            outfile.close()
        except IOError:
            print("No detailed output from torbeam")
        print("TORBEAM: Channel " + str(ich + 1) + "/" + str(len(ray_launch["f"][t_index][::N_ray])) + " complete")
    os.chdir(org_path)
# make_ext_data_for_testing_grids("/ptmp1/work/sdenk/ECRad3/Ext_data", 30839, np.array([2.5]), "AUGD", "EQH", 0)
# make_ext_data_for_testing("/ptmp1/work/sdenk/ECRad2/Ext_data", 33698, np.array([1.06]), "AUGD", "EQH", 0)

def make_LUKE_input_mat(working_dir, shot, times, IDA_exp="AUGD", IDA_ed=0, EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005):
    index = 0
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed, bt_vac_correction=bt_vac_correction)
    plasma_data = load_IDA_data(shot, timepoints=times, exp="AUGD", ed=0)
    gy_list = load_all_active_ECRH(shot)
    mdict = {}
    mdict["t"] = []
    mdict["Psi_sep"] = []
    mdict["Psi_ax"] = []
    mdict["R_ax"] = []
    mdict["z_ax"] = []
    mdict["R_sep"] = []
    mdict["z_sep"] = []
    mdict["R"] = []
    mdict["z"] = []
    mdict["Psi"] = []
    mdict["rhop"] = []
    mdict["Br"] = []
    mdict["Bt"] = []
    mdict["Bz"] = []
    mdict["rhop_prof"] = []
    mdict["Te_prof"] = []
    mdict["ne_prof"] = []
    mdict["I_p"] = []
    if(EQ_diag == "IDE"):
        mdict["I_p_cor"] = []
    mdict["Vloop"] = []
    for i_gy in range(len(gy_list)):
        mdict["freq_gy{0:d}".format(i_gy + 1)] = []
        mdict["mode_gy{0:d}".format(i_gy + 1)] = []
        mdict["pol_angle_gy{0:d}".format(i_gy + 1)] = []
        mdict["tor_angle_gy{0:d}".format(i_gy + 1)] = []
        mdict["x_gy{0:d}".format(i_gy + 1)] = []
        mdict["y_gy{0:d}".format(i_gy + 1)] = []
        mdict["z_gy{0:d}".format(i_gy + 1)] = []
        mdict["curv_y_gy{0:d}".format(i_gy + 1)] = []
        mdict["curv_z_gy{0:d}".format(i_gy + 1)] = []
        mdict["width_y_gy{0:d}".format(i_gy + 1)] = []
        mdict["width_z_gy{0:d}".format(i_gy + 1)] = []
        mdict["PW_gy{0:d}".format(i_gy + 1)] = []
    for time in plasma_data["time"]:
        EQ_t = EQ_obj.GetSlice(time)
        mdict["t"].append(time)
        mdict["Psi_sep"].append(EQ_t.Psi_sep)
        mdict["Psi_ax"].append(EQ_t.Psi_ax)
        mdict["R_ax"].append(EQ_t.R_ax)
        mdict["z_ax"].append(EQ_t.z_ax)
        mdict["R_sep"].append(EQ_t.R_sep)
        mdict["z_sep"].append(EQ_t.z_sep)
        mdict["R"].append(EQ_t.R)
        mdict["z"].append(EQ_t.z)
        mdict["Psi"].append(EQ_t.Psi)
        mdict["rhop"].append(EQ_t.rhop)
        mdict["Br"].append(EQ_t.Br)
        mdict["Bt"].append(EQ_t.Bt)
        mdict["Bz"].append(EQ_t.Bz)
        mdict["rhop_prof"].append(plasma_data["rhop"][index])
        mdict["Te_prof"].append(plasma_data["Te"][index])
        mdict["ne_prof"].append(plasma_data["ne"][index])
        if(EQ_diag == "IDE"):
            mdict["I_p_cor"].append(get_RELAX_target_current(shot, time, exp=EQ_exp, ed=EQ_ed, smoothing=1.e-3))
            mdict["I_p"].append(get_total_current(shot, time, exp=EQ_exp, diag="IDG", ed=0, smoothing=1.e-3))
        else:
            mdict["I_p"].append(get_total_current(shot, time, exp=EQ_exp, diag="FPC", ed=0, smoothing=1.e-3))
        mdict["Vloop"].append(get_Vloop(shot, time, exp="AUGD", ed=0, smoothing=1.e-2))
        i_gy = 0
        for gy in gy_list:
            mdict["freq_gy{0:d}".format(i_gy + 1)].append(gy.f)
            mdict["mode_gy{0:d}".format(i_gy + 1)].append(-1)  # Just X mode for now
            if(np.isscalar(gy.theta_pol)):
                mdict["pol_angle_gy{0:d}".format(i_gy + 1)].append(-gy.theta_pol)
                mdict["tor_angle_gy{0:d}".format(i_gy + 1)].append(-gy.phi_tor)
            else:
                mdict["pol_angle_gy{0:d}".format(i_gy + 1)].append(-gy.theta_pol[np.argmin(np.abs(gy.time - time))])
                mdict["tor_angle_gy{0:d}".format(i_gy + 1)].append(-gy.phi_tor[np.argmin(np.abs(gy.time - time))])
            mdict["x_gy{0:d}".format(i_gy + 1)].append(gy.x)
            mdict["y_gy{0:d}".format(i_gy + 1)].append(gy.y)
            mdict["z_gy{0:d}".format(i_gy + 1)].append(gy.z)
            mdict["curv_y_gy{0:d}".format(i_gy + 1)].append(gy.curv_y)
            mdict["curv_z_gy{0:d}".format(i_gy + 1)].append(gy.curv_z)
            mdict["width_y_gy{0:d}".format(i_gy + 1)].append(gy.width_y)
            mdict["width_z_gy{0:d}".format(i_gy + 1)].append(gy.width_z)
            mdict["PW_gy{0:d}".format(i_gy + 1)].append(gy.PW[np.argmin(np.abs(gy.time - time))])
            i_gy += 1
        index += 1
    savemat(os.path.join(working_dir, "LUKE_data_shot_{0:d}".format(shot)), mdict)
    print("Successfully created", os.path.join(working_dir, "LUKE_data_shot_{0:d}".format(shot)))


def load_TB_beam_details(filename):
    # Note: All TB inputs with lenght units are expected to be in "m" !
    # Note: All output is converted to "m"
    tb_data = np.loadtxt(filename, skiprows=1)
    tb_dict = {}
    tb_dict["s"] = tb_data.T[0]
    tb_dict["R"] = tb_data.T[1]
    tb_dict["z"] = tb_data.T[2]
    tb_dict["phi"] = np.deg2rad(tb_data.T[3])  # Rad here
    tb_dict["rho_p"] = tb_data.T[4]
    tb_dict["n_e"] = tb_data.T[5] * 1.e19  # To SI units
    tb_dict["T_e"] = tb_data.T[6] * 1.e3  # To eV
    tb_dict["B_R"] = tb_data.T[7]
    tb_dict["B_t"] = tb_data.T[8]
    tb_dict["B_z"] = tb_data.T[9]
    tb_dict["N"] = tb_data.T[10]
    tb_dict["N_par"] = tb_data.T[11]
    tb_dict["k_im"] = tb_data.T[12]
    tb_dict["P_norm"] = tb_data.T[13]
    tb_dict["dIds"] = tb_data.T[14]
    tb_dict["dPds/P"] = tb_data.T[15]
    return tb_dict

def eval_Psi(params, args):
    Psi_spl = args[0]
    return Psi_spl(params[0], params[1], grid=False)

def make_Ext_data_from_TB_files(path, outpath, write_stuff=True):
    tb_file = open(path)
    cur_line = tb_file.readline()
    cur_line = np.fromstring(tb_file.readline(), dtype=np.int, sep=" ")
    m = cur_line[0]
    n = cur_line[1]
    cur_line = tb_file.readline()
    cur_line = np.fromstring(tb_file.readline(), dtype=np.float, sep=" ")
    Psi_sep = cur_line[-1]
    data_dict = {}
    data_dict["R"] = []
    data_dict["z"] = []
    data_dict["Br"] = []
    data_dict["Bt"] = []
    data_dict["Bz"] = []
    data_dict["Psi"] = []
    for key in ["R", "z", "Br", "Bt", "Bz", "Psi"]:
        # Descriptor
        line = tb_file.readline()
        if(key is "R"):
            elemt_cnt = m
            reshape = m
        elif(key is "z"):
            elemt_cnt = n
            reshape = n
        else:
            elemt_cnt = m * n
            reshape = (n, m)
        while len(data_dict[key]) < elemt_cnt:
            line = tb_file.readline()
            noms = np.fromstring(line, dtype=np.float, sep=" ")
            if(len(noms) == 0):
                print(line)
                print(len(np.array(data_dict[key]).flatten()), elemt_cnt)
                raise IOError
            else:
                for num in noms:
                    data_dict[key].append(num)
        data_dict[key] = np.array(data_dict[key]).reshape(reshape).T
    if(Psi_sep < data_dict["Psi"][m / 2][n / 2]):
        data_dict["Psi"] *= -1.0
        Psi_sep *= -1.0
    Psi_spl = RectBivariateSpline(data_dict["R"], data_dict["z"], data_dict["Psi"])
    res = minimize(eval_Psi, [np.mean(data_dict["R"]), 0.0], args=[Psi_spl], bounds=[[np.min(data_dict["R"]), \
                                                                                      np.max(data_dict["R"])], \
                                                                                     [np.min(data_dict["z"]), \
                                                                                      np.max(data_dict["z"])]])
    R_ax = res.x[0]
    z_ax = res.x[1]
    Psi_ax = Psi_spl(R_ax, z_ax)
    if(np.min(data_dict["Psi"]) - Psi_ax < 0.0):
        Psi_ax = np.min(data_dict["Psi"])
    rhop = np.sqrt((data_dict["Psi"] - Psi_ax) / (Psi_sep - Psi_ax))
#    print(R_ax, z_ax, Psi_spl(R_ax, z_ax, grid=False))
#    plt.contourf(R, z, Bt.T, levels=np.linspace(0.0, 8.0, 30))  # np.sqrt(Psi.T)
    cs = plt.contour(data_dict["R"], data_dict["z"], np.sqrt(rhop.T), levels=[1.2])  #
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    R_wall = v[:, 0]
    z_wall = v[:, 1]
    if(not os.path.isdir(outpath)):
        os.mkdir(outpath)
    np.savetxt(os.path.join(outpath, "Ext_vessel.bd"), np.array([R_wall, z_wall]).T)
    plt.contour(data_dict["R"], data_dict["z"], rhop.T, levels=np.linspace(0.1, 1.2, 12), linestyle="--")
    plt.contour(data_dict["R"], data_dict["z"], rhop.T, levels=[1.0], linestyle="-")
    plt.plot(R_ax, z_ax, "x")
    plt.plot(R_wall, z_wall, "+")
    plt.show()
    n_prof = np.fromstring(tb_file.readline(), dtype=np.int, sep=" ")[0]
    print(n_prof)
    rhop_prof = []
    Te = []
    ne = []
    for i in range(n_prof):
        cur_line = np.fromstring(tb_file.readline(), dtype=np.float, sep=" ")
        rhop_prof.append(cur_line[0])
        ne.append(cur_line[1])
        Te.append(cur_line[2])
    rhop = np.array(rhop)
    ne = np.array(ne) * 1.e19
    Te = np.array(Te) * 1.e3
    fig = plt.figure()
    plt.plot(rhop_prof, Te)
    fig = plt.figure()
    plt.plot(rhop_prof, ne)
    plt.show()
    if(write_stuff):
        make_ext_data_for_testing_from_data(outpath, 1, [60.0], data_dict["R"], data_dict["z"], \
                                        data_dict["Br"], data_dict["Bt"], data_dict["Bz"], data_dict["Psi"], \
                                        R_ax, z_ax, 0.0, 1.0, rhop, ne, Te)

class launch:
    def __init__(self):
        self.gy_launch = False

    def parse_custom(self, f, x, y, z, phi_tor, theta_pol, width=None, dist_foc=None):
        self.f = f
        self.x = x
        self.y = y
        self.z = z
        self.R = np.sqrt(x ** 2 + y ** 2)
        self.phi = np.arctan2(self.y, self.x)
        self.phi_tor = phi_tor
        self.theta_pol = theta_pol
        self.curv_y = dist_foc
#        (np.pi * (f ** 2 * np.pi * width ** 4 + np.sqrt(f ** 4 * np.pi ** 2 * width ** 8 - \
#                                                                      4 * cnst.c ** 2 * f ** 2 * width ** 4 * \
#                                                                      z ** 2))) / (2.e0 * cnst.c ** 2 * z)
        self.curv_z = self.curv_y
        self.width_y = width
        self.width_z = width

    def parse_gy(self, gy, time, avt=None, alpha=None, beta=None):
        self.f = gy.f
        self.R = gy.R
        self.phi = gy.phi
        self.x = gy.x
        self.y = gy.y
        self.z = gy.z
        if(avt is not None):
            t1 = np.argmin(np.abs(gy.time - (time - avt)))
            t2 = np.argmin(np.abs(gy.time - (time + avt)))
            if(t1 == t2):
                t2 += 1
            if(np.isscalar(gy.phi_tor)):
                self.phi_tor = -gy.phi_tor  # ECRH -> TOBREAM convention!
                self.theta_pol = -gy.theta_pol  # ECRH -> TOBREAM convention!
            else:
                self.phi_tor = np.mean(-gy.phi_tor[t1:t2])  # ECRH -> TOBREAM convention!
                self.theta_pol = np.mean(-gy.theta_pol[t1:t2])  # ECRH -> TOBREAM convention!
            self.PW = np.mean(gy.PW[t1:t2])
        else:
            if(np.isscalar(gy.phi_tor)):
                self.phi_tor = -gy.phi_tor  # ECRH -> TOBREAM convention!
                self.theta_pol = -gy.theta_pol  # ECRH -> TOBREAM convention!
            else:
                self.phi_tor = -gy.phi_tor[np.argmin(np.abs(gy.time - time))]  # ECRH -> TOBREAM convention!
                self.theta_pol = -gy.theta_pol[np.argmin(np.abs(gy.time - time))]  # ECRH -> TOBREAM convention!
            self.PW = gy.PW[np.argmin(np.abs(gy.time - time))]
        self.curv_y = gy.curv_y
        self.curv_z = gy.curv_z
        self.width_y = gy.width_y
        self.width_z = gy.width_z
        self.alpha = alpha  # Launching angles following ITER convetion
        self.beta = beta
        if(self.curv_y == 0.0 or self.curv_z == 0.0 or self.width_y == 0.0 or self.width_y == 0.0):
            print("Warning!: gyrotron not properly set up when parsing launch configuraiton")
        self.gy_launch = True

if(__name__ == "__main__"):
    make_Ext_data_from_TB_files("/tokp/work/sdenk/ECRad2/ECRad_data/topfile", "/tokp/work/sdenk/ECRad2/Ext_data/", False)
