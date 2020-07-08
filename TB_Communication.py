'''
Created on Oct 11, 2016

@author: sdenk
'''
import os
import numpy as np
import sys
sys.path.append("../ECRad_Pylib")
from subprocess import call
from Global_Settings import globalsettings
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
from plotting_configuration import *
from shutil import copyfile

tb_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/"

tb_path_itm = "/marconi_work/eufus_gw/work/g2sdenk/torbeam/lib-OUT/"
# "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam/"
#
class Ray:
    def __init__(self, s, x, y, z, H, N, N_cold, Te=0, ne=0, Y=0, X=0, x_tb=0, y_tb=0, z_tb=0, \
                                                                       x_tbp1=0, y_tbp1=0, z_tbp1=0, \
                                                                       x_tbp2=0, y_tbp2=0, z_tbp2=0):
        self.s = s  # can either be 1D or 2D depending on the number of rays (first dimension)
#        print("s", s)
        self.x = x
#        print("x", x)
        self.y = y
#        print("y", y)
        self.z = z
#        print("z", z)
        self.R = np.sqrt(x ** 2 + y ** 2)
        self.phi = np.arctan(y / x) * 180.0 / np.pi
        self.x_tb = x_tb
        self.y_tb = y_tb
        self.z_tb = z_tb
        self.x_tbp1 = x_tbp1
        self.y_tbp1 = y_tbp1
        self.z_tbp1 = z_tbp1
        self.x_tbp2 = x_tbp2
        self.y_tbp2 = y_tbp2
        self.z_tbp2 = z_tbp2
        self.R_tb = 0
        self.phi_tb = 0
        self.R_tbp1 = 0
        self.phi_tbp1 = 0
        self.R_tbp2 = 0
        self.phi_tbp2 = 0
        if(type(x_tb) != int):
            self.R_tb = np.sqrt(x_tb ** 2 + y_tb ** 2)
            self.phi_tb = np.arctan(y / x) * 180.0 / np.pi
            if(type(x_tbp1) != int):
                self.R_tbp1 = np.sqrt(x_tbp1 ** 2 + y_tbp1 ** 2)
                self.phi_tbp1 = np.arctan(y_tbp1 / x_tbp1) * 180.0 / np.pi
                self.R_tbp2 = np.sqrt(x_tbp2 ** 2 + y_tbp2 ** 2)
                self.phi_tbp2 = np.arctan(y_tbp2 / x_tbp2) * 180.0 / np.pi
        self.H = H
        self.N = N
        self.N_cold = N_cold
        self.Y = Y
        self.X = X

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
    topfile.write('Number of radial and vertical grid points for discharge :{0:5d}: {1:1.4f}\n'.format(shot, time))
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
    if(cnt is not columns and cnt is not 0):
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
    if(cnt is not columns and cnt is not 0):
        topfile.write('\n')
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

def make_Te_ne_files(working_dir, rho, Te, ne):
    # makes Te and ne files for TORBEAM and ECRad
    Te_file = open(os.path.join(working_dir, "Te_file.dat"), "w")
    Te_tb_file = open(os.path.join(working_dir, "Te.dat"), "w")
    lines = 150
    Te_file.write("{0: 7d}".format(len(rho)) + "\n")
    for i in range(len(rho)):
        Te_file.write("{0: 1.12E} {1: 1.12E}".format(rho[i], Te[i]) + "\n")
    Te_file.flush()
    Te_file.close()
    Te_tb_file.write("{0: 7d}".format(lines) + "\n")
    Te_spline = InterpolatedUnivariateSpline(rho, Te, k=1)
    rho_short = np.linspace(np.min(rho), np.max(rho), lines)
    for i in range(len(rho_short)):
        try:
            Te_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rho_short[i], Te_spline(rho_short[i]).item() / 1.e03) + "\n")
        except ValueError:
            print(rho_short[i], Te_spline(rho_short[i]))
            raise(ValueError)
    Te_tb_file.flush()
    Te_tb_file.close()
    ne_file = open(os.path.join(working_dir, "ne_file.dat"), "w")
    ne_file.write("{0: 7d}".format(len(rho)) + "\n")
    for i in range(len(rho)):
        ne_file.write("{0: 1.12E} {1: 1.12E}".format(rho[i], ne[i]) + "\n")
    ne_file.flush()
    ne_file.close()
    ne_tb_file = open(os.path.join(working_dir, "ne.dat"), "w")
    ne_tb_file.write("{0: 7d}".format(lines) + "\n")
    ne_spline = InterpolatedUnivariateSpline(rho, ne, k=1)
    rho_short = np.linspace(np.min(rho), np.max(rho), lines)
    for i in range(len(rho_short)):
        ne_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rho_short[i], ne_spline(rho_short[i]).item() / 1.e19) + "\n")
    ne_tb_file.flush()
    ne_tb_file.close()

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

def make_TORBEAM_no_data_load(working_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, \
                              psi, Br, Bt, Bz, psi_ax, psi_sep, launches, ITM=False, \
                              ITER=False, Z_eff=None, mode = -1):
    TB_out_dir = os.path.join(working_dir, "{0:d}_{1:1.3f}_rays".format(shot, time))
    if(not os.path.isdir(TB_out_dir)):
        os.mkdir(TB_out_dir)
    org_path = os.getcwd()
    tb_lib_path = globalsettings.TB_path
    os.chdir(TB_out_dir)
    prepare_TB_data_no_data_load(TB_out_dir, shot, time, rho_prof, Te_prof, ne_prof, R, z, psi, Br, Bt, Bz, psi_ax, psi_sep, ITM)
    beam_index = 0
    for launch in launches:
        make_inbeam(TB_out_dir, launch, mode, time, 0, cyl=False, ITM=ITM, ITER=ITER, Z_eff=Z_eff)
        try:
            call([os.path.join(tb_lib_path, "a.out"), ""])
        except OSError:
            print("Weird OS error")
            print()
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
    inbeam_file = open(os.path.join(tb_lib_path, "inbeam_tracing.dat"))
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
        try:
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
        except AttributeError:
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
    for checked_quant in double_check_dict:
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

def make_mdict_from_TB_files(path, eq_only=False):
    mdict = {}
    tb_file = open(path)
    tb_file.readline()
    cur_line = np.fromstring(tb_file.readline(), dtype=np.int, sep=" ")
    m = cur_line[0]
    n = cur_line[1]
    cur_line = tb_file.readline()
    cur_line = np.fromstring(tb_file.readline(), dtype=np.float, sep=" ")
    mdict["Psi_sep"] = cur_line[-1]
    mdict["R"] = []
    mdict["z"] = []
    mdict["Br"] = []
    mdict["Bt"] = []
    mdict["Bz"] = []
    mdict["Psi"] = []
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
        while len(mdict[key]) < elemt_cnt:
            line = tb_file.readline()
            noms = np.fromstring(line, dtype=np.float, sep=" ")
            if(len(noms) == 0):
                print(line)
                print(len(np.array(mdict[key]).flatten()), elemt_cnt)
                raise IOError
            else:
                for num in noms:
                    mdict[key].append(num)
        mdict[key] = np.array(mdict[key]).reshape(reshape).T
    if(mdict["Psi_sep"] < mdict["Psi"][m / 2][n / 2]):
        mdict["Psi"] *= -1.0
        mdict["Psi_sep"] *= -1.0
    Psi_spl = RectBivariateSpline(mdict["R"], mdict["z"], mdict["Psi"])
    res = minimize(eval_Psi, [np.mean(mdict["R"]), 0.0], args=[Psi_spl], bounds=[[np.min(mdict["R"]), \
                                                                                      np.max(mdict["R"])], \
                                                                                     [np.min(mdict["z"]), \
                                                                                      np.max(mdict["z"])]])
    R_ax = res.x[0]
    z_ax = res.x[1]
    mdict["Psi_ax"] = Psi_spl(R_ax, z_ax)
    if(np.min(mdict["Psi"]) - mdict["Psi_ax"] < 0.0):
        mdict["Psi_ax"] = np.min(mdict["Psi"])
#     plt.contour(mdict["R"], mdict["z"], mdict["rhop"].T, levels=np.linspace(0.1, 1.2, 12), linestyle="--")
#     plt.contour(mdict["R"], mdict["z"], mdict["rhop"].T, levels=[1.0], linestyle="-")
#     plt.plot(R_ax, z_ax, "x")
#     plt.show()
    if(not eq_only):
        n_prof = np.fromstring(tb_file.readline(), dtype=np.int, sep=" ")[0]
        mdict["rhop_prof"] = []
        mdict["Te"] = []
        mdict["ne"] = []
        for i in range(n_prof):
            cur_line = np.fromstring(tb_file.readline(), dtype=np.float, sep=" ")
            mdict["rhop_prof"].append(cur_line[0])
            mdict["Te"].append(cur_line[1])
            mdict["ne"].append(cur_line[2])
    for key in mdict:
        if(key not in ["R", "z"]):
            mdict[key] = np.array([mdict[key]]) #equlibrium entries is expected to have time as first dimension
    return mdict

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
    pass