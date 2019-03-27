'''
Created on Mar 6, 2018

@author: sdenk
'''
from electron_distribution_utils import read_LUKE_profiles, read_waves_mat_to_beam, read_dist_mat_to_beam
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import constants as cnst
import numpy as np
from plotting_configuration import *
import os
from equilibrium_utils_AUG import EQData


def compare_Beam_quantities(shot, time, wave_mat_filename, tb_data_path, EQ_exp, EQ_diag, EQ_ed):
    wave_mat = loadmat(wave_mat_filename, squeeze_me=True)
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    EQ_slice = EQ_obj.GetSlice(time)
    Gray_beam = read_waves_mat_to_beam(wave_mat, EQ_slice)
    TB_data = np.loadtxt(tb_data_path)
    # s [m], R [m], Z[m], phi [deg], rho_p, n_e [10^19 m-3], T_e [keV], B_R, B_phi, B_Z [T], N, N_par, k_im [m-1], Power/P_0, dIds [A/m], (dP/ds)/P
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax.plot(TB_data.T[4], TB_data.T[-3])
    ax.plot(Gray_beam.rays[1][0]["rhop"], 1.0 - Gray_beam.rays[1][0]["PW"] / np.max(Gray_beam.rays[1][0]["PW"]), "--")
    ax2.plot(TB_data.T[1], TB_data.T[2])
    ax2.plot(Gray_beam.rays[1][0]["R"], Gray_beam.rays[1][0]["z"], "--")
    ax3.plot(TB_data.T[4], cnst.e / cnst.m_e * np.sqrt(TB_data.T[7] ** 2 + TB_data.T[8] ** 2 + TB_data.T[9] ** 2) / (2.e9 * np.pi))
    ax3.plot(Gray_beam.rays[1][0]["rhop"], Gray_beam.rays[1][0]["omega_c"] / (2.e9 * np.pi), "--")
    plt.show()

def compare_PowerDepo(shot, time, EQ_exp, EQ_diag, EQ_ed, tb_data_path, dist_mat_filename, wave_mat_filename, Luke_mat_filename=None):
    wave_mat = loadmat(wave_mat_filename, squeeze_me=True)
    dist_mat = loadmat(dist_mat_filename, squeeze_me=True)
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    EQ_slice = EQ_obj.GetSlice(time)
    RELAX_beam = read_dist_mat_to_beam(dist_mat)
    if(RELAX_beam.PW_tot < 10.e4):
        scale = 1.e3
        prefix = r"\kilo"
    else:
        scale = 1.e0
        prefix = r"\mega"
    Gray_beam = read_waves_mat_to_beam(wave_mat, EQ_slice)
    if(not Luke_mat_filename is None):
        LUKE_beam, C3PO_beam = read_LUKE_profiles(Luke_mat_filename)
    rhop_tb = np.linspace(0.0, 1.2, 200)
    PW_tot_tb = 0.0
    pw_tb = np.zeros(len(rhop_tb))
    j_tb = np.zeros(len(rhop_tb))
    i_beam = 1
    tb_beams = []
    fig = plt.figure(figsize=(8.5, 8.5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    while(True):
        cur_tb_path = os.path.join(tb_data_path, "pw_j_beam_{0:d}.dat".format(i_beam))
        if(not os.path.isfile(cur_tb_path)):
            break
        else:
            inbeam_file = open(os.path.join(tb_data_path, "inbeam{0:d}.dat".format(i_beam)))
            inbeam_lines = inbeam_file.readlines()
            inbeam_file.close()
            for line in inbeam_lines:
                if("xpw0" in line):
                    PW_tot_tb += float(line.rsplit("=", 1)[1].replace(r"\n", "").replace(r",", ""))
            rho, Pw, j = np.loadtxt(cur_tb_path, unpack=True)
            PW_spl = InterpolatedUnivariateSpline(rho, Pw, ext=0)
            pw_tb += PW_spl(rhop_tb)
            j_spl = InterpolatedUnivariateSpline(rho, j, ext=0)
            j_tb += j_spl(rhop_tb)
            tb_beam_dict = {}
            tb_beam_dict["R"], tb_beam_dict["z"], tb_beam_dict["R1"], \
                tb_beam_dict["z1"], tb_beam_dict["R2"], tb_beam_dict["z2"] = np.loadtxt(os.path.join(tb_data_path, "Rz_beam_{0:d}.dat".format(i_beam)), unpack=True)
            tb_beam_dict["x"], tb_beam_dict["y"], tb_beam_dict["x1"], \
                tb_beam_dict["y1"], tb_beam_dict["x2"], tb_beam_dict["yz2"] = np.loadtxt(os.path.join(tb_data_path, "xy_beam_{0:d}.dat".format(i_beam)), unpack=True)
            tb_beams.append(tb_beam_dict)
        i_beam += 1
#    for tb_beam in tb_beams:
#        plt.plot(tb_beam["R"] / 100.0, tb_beam["z"] / 100.0, "-b")
#        plt.plot(tb_beam["R1"] / 100.0, tb_beam["z1"] / 100.0, ":b")
#        plt.plot(tb_beam["R2"] / 100.0, tb_beam["z2"] / 100.0, ":b")
#    for i in range(len(Gray_beam.rays)):
#        for j in range(len(Gray_beam.rays[i])):
#            plt.plot(Gray_beam.rays[i][j]["R"], Gray_beam.rays[i][j]["z"], "--k")
    scaling_TB = RELAX_beam.PW_tot / PW_tot_tb
    print("TORBEAM normalization", scaling_TB)
    ax1.plot(rhop_tb, pw_tb * scaling_TB * scale, ":", label="TORBEAM")
    ax1.plot(RELAX_beam.rhop, RELAX_beam.PW * scale, "-", label="RELAX")
    if(not Luke_mat_filename is None):
        ax1.plot(LUKE_beam.rhop, LUKE_beam.PW * scale, "-", label="LUKE")
        ax1.plot(C3PO_beam.rhop, C3PO_beam.PW * scale, "--", label="C3PO")
    scaling_Gray = RELAX_beam.PW_tot / Gray_beam.PW_tot
    print("GRAY normalization", scaling_Gray)
    ax1.plot(Gray_beam.rhop, Gray_beam.PW * scaling_Gray * scale, "--", label="GRAY")
    ax2.plot(rhop_tb, j_tb * scaling_TB * scale, ":", label="TORBEAM")
    ax2.plot(RELAX_beam.rhop, RELAX_beam.j * scale, "-", label="RELAX")
    if(not Luke_mat_filename is None):
        ax2.plot(LUKE_beam.rhop, LUKE_beam.j * scale, "-", label="LUKE")
    ax2.plot(Gray_beam.rhop, Gray_beam.j * scaling_Gray * scale, "--", label="GRAY")
    ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"$P \mathrm{d}V\,[\si{" + prefix + r"\watt\per\cubic\metre}]$")
    ax2.set_ylabel(r"$j \mathrm{d}V\,[\si{" + prefix + r"\ampere\per\cubic\metre}]$")
    ax1.get_xaxis().set_major_locator(MaxNLocator(nbins=4, prune='lower'))
    ax1.get_xaxis().set_minor_locator(MaxNLocator(nbins=8))
    ax1.get_yaxis().set_major_locator(MaxNLocator(nbins=4))
    ax1.get_yaxis().set_minor_locator(MaxNLocator(nbins=8))
    ax2.get_xaxis().set_major_locator(MaxNLocator(nbins=4, prune='lower'))
    ax2.get_xaxis().set_minor_locator(MaxNLocator(nbins=8))
    ax2.get_yaxis().set_major_locator(MaxNLocator(nbins=4))
    ax2.get_yaxis().set_minor_locator(MaxNLocator(nbins=8))
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()







if(__name__ == "__main__"):
#    compare_PowerDepo(33705, 4.90, "AUGD", "IDE", 0, "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_10/ecfm_data/fRe/", \
#                      "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_10/ecfm_data/", \
#                      "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_10/ecfm_data/fRe/Dist_33705_4.90.mat", \
#                      "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_10/ecfm_data/fRe/GRAY_rays_33705_4.90.mat")
#    compare_PowerDepo(33705, 4.90, "AUGD", "IDE", 0, "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/33705_4.900_rays/", \
#                      "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_10/ecfm_data/", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/Dist_33705_4.90.mat", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/GRAY_rays_33705_4.90.mat")
#    compare_PowerDepo(33697, 4.80, "AUGD", "IDE", 0, "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/33697_4.800_rays/", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/Dist_33697_4.80.mat", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/GRAY_rays_33697_4.80.mat")
#    compare_PowerDepo(33705, 4.90, "AUGD", "IDE", 0, "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/33705_4.900_rays/", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/Dist_33705_4.90.mat", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/GRAY_rays_33705_4.90.mat")
#    compare_PowerDepo(34663, 3.60, "AUGD", "IDE", 0, "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/34663_3.600_rays/", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/Dist_34663_3.60.mat", \
#                      "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/GRAY_rays_34663_3.60.mat")
    compare_Beam_quantities(34663, 3.60, "/afs/ipp/home/s/sdenk/Documentation/Data/DistData/GRAY_rays_34663_3.60.mat", \
                            "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/fort.41", "AUGD", "IDE", 0)

