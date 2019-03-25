'''
Created on Jun 16, 2017

@author: sdenk
'''
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import os
from GlobalSettings import AUG, TCV
from electron_distribution_utils import read_waves_mat_to_beam, read_dist_mat_to_beam
if(AUG):
    from equilibrium_utils_AUG import EQData
elif(TCV):
    from equilibrium_utils_TCV import make_B_min, EQData
else:
    print('Neither AUG nor TCV selected')
    raise(ValueError('No system selected!'))
from scipy.io import loadmat
from ECRad_Results import ECRadResults
from plotting_configuration import *
from plotting_core import plotting_core

def overlap_pw_deposition(path, shot, time, waves_mat_file, dist_mat_file, ECRad_save_file, \
                          diag, channel_list=None, mode="X", EQ_exp="AUGD", EQ_diag="EQH", \
                          EQ_ed=0, bt_vac_correction=1.005, noTB=False, ray_depo_plot=True):
    result = ECRadResults()
    waves_mat = loadmat(os.path.join(path, waves_mat_file), squeeze_me=True)
    dist_mat = loadmat(os.path.join(path, dist_mat_file), squeeze_me=True)
    if(ECRad_save_file is not None):
        result.Config = result.from_mat_file(ECRad_save_file)
        itime = np.argmin(result.Config.time - time)
    BPD = []
    BPD_rhop = []
    BPD_labels = []
    diag_resonances = {}
    if(not result.init):
        print("Error loading stuff")

    for ich in channel_list:
        if(mode == "X"):
            BPD_rhop.append(result.BPD["rhopX"][itime][result.diag == diag][ich - 1])
            BPD.append(result.BPD["BPDX"][itime][result.diag == diag][ich - 1])
        else:
            BPD_rhop.append(result.BPD["rhopO"][itime][result.diag == diag][ich - 1])
            BPD.append(result.BPD["BPDO"][itime][result.diag == diag][ich - 1])
        # Power deposition profiles are only defined for whole flux surfaces, hence we want the BPD to be the same here
        BPD[-1] = BPD[-1][0:len(BPD_rhop[-1]) / 2][::-1] + BPD[-1][len(BPD_rhop[-1]) / 2:len(BPD_rhop[-1])]
        BPD_rhop[-1] = BPD_rhop[-1][len(BPD_rhop[-1]) / 2:len(BPD_rhop[-1])]
        BPD_labels.append(r"BPD ch. no. {0:d}".format(ich))
    diag_rays = {}
    diag_rays["R"] = []
    diag_rays["z"] = []
    diag_resonances["R"] = []
    diag_resonances["z"] = []
    diag_resonances["rhop"] = []
    for ich in channel_list:
        if(result.Config.N_ray == 1):
            if(mode == "X"):
                diag_rays["R"].append([np.sqrt(result.ray["xX"][itime][result.diag == diag][ich - 1] ** 2 + \
                                         result.ray["yX"][itime][result.diag == diag][ich - 1] ** 2)])
                diag_rays["z"].append([result.ray["zX"][itime][result.diag == diag][ich - 1]])
                i_res = np.argmax(result.ray["BPDX"][itime][result.diag == diag][ich - 1])
                diag_resonances["rhop"].append(result.ray["rhopX"][itime][result.diag == diag][ich - 1][i_res])
            else:
                diag_rays["R"].append([np.sqrt(result.ray["xO"][itime][result.diag == diag][ich - 1] ** 2 + \
                                         result.ray["yO"][itime][result.diag == diag][ich - 1] ** 2)])
                diag_rays["z"].append([result.ray["zO"][itime][result.diag == diag][ich - 1]])
                i_res = np.argmax(result.ray["BPDO"][itime][result.diag == diag][ich - 1])
                diag_resonances["rhop"] = result.ray["rhopO"][itime][result.diag == diag][ich - 1][i_res]
            diag_resonances["R"].append(diag_rays["R"][-1])
            diag_resonances["z"].append(diag_rays["z"][-1])
        else:
            diag_rays["R"].append([])
            diag_rays["z"].append([])
            for ir in range(result.Config.N_ray):
                if(mode == "X"):
                    diag_rays["R"][-1].append(np.sqrt(result.ray["xX"][itime][result.diag == diag][ich - 1][ir] ** 2 + \
                                             result.ray["yX"][itime][result.diag == diag][ich - 1][ir] ** 2))
                    diag_rays["z"][-1].append(result.ray["zX"][itime][result.diag == diag][ich - 1][ir])
                    if(ir == 0):
                        i_res = np.argmax(result.ray["BPDX"][itime][result.diag == diag][ich - 1][ir])
                        diag_resonances["R"].append(diag_rays["R"][-1][-1][i_res])
                        diag_resonances["z"].append(diag_rays["z"][-1][-1][i_res])
                        diag_resonances["rhop"].append(result.ray["rhopX"][itime][result.diag == diag][ich - 1][ir][i_res])
                else:
                    diag_rays["R"][-1].append(np.sqrt(result.ray["xX"][itime][result.diag == diag][ich - 1][ir] ** 2 + \
                                             result.ray["yX"][itime][result.diag == diag][ich - 1][ir] ** 2))
                    diag_rays["z"][-1].append(result.ray["zX"][itime][result.diag == diag][ich - 1][ir])
                    if(ir == 0):
                        i_res = np.argmax(result.ray["BPDO"][itime][result.diag == diag][ich - 1][ir])
                        diag_resonances["R"].append(diag_rays["R"][-1][-1][i_res])
                        diag_resonances["z"].append(diag_rays["z"][-1][-1][i_res])
                        diag_resonances["rhop"].append(result.ray["rhopO"][itime][result.diag == diag][ich - 1][ir][i_res])
    EQObj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed, bt_vac_correction=bt_vac_correction)
    EQSlice = EQObj.read_EQ_from_shotfile(time)
    linear_beam = read_waves_mat_to_beam(waves_mat, EQSlice)
    quasi_linear_beam = read_dist_mat_to_beam(dist_mat)
    fig = plt.figure(figsize=(8.5, 8.0), tight_layout=False)
    fig.clf()
    pc_obj = plotting_core(fig)
    fig = pc_obj.overlap_plot(path, shot, time, EQSlice, BPD_rhop, BPD, BPD_labels, diag_rays, diag_resonances, linear_beam, \
                              quasi_linear_beam, noTB, ray_depo_plot)
    plt.show()

def ECRH_and_LOS_config(path, shot, time, waves_mat_file, ECRad_save_file, \
                          channel_list=None, mode="X", EQ_exp="AUGD", EQ_diag="EQH", \
                          EQ_ed=0, bt_vac_correction=1.005, plot_label=None, plot_label2=None):
    result = ECRadResults()
    if(ECRad_save_file is not None):
        result.Config = result.from_mat_file(ECRad_save_file)
        itime = np.argmin(result.Config.time - time)
    EQObj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed, bt_vac_correction=bt_vac_correction)
    EQSlice = EQObj.read_EQ_from_shotfile(time)
    waves_mat = loadmat(os.path.join(path, waves_mat_file), squeeze_me=True)
    linear_beam = read_waves_mat_to_beam(waves_mat, EQSlice)
    fig = plt.figure(figsize=(7.25, 8.0), tight_layout=False)
    fig2 = plt.figure(figsize=(8.5, 8.0), tight_layout=False)
    fig.clf()
    fig2.clf()
    pc_obj = plotting_core(fig, fig2, title=False)
    args = []
    args.append(None)
    args.append(shot)
    args.append(time)
    args.append(EQSlice.R)
    args.append(EQSlice.z)
    args.append(EQSlice.rhop)
    args.append(EQSlice.R_ax)
    args.append(EQSlice.z_ax)
    args.append(linear_beam)
    args.append(result)
    args.append(channel_list)
    args.append(mode)
    args.append(path)
    fig, fig2 = pc_obj.beam_plot(args)
    if(plot_label is not None):
        fig.text(0.05, 0.95, plot_label + r")")
    if(plot_label2 is not None):
        fig2.text(0.05, 0.95, plot_label2 + r")")
    fig.text(0.125, 0.95, r"\#{0:d}".format(shot) + r", $t = $ \SI{" + "{0:1.2f}".format(time) + r"}{\second}")
    fig2.text(0.125, 0.95, r"\#{0:d}".format(shot) + r", $t = $ \SI{" + "{0:1.2f}".format(time) + r"}{\second}")
    fig.gca().set_xlim((1.0, 2.1))
    fig.gca().set_ylim((-0.4, 0.3))
    fig2.gca().set_xlim((-2.3, 0.5))
    fig2.gca().set_ylim((-0.5, 2.45))
    plt.show()

# overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 33697, 4.80, \
#                          "GRAY_rays_33697_4.80.mat", "Dist_33697_4.80.mat", \
#                          "/tokp/work/sdenk/ECRad/ECRad_33697_CTA_ed11.mat", \
#                          "CTA", [3, 48], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005)
# overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 33697, 4.80, \
#                          "GRAY_rays_33697_4.80.mat", "Dist_33697_4.80.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_33697_ECECTA_ed 1.mat", \
#                          "CTA", [3, 48], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005)
# overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 33705, 4.90, \
#                          "GRAY_rays_33705_4.90.mat", "Dist_33705_4.90.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_33705_CTA_ed 1.mat", \
#                          "CTA", [3, 48], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005)
# overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 34663, 3.60, \
#                          "GRAY_rays_34663_3.60.mat", "Dist_34663_3.60.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_34663_ECECTA_ed3.mat", \
#                          "CTA", [], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005, ray_depo_plot=False)
# overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/fRelax/", 31594, 1.30, \
#                          "GRAY_rays_31594_1.30.mat", "Dist_31594_1.30.mat", \
#                          None, \
#                          None, None, mode="X", EQ_exp="AUGD", EQ_diag="EQH", \
#                          EQ_ed=0, bt_vac_correction=1.005)
overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 33705, 4.90, \
                          "GRAY_rays_33705_4.90.mat", "Dist_33705_4.90.mat", \
                          "/tokp/work/sdenk/ECRad2/ECRad_33705_ECECTA_ed2.mat", \
                          "CTA", [3, 48], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
                          EQ_ed=0, bt_vac_correction=1.005, ray_depo_plot=False)
# ECRH_and_LOS_config("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 33697, 4.80, \
#                          "GRAY_rays_33697_4.80.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_33697_ECECTA_ed3.mat", \
#                          [3, 70], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005, plot_label="a")
# ECRH_and_LOS_config("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 35140, 2.31, \
#                          "GRAY_rays_35140_2.31.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_35140_ECECTCCTA_ed4.mat", \
#                          [3, 70, 105], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005, plot_label="a")
# ECRH_and_LOS_config("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 35140, 6.42, \
#                          "GRAY_rays_35140_5.43.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_35140_ECECTCCTA_ed4.mat", \
#                          [3, 70, 105], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005, plot_label="a")
# ECRH_and_LOS_config("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 33705, 4.90, \
#                          "GRAY_rays_33705_4.90.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_33705_ECECTA_ed5.mat", \
#                          [3, 70], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005, plot_label="b", plot_label2="a")
# ECRH_and_LOS_config("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 34663, 3.60, \
#                          "GRAY_rays_34663_3.60.mat", \
#                          "/tokp/work/sdenk/ECRad2//ECRad_34663_ECECTA_ed4.mat", \
#                          [3, 70], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005, plot_label="c", plot_label2="b")
# overlap_pw_deposition("/afs/ipp/u/sdenk/Documentation/Data/DistData/", 34663, 3.60, \
#                          "GRAY_rays_34663_3.60.mat", "Dist_34663_3.60.mat", \
#                          "/tokp/work/sdenk/ECRad2/ECRad_34663_CTA_ed 1.mat", \
#                          "CTA", [3, 48], mode="X", EQ_exp="AUGD", EQ_diag="IDE", \
#                          EQ_ed=0, bt_vac_correction=1.005)
