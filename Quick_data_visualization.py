'''
Created on Jul 21, 2017

@author: sdenk
'''
import numpy as np
from subprocess import call
from scipy.integrate import quad
from glob import glob
from matplotlib import cm
import os
from shutil import copyfile
from data_processing import remove_mode
from Diags import ECRH_diag
from equilibrium_utils_AUG import EQData, make_rhop_signed_axis
from shotfile_handling_AUG import get_diag_data_no_calib, get_data_calib, load_IDA_data, get_shot_heating, \
                                  get_NPA_data, get_ECE_spectrum, get_Thomson_data, get_plasma_current, \
                                  get_NPA_data, get_ECE_spectrum, \
                                  get_Thomson_data, get_RMC_data_calib, get_data_calib_entire_shot, get_z_mag
import fconf
from get_ECRH_config import get_ECRH_viewing_angles
from Diags import Diag
from plotting_configuration import *
from plotting_core import plotting_core
from scipy.signal import resample
from scipy.io import readsav
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline

def compare_power_deops(filename):
    sav = readsav(filename)
    rhop_Te = rhop_Te = sav["plotdata"][0][5][0][0][0][1][0]
    Te = sav["plotdata"][0][5][0][1][0][1]
    egop = sav["plotdata"][0][6][0][1][0][0]





def B_plot(folder, shotno, time, comp_folder, ich, N_filename, mode):
    fig1 = plt.figure()
    fig2 = plt.figure()
    pc_obj = plotting_core(fig1, fig2)
    fig1, fig2 = pc_obj.B_plot(folder, shotno, time, comp_folder, ich, 1.65, os.path.join(folder, "ECRad_data","IchTB", N_filename), mode, True)
    plt.show()


def standard_overview_plot_w_CTA(shot):
    fig = plt.figure(figsize=(7.5, 8.5))
    fig_2 = plt.figure(figsize=(12.0, 8.49))
    pc = plotting_core(fig, fig_2)
    # fig, rhop_res1, rhop_res2 = pc.Show_TRad(r"C:\nssf", "26639", "2.05", 4,6, "Th", False, False)
    # fig =pc.stacked_plot(fig,filename, shot, channelposition)
    # root = "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/31223/"
    # 30907_cview/"
    # filenamelist = ["31223_DCN.txt", "31223_NBI.txt", "31223_PECRH.txt", "31223_ECE.txt"]
    # for i in range(len(filenamelist)):
    #    filenamelist[i] = root + filenamelist[i]
    # shot = 30907
    # labels = ["DCN H1", "DCN H5", "NBI Power", "ERCH Power","0.00","1.05"]
    time = [0.0, 8.4]
    data4 = get_NPA_data(shot)
    print(data4[0].shape, data4[1].shape)
    if(data4[0].shape != data4[1].shape):
        return
    data4[1][data4[1] > 0.15e5] = 0.45e5
    IDA_ed, time, plasma_dict = load_IDA_data(shot, np.linspace(time[0], time[1], 300))
    ECE_diag = Diag("ECE", "AUGD", "RMD", 0)
    std_dev, ECE_data = get_data_calib(diag=ECE_diag, shot=shot, time=time, t_smooth=1.e-3)
    Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "CTA" + "_" + str(shot) + "_calib_fact.txt", skiprows=2)
    CTA_diag = Diag("CTA", "AUGD", "CTA", 0)
    std_dev, CTA_data = get_data_calib(diag=CTA_diag, shot=shot, time=time, t_smooth=1.e-3, \
                                       calib=Calib_file.T[1], std_dev_calib=Calib_file.T[2] / 100.0 * Calib_file.T[1])
    ne = np.zeros(time.shape)
    Te = np.zeros(time.shape)
    Trad = np.zeros(time.shape)
    ECE_ch_no1 = 50
    ECE_ch_no2 = 50
    data2 = get_shot_heating(shot)
    for i in range(len(time)):
        ne[i] = np.mean(plasma_dict["ne"][i][plasma_dict["rhop"][i] < 0.1]) * 1.e-19
        Te[i] = np.mean(plasma_dict["Te"][i][plasma_dict["rhop"][i] < 0.1]) * 1.e-3
        # Trad[i] = np.mean(ECE_data[1][i][ECE_ch_no1 - 1 : ECE_ch_no2 - 1])
        Trad[i] = np.mean(CTA_data[1][i][ECE_ch_no1 - 1 : ECE_ch_no2])
    # plt.plot(time, Trad)
    fig = pc.stacked_plot_2_0(fig, [time, ne], data2, [[time, Te], [time, Trad]], data4)  # dist_plot(fig,Te)
    # fig = pc.stacked_plot_time_trace_3(fig, filenamelist,shot, labels)
                                    # single_plot(fig,filename, shot, channelposition,2.4)
    # fig, fig_2 = pc.get_figures()
    plt.show()

def TE_angle_CTA(shot):
    fig = plt.figure(figsize=(7.5, 8.5))
    fig_2 = plt.figure(figsize=(12.0, 8.49))
    pc = plotting_core(fig, fig_2)
    # fig, rhop_res1, rhop_res2 = pc.Show_TRad(r"C:\nssf", "26639", "2.05", 4,6, "Th", False, False)
    # fig =pc.stacked_plot(fig,filename, shot, channelposition)
    # root = "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/31223/"
    # 30907_cview/"
    # filenamelist = ["31223_DCN.txt", "31223_NBI.txt", "31223_PECRH.txt", "31223_ECE.txt"]
    # for i in range(len(filenamelist)):
    #    filenamelist[i] = root + filenamelist[i]
    # shot = 30907
    # labels = ["DCN H1", "DCN H5", "NBI Power", "ERCH Power","0.00","1.05"]
    time = [0.0, 8.4]
    IDA_ed, time, plasma_dict = load_IDA_data(shot, np.linspace(time[0], time[1], 300))
    ne = np.zeros(time.shape)
    Te = np.zeros(time.shape)
    data = []
    for i in range(len(time)):
        Te[i] = np.mean(plasma_dict["Te"][i][plasma_dict["rhop"][i] < 0.1]) * 1.e-3
    data.append({})
    data[-1]["x"] = time
    data[-1]["y"] = Te
    data[-1]["name"] = r"Core $T_\mathrm{e}$"
    data[-1]["ax_flag"] = r"Te_trace"
    data[-1]["ax_index"] = 0
    CTA_diag = Diag("CTA", "AUGD", "CTA", 0)
    CTA_time, CTA_data = get_diag_data_no_calib(diag=CTA_diag, shot=shot)
    data.append({})
    data[-1]["x"] = CTA_time
    data[-1]["y"] = CTA_data[38]
    data[-1]["name"] = "CTA channel 39"
    data[-1]["ax_flag"] = "Sig_vs_time"
    data[-1]["ax_index"] = 1
    gy = get_ECRH_viewing_angles(shot, 6, CTA_diag.base_freq_140)
    data.append({})
    data[-1]["x"] = gy.time
    data[-1]["y"] = gy.theta_pol
    data[-1]["name"] = r"$\theta_\mathrm{pol}$"
    data[-1]["ax_flag"] = r"theta_phi_trace"
    data[-1]["ax_index"] = 2
    data.append({})
    data[-1]["x"] = gy.time
    data[-1]["y"] = gy.phi_tor
    data[-1]["name"] = r"$\phi_\mathrm{tor}$"
    data[-1]["ax_flag"] = r"theta_phi_trace"
    data[-1]["ax_index"] = 2
    # plt.plot(time, Trad)
    fig = pc.stacked_plot_2_0_easy_trace(fig, data)  # dist_plot(fig,Te)
    plt.show()
# test_plot(33698)

def IDA_ne_2D_comp(shot1, shot2, t1, t2):
    IDA_ed, time1, plasma_dict1 = load_IDA_data(shot1, np.linspace(t1, t2, 3000))
    IDA_ed, time2, plasma_dict2 = load_IDA_data(shot2, np.linspace(t1, t2, 3000))
    time1_new = []
    ne1_new = []
    smoothing = 20
    i1 = 0
    i2 = i1 + smoothing
    while(i2 < len(time1) and not i1 == i2):
        time1_new.append(np.mean(time1[i1:i2]))
        ne1_new.append(np.mean(plasma_dict1["ne"][i1:i2], axis=0))
        i1 = i2
        if(i1 + smoothing < len(time1)):
            i2 = i1 + smoothing
        else:
            i2 = len(time1) - 1
    ne1_new = np.array(ne1_new)
    i1 = 0
    i2 = i1 + smoothing
    time2_new = []
    ne2_new = []
    while(i2 < len(time2) and not i1 == i2):
        time2_new.append(np.mean(time2[i1:i2]))
        ne2_new.append(np.mean(plasma_dict2["ne"][i1:i2], axis=0))
        i1 = i2
        if(i1 + smoothing < len(time1)):
            i2 = i1 + smoothing
        else:
            i2 = len(time1) - 1
#        plt.plot(plasma_dict1["rhop"][0], ne2_new[-1])
#        plt.show()
    ne2_new = np.array(ne2_new)
    time3, trace1 = get_plasma_current(shot1)
    time4, trace2 = get_plasma_current(shot2)
    heating = get_shot_heating(shot1)
    time5 = heating[0][0]
    trace3 = heating[0][1]
    heating = get_shot_heating(shot2)
    time6 = heating[0][0]
    trace4 = heating[0][1]
    levels = np.linspace(0.25, 3.5, 20)
    print("Plotting")
    fig = plt.figure(figsize=(12.0, 8.49))
    ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 2), (0, 1), rowspan=2, sharey=ax1)
    ax3 = plt.subplot2grid((4, 2), (2, 0), sharex=ax1)
    ax3b = ax3.twinx()
    ax4 = plt.subplot2grid((4, 2), (2, 1), sharex=ax1, sharey=ax3)
    ax4b = ax4.twinx()
    ax5 = plt.subplot2grid((4, 2), (3, 0), sharex=ax1)
    ax3b.get_shared_y_axes().join(ax3b, ax4b)
    ax6 = plt.subplot2grid((4, 2), (3, 1), sharex=ax1, sharey=ax5)
    cont1 = ax1.contourf(time1_new, plasma_dict1["rhop"][0], ne1_new.T / 1.e19, cmap=plt.cm.get_cmap("plasma"), levels=levels)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax1.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    cont2 = ax2.contourf(time2_new, plasma_dict2["rhop"][0], ne2_new.T / 1.e19, cmap=plt.cm.get_cmap("plasma"), levels=levels)
    cb = fig.colorbar(cont1, ax=ax2, ticks=[1.0, 2.0, 3.0])
    cb.set_label(r"$n_\mathrm{e}$ [\SI{1.e19}{\per\cubic\metre}]")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel(r"$\rho_\mathrm{pol}$")
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.plot(time3[::10][np.logical_and(time3[::10] > t1, time3[::10] < t2)], trace1[::10][np.logical_and(time3[::10] > t1, time3[::10] < t2)])
    ax3.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax3.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax3.spines['left'].set_color('blue')
    ax3.yaxis.label.set_color('blue')
    ax3.tick_params(axis='y', colors='blue')
    ax3.get_xaxis().set_visible(False)
    ax3b.get_yaxis().set_visible(False)
    ax3.set_ylabel(r"$I_\mathrm{p}$ [\si{\mega\ampere}]")
    ax3b.set_ylabel(r"$n_\mathrm{e, axis}$ [\SI{1.e19}{\per\cubic\metre}]")
    ax4b.spines['right'].set_color('red')
    ax4b.yaxis.label.set_color('red')
    ax4b.tick_params(axis='y', colors='red')
    ax4.plot(time4[::10][np.logical_and(time4[::10] > t1, time4[::10] < t2)], trace2[::10][np.logical_and(time4[::10] > t1, time4[::10] < t2)])
    ax4.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4b.plot(time2_new, ne2_new.T[0] / 1.e19, "r")
    ax4b.set_ylabel(r"$n_\mathrm{e, axis}$ [\SI{1.e19}{\per\cubic\metre}]")
    ax5.plot(time5[::10][np.logical_and(time5[::10] > t1, time5[::10] < t2)], trace3[::10][np.logical_and(time5[::10] > t1, time5[::10] < t2)])
    ax5.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax5.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax5.set_xlabel(r"$t$ [\si{\second}]")
    ax5.set_ylabel(r"$P_\mathrm{ECRH}$ [\si{\mega\watt}]")
    ax6.plot(time6[::10][np.logical_and(time6[::10] > t1, time6[::10] < t2)], trace4[::10][np.logical_and(time6[::10] > t1, time6[::10] < t2)])
    ax6.get_yaxis().set_visible(False)
    ax6.set_xlabel(r"$t$ [\si{\second}]")
    plt.show()

def CoreTrace(shot1, shot2, t1, t2):
    IDA_ed, time1, plasma_dict1 = load_IDA_data(shot1, np.linspace(t1, t2, 3000))
    IDA_ed, time2, plasma_dict2 = load_IDA_data(shot2, np.linspace(t1, t2, 3000))
    time1_new = []
    ne1_new = []
    Te1_new = []
    smoothing = 20
    i1 = 0
    i2 = i1 + smoothing
    while(i2 < len(time1) and not i1 == i2):
        time1_new.append(np.mean(time1[i1:i2]))
        ne1_new.append(np.mean(plasma_dict1["ne"][i1:i2], axis=0))
        Te1_new.append(np.mean(plasma_dict1["Te"][i1:i2], axis=0))
        i1 = i2
        if(i1 + smoothing < len(time1)):
            i2 = i1 + smoothing
        else:
            i2 = len(time1) - 1
    ne1_new = np.array(ne1_new)
    Te1_new = np.array(Te1_new)
    i1 = 0
    i2 = i1 + smoothing
    time2_new = []
    ne2_new = []
    Te2_new = []
    while(i2 < len(time2) and not i1 == i2):
        time2_new.append(np.mean(time2[i1:i2]))
        ne2_new.append(np.mean(plasma_dict2["ne"][i1:i2], axis=0))
        Te2_new.append(np.mean(plasma_dict2["Te"][i1:i2], axis=0))
        i1 = i2
        if(i1 + smoothing < len(time1)):
            i2 = i1 + smoothing
        else:
            i2 = len(time1) - 1
#        plt.plot(plasma_dict1["rhop"][0], ne2_new[-1])
#        plt.show()
    ne2_new = np.array(ne2_new)
    Te2_new = np.array(Te2_new)
    time3, trace1 = get_plasma_current(shot1)
    time4, trace2 = get_plasma_current(shot2)
    heating = get_shot_heating(shot1)
    time5 = heating[0][0]
    trace3 = heating[0][1]
    time6 = heating[1][0]
    trace4 = heating[1][1]
    heating = get_shot_heating(shot2)
    time7 = heating[0][0]
    trace5 = heating[0][1]
    time8 = heating[1][0]
    trace6 = heating[1][1]
    print("Plotting")
    fig = plt.figure(figsize=(12.0, 8.49))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1), sharey=ax1)
    ax3 = plt.subplot2grid((3, 2), (1, 0), sharex=ax1)
    ax3b = ax3.twinx()
    ax4 = plt.subplot2grid((3, 2), (1, 1), sharex=ax1, sharey=ax3)
    ax4b = ax4.twinx()
    ax5 = plt.subplot2grid((3, 2), (2, 0), sharex=ax1)
    ax3b.get_shared_y_axes().join(ax3b, ax4b)
    ax6 = plt.subplot2grid((3, 2), (2, 1), sharex=ax1, sharey=ax5)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax1.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax1.plot(time1_new, Te1_new.T[0] / 1.e3, "r")
    ax1.set_ylabel(r"$T_\mathrm{e}$ [\si{\kilo\electronvolt}]")
    ax2.plot(time2_new, Te2_new.T[0] / 1.e3, "r")
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax3.plot(time3[::10][np.logical_and(time3[::10] > t1, time3[::10] < t2)], trace1[::10][np.logical_and(time3[::10] > t1, time3[::10] < t2)])
    ax3.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax3.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax3.spines['left'].set_color('blue')
    ax3.yaxis.label.set_color('blue')
    ax3.tick_params(axis='y', colors='blue')
    ax3.get_xaxis().set_visible(False)
    ax3b.get_yaxis().set_visible(False)
    ax3.set_ylabel(r"$I_\mathrm{p}$ [\si{\mega\ampere}]")
    ax3b.plot(time1_new, ne1_new.T[0] / 1.e19, "r")
    ax3b.set_ylabel(r"$n_\mathrm{e, axis}$ [\SI{1.e19}{\per\cubic\metre}]")
    ax4b.spines['right'].set_color('red')
    ax4b.yaxis.label.set_color('red')
    ax4b.tick_params(axis='y', colors='red')
    ax4.plot(time4[::10][np.logical_and(time4[::10] > t1, time4[::10] < t2)], trace2[::10][np.logical_and(time4[::10] > t1, time4[::10] < t2)])
    ax4.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4b.plot(time2_new, ne2_new.T[0] / 1.e19, "r")
    ax4b.set_ylabel(r"$n_\mathrm{e, axis}$ [\SI{1.e19}{\per\cubic\metre}]")
    ax5.plot(time5[::10][np.logical_and(time5[::10] > t1, time5[::10] < t2)], trace3[::10][np.logical_and(time5[::10] > t1, time5[::10] < t2)])
    ax5.plot(time6[::10][np.logical_and(time6[::10] > t1, time6[::10] < t2)], trace4[::10][np.logical_and(time6[::10] > t1, time6[::10] < t2)])
    ax5.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax5.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax5.set_xlabel(r"$t$ [\si{\second}]")
    ax5.set_ylabel(r"$P_\mathrm{ECRH/NBI}$ [\si{\mega\watt}]")
    ax6.plot(time7[::10][np.logical_and(time7[::10] > t1, time7[::10] < t2)], trace5[::10][np.logical_and(time7[::10] > t1, time7[::10] < t2)])
    ax6.plot(time8[::10][np.logical_and(time8[::10] > t1, time8[::10] < t2)], trace6[::10][np.logical_and(time8[::10] > t1, time8[::10] < t2)])
    ax6.get_yaxis().set_visible(False)
    ax6.set_xlabel(r"$t$ [\si{\second}]")
    plt.tight_layout()
    plt.show()

def quick_ECE_trace(shot, channel_list, slicing=500):
#    time, f, data = get_RMC_data_calib(shot)
#    data = data.T
#    for ch in channel_list:
#        new_data = []
#        new_time = []
#        it1 = 0
#        it2 = slicing
#        while it2 < len(time):
#            new_data.append(np.mean(data[ch - 1][it1:it2]))
#            new_time.append(np.mean(time[it1:it2]))
#            it1 = it2
#            if(it2 + slicing >= len(time) and it2 < len(time)):
#                it2 = len(time) - 1
#            else:
#                it2 += slicing
#        new_time = np.array(new_time)
#        new_data = np.array(new_data)
    ECE_diag = Diag("ECE", "AUGD", "RMD", 0)
    t, sigs = get_data_calib_entire_shot(ECE_diag, shot)
    for ch in channel_list:
        binned_signal, binned_t = resample(sigs[ch - 1], len(sigs[ch - 1][::slicing]), t=t)
        plt.plot(binned_t, binned_signal / np.max(binned_signal))
    plt.show()

def ECE_time_trace_2D(shot, time_window, event_time, lines=False, ch_off=[], u_ch=None, Trad_min=1.e2, Trad_max=1.e5):
    time, Freq, Trad = get_RMC_data_calib(shot, time_window, ch_off)
    print(time)
    print("largest Trad", np.round(np.ceil(np.max(Trad)), -3))
    Trad[Trad < Trad_min] = Trad_min
    Trad[Trad > Trad_max] = Trad_max
    if(u_ch is not None):
        Freq = Freq[:u_ch]
        Trad = Trad.T[:u_ch].T
    cont = plt.contourf((time - event_time) * 1.e3, Freq / 1.e9, Trad.T, \
                        cmap=plt.cm.get_cmap("plasma"), \
                        norm=plt.cm.colors.LogNorm(vmin=Trad_min, vmax=Trad_max), vmin=Trad_min, vmax=Trad_max, \
                        levels=np.logspace(np.round(np.log10(Trad_min), 0), np.round(np.log10(Trad_max), 0), 20))
    plt.gca().set_xlabel(r"$t\,[\si{\milli\second}]$")
    plt.gca().set_ylabel(r"$f\,[\si{\giga\hertz}]$")
    plt.gcf().suptitle("\# {0:d}".format(shot))
    if(lines):
        plt.hlines(Freq / 1.e9, xmin=np.min(time - event_time) * 1.e3, xmax=np.max(time - event_time) * 1.e3, linewidth=2)
    cb = plt.colorbar(cont, ax=plt.gca(), ticks=np.logspace(2, 4, 3))
    cb.set_label(r"$T_\mathrm{rad}\,[\si{\electronvolt}]$")
    plt.show()

def quicktimetrace(shot1, t1, t2, show_I_p=False, show_z_mag=False):
    IDA_ed, time1, plasma_dict1 = load_IDA_data(shot1, np.linspace(t1, t2, 3000))
    time1_new = []
    ne1_new = []
    Te1_new = []
    smoothing = 20
    i1 = 0
    i2 = i1 + smoothing
    while(i2 < len(time1) and not i1 == i2):
        time1_new.append(np.mean(time1[i1:i2]))
        ne1_new.append(np.mean(plasma_dict1["ne"][i1:i2], axis=0))
        Te1_new.append(np.mean(plasma_dict1["Te"][i1:i2], axis=0))
        i1 = i2
        if(i1 + smoothing < len(time1)):
            i2 = i1 + smoothing
        else:
            i2 = len(time1) - 1
    ne1_new = np.array(ne1_new)
    Te1_new = np.array(Te1_new)
    heating = get_shot_heating(shot1)
    ECRH_Time = heating[0][0]
    ECRH_trace = heating[0][1]
    NBI_Time = heating[1][0]
    NBI_trace = heating[1][1]
    print("Plotting")
    fig = plt.figure(figsize=(12.0, 12.0))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
    ax1.plot(time1_new, Te1_new.T[0] / 1.e3)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax1.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel(r"$T_\mathrm{e}$ [\si{\kilo\electronvolt}]")
    ax2.plot(time1_new, ne1_new.T[0] / 1.e19, label=r"$n_\mathrm{e}$")
    ax2.set_ylabel(r"$n_\mathrm{e}$ [\SI{1.e19}{\per\cubic\metre}]")
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylim(0.0, np.round(np.max(ne1_new.T[0]) / 1.e19))
    ax2.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax2.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    if(show_I_p):
        time_Ip, I_p = get_plasma_current(shot1)
        ax2b = ax2.twinx()
        ax2b.plot(time_Ip[::10][np.logical_and(time_Ip[::10] > t1, time_Ip[::10] < t2)], \
                  I_p[::10][np.logical_and(time_Ip[::10] > t1, time_Ip[::10] < t2)], "--r", label=r"$I_\mathrm{p}$")
        lns = ax2.get_lines() + ax2b.get_lines()
        labs = [l.get_label() for l in lns]
        leg = ax2b.legend(lns, labs)
        leg.get_frame().set_alpha(0.5)
        leg.draggable()
        ax2b.set_ylabel(r"$I_\mathrm{p}$ [\si{\mega\ampere}]")
        ax2b.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
        ax2b.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    if(show_z_mag):
        ax2b = ax2.twinx()
        time_z_mag, z_mag = get_z_mag(shot1)
        ax2b.plot(time_z_mag[::10][np.logical_and(time_z_mag[::10] > t1, time_z_mag[::10] < t2)], \
                  z_mag[::10][np.logical_and(time_z_mag[::10] > t1, time_z_mag[::10] < t2)] * 1.e2, "--r", label=r"$z_\mathrm{mag}$")
        lns = ax2.get_lines() + ax2b.get_lines()
        labs = [l.get_label() for l in lns]
        leg = ax2b.legend(lns, labs)
        leg.get_frame().set_alpha(0.5)
        leg.draggable()
        ax2b.set_ylabel(r"$z_\mathrm{mag}$ [\si{\centi\metre}]")
        ax2b.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
        ax2b.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax3.plot(ECRH_Time[::10][np.logical_and(ECRH_Time[::10] > t1, ECRH_Time[::10] < t2)], ECRH_trace[::10][np.logical_and(ECRH_Time[::10] > t1, ECRH_Time[::10] < t2)], label="ECRH")
    ax3.plot(NBI_Time[np.logical_and(NBI_Time > t1, NBI_Time < t2)], NBI_trace[np.logical_and(NBI_Time > t1, NBI_Time < t2)], label="NBI")
    leg = ax3.legend()
    leg.draggable()
    ax3.get_yaxis().set_major_locator(MaxNLocator(nbins=3))
    ax3.get_yaxis().set_minor_locator(MaxNLocator(nbins=3))
    ax3.set_ylabel(r"$P_\mathrm{NBI/ECRH}$ [\si{\mega\watt}]")
    ax3.set_xlabel(r"$t$ [\si{\second}]")
    ax1.set_xlim(np.min(time1_new), np.max(time1_new))
    plt.suptitle("\# {0:d}".format(shot1))
    plt.tight_layout(h_pad=0.2, rect=[0.00, 0.00, 1.0, 1.0])
    plt.show()

def Te_B_plot(shot, time, IDA_exp, IDA_ed, EQ_exp, EQ_diag, EQ_ed):
    IDA_ed, time, IDA_dict = load_IDA_data(shot, [time], exp=IDA_exp, ed=IDA_ed)
    Te_spl = InterpolatedUnivariateSpline(IDA_dict["rhop"], IDA_dict["Te"])
    time = time[0]
    print("IDA time", time)
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    EQSlice = EQ_obj.GetSlice(time)
    Bt_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, np.abs(EQSlice.Bt))
    rhop_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, EQSlice.rhop)
    R = EQSlice.R
    R = R[R > 1.0]
    R = R[R < 2.3]
    R_ax, z_ax = EQ_obj.get_axis(time)
    z_0 = np.zeros(len(R))
    z_0[:] = z_ax
    Bt = Bt_spl(R, z_0, grid=False)
    rhop = rhop_spl(R, z_0, grid=False)
    sep_pos_spl = InterpolatedUnivariateSpline(R, rhop - 1.0)
    R_sep = sep_pos_spl.roots()
    R_sep_HFS = R_sep[0]
    R_sep_LFS = R_sep[1]
    print([R_sep_HFS, R_ax, R_sep_LFS])
    fig = plt.figure(figsize=[8.5, 8.5])
    ax = fig.add_subplot(111)
    ax.plot(R, Bt, "k")
    ax_Te = ax.twinx()
    ax_Te.plot(R, Te_spl(rhop) * 1.e-3, "--b")
    ax.set_xticks(np.array([R_sep_HFS, R_ax, R_sep_LFS]))
    ax.set_xticklabels([r"plasma edge HFS", r"$R_0$", r"plasma edge LFS"])
    ax.text(R_ax - 0.3, 0.9 * np.max(Bt), "HFS")
    ax.text(R_ax + 0.15, 0.9 * np.max(Bt), "LFS")
    ax.set_yticks([], [])
    ax_Te.set_yticks([], [])
    ax.set_xlabel(r"$R$ [\si{\metre}]")
    ax.set_ylabel(r"$\vert B\vert$ [\si{\tesla}]")
    ax_Te.set_ylabel(r"$T_\mathrm{e}$ [\si{\kilo\electronvolt}]", color="blue")
    ax.vlines(R_ax, np.min(Bt), np.max(Bt), linestyle=":")
    plt.show()


def profile_stiffness(shots, times, IDA_eds, IDA_exps="AUGD", EQ_exps="AUGD", EQ_diags="EQH", EQ_eds=9):
    rho_min = 0.025
    rhop_max = 0.99
    fig_grad = plt.figure()
    fig_Te = plt.figure()
    ax_grad = fig_grad.add_subplot(111)
    ax_Te = fig_Te.add_subplot(111)
    for shot, time, IDA_ed, IDA_exp, EQ_exp, EQ_diag, EQ_ed in zip(shots, times, IDA_eds, IDA_exps, EQ_exps, EQ_diags, EQ_eds):
        IDA_ed, time, plasma_dict = load_IDA_data(shot, [time], exp=IDA_exp, ed=IDA_ed)
        EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
        Te = plasma_dict["Te"][0][plasma_dict["rhop"][0] < rhop_max] * 1.e-3
        rhop = plasma_dict["rhop"][0][plasma_dict["rhop"][0] < rhop_max]
        Te = Te[rhop > rho_min]
        rhop = rhop[rhop > rho_min]
        R_av = EQ_obj.get_mean_r(time[0], rhop)
#        plt.plot(rhop, R_av)
#        plt.show()
        Te_spl = InterpolatedUnivariateSpline(R_av, Te)
        ax_grad.plot(R_av, Te_spl(R_av, nu=1) / Te, label=r"\#{0:d}".format(shot))
        ax_Te.plot(R_av, Te, label=r"\#{0:d}".format(shot))
    ax_grad.set_xlabel(r"$\overline{r}\,[\si{\metre}]$")
    ax_grad.set_ylabel(r"$\frac{\nabla T_\mathrm{e}}{T_\mathrm{e}}$")
    ax_grad.legend()
    ax_Te.set_xlabel(r"$\overline{r}\,[\si{\metre}]$")
    ax_Te.set_ylabel(r"$T_\mathrm{e}\,[\si{\kilo\electronvolt}]$")
    ax_Te.legend()
    plt.show()
        # Trad[i] = np.mean(ECE_data[1][i][ECE_ch_no1 - 1 : ECE_ch_no2 - 1])

if(__name__ == "__main__"):
    pass
    B_plot("/tokp/work/sdenk/ECRad/", 35322, 2.1684, None, 18, "Nch018_X.dat", "X")
#     CTA = ECRH_diag("CTA", "AUGD", "CTA", 0, 6, 1.0, True, t_smooth=1.e-1)
#     t, s = get_diag_data_no_calib(CTA, 33705, preview=False, single_channel=25)
#     s = s[t > 4.85]
#     t = t[t > 4.85]
#     s = s[t < 4.95]
#     t = t[t < 4.95]
#     remove_mode(t, s, 2)
#    profile_stiffness([33697, 34663, 33000], [4.80, 3.60, 6.01], [27, 8 , 0], IDA_exps=["SDENK", "SDENK", "AUGD"], EQ_exps=["AUGD", "AUGD", "AUGD"], EQ_diags=["IDE", "IDE", "EQH"], EQ_eds=[0, 0, 0])
#    Te_B_plot(33705, 4.9, "SDENK", 6, "AUGD", "IDE", 0)
#    CoreTrace(34661, 34663, 2.70, 6.25)
#    compare_power_deops("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/heatingcomp_te_plflx.ps.sav")
#    ECE_time_trace_2D(33134, [3.1822, 3.1830], 3.18252, True)  # [3.8432, 3.8433], 3.84324
#    ECE_time_trace_2D(34276, [2.4168, 2.41698], 2.41688, True, [8], 25, Trad_min=1.e1, Trad_max=4.e3)  # [3.8432, 3.8433], 3.84324
#    ECE_time_trace_2D(34401, [6.6877, 6.6878], 6.68775)  # [3.8432, 3.8433], 3.84324
#    quicktimetrace(35141, 0.5, 9, False, True)
#    ECE_time_trace_2D(33705, [3.58, 3.62], 0.0)  # [3.8432, 3.8433], 3.84324
#    quicktimetrace(33697, 0.5, 9)
#    ECE_time_trace_2D(33134, [3.8432, 3.8433], 3.84324)
#    EC  E_time_trace_2D(33134, [7.01363, 7.01365], 7.01364)
#    quicktimetrace(34663, 0.25, 9.0)
    # TE_angle_CTA(34663)
    # TE_angle_CTA(33033)
#    quick_ECE_trace(32740, [20, 41], 1)
