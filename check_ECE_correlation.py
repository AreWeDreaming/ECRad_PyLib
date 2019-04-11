'''
Created on Apr 20, 2016

@author: sdenk
'''
from shotfile_handling import get_diag_data_no_calib, get_data_calib_entire_shot, filter_CTA, get_ECRH_PW, load_IDA_data, get_data_calib
import numpy as np
from plotting_configuration import *
from scipy.signal import medfilt
from Diags import Diag

def correlate_s_ECE_ECRH(shot, diag1, ch1, t_match=2.4):
    t1, data1 = get_diag_data_no_calib(diag1, shot)
    t_ECRH, ECRH_PW = get_ECRH_PW(shot, 'ECS', 'AUGD', 0)
    data1[ch1] -= np.mean(data1[ch1][0:np.argmin(np.abs(t1 - 0.005))])
    data1[ch1] = data1[ch1] / np.mean(data1[ch1][np.argmin(np.abs(t1 - (t_match - 0.005))):np.argmin(np.abs(t1 - (t_match + 0.005)))]) * \
                 np.mean(ECRH_PW[np.argmin(np.abs(t_ECRH - (t_match - 0.005))) : np.argmin(np.abs(t_ECRH - (t_match + 0.005)))])
    plt.plot(t1, data1[ch1], '-')
    plt.plot(t_ECRH, ECRH_PW, '--')
    plt.show()


def check_ECE_correlation(shot, t_calib, ECE_diag, ch_ECE, diag1, ch1, diag2=None, ch2=None, calib=None, Trad=None):
    # diag 1 calibrated diag 2 not
    # get_diag_data_no_calib(diag1, shot)
    if(not Trad):
        ECE_t, ECE_data = get_data_calib_entire_shot(ECE_diag, shot)
    else:
        Trad_calib = Trad
    t1, data1 = get_diag_data_no_calib(diag1, shot)
    data1[ch1] -= np.mean(data1[ch1][0:np.argmin(np.abs(t1 - 0.005))])
    if(diag2 is not None):
        t2, data2 = get_diag_data_no_calib(diag2, shot)
        data2[ch2] -= np.mean(data2[ch2][0:np.argmin(np.abs(t2 - 0.005))])
    if(calib is None and Trad is None):
        data1[ch1] = data1[ch1] / np.mean(data1[ch1][np.argmin(np.abs(t1 - (t_calib - 0.005))):np.argmin(np.abs(t1 - (t_calib + 0.005)))]) * \
                 np.mean(ECE_data[ch_ECE][np.argmin(np.abs(ECE_t - (t_calib - 0.005))) : np.argmin(np.abs(ECE_t - (t_calib + 0.005)))])
        Trad_calib = np.mean(ECE_data[ch_ECE][np.argmin(np.abs(ECE_t - (t_calib - 0.005))) : np.argmin(np.abs(ECE_t - (t_calib + 0.005)))])
        if(diag2 is not None):
            data2[ch2] = data2[ch2] / np.mean(data2[ch2][np.argmin(np.abs(t2 - (t_calib - 0.005))):np.argmin(np.abs(t2 - (t_calib + 0.005)))]) * \
                 np.mean(ECE_data[ch_ECE][np.argmin(np.abs(ECE_t - (t_calib - 0.005))) : np.argmin(np.abs(ECE_t - (t_calib + 0.005)))])
    elif(calib is not None):
        data1[ch1] *= calib
        if(diag2 is not None):
            data2[ch2] *= calib
    else:
        data1[ch1] = data1[ch1] / np.mean(data1[ch1][np.argmin(np.abs(t1 - (t_calib - 0.005))):np.argmin(np.abs(t1 - (t_calib + 0.005)))]) * \
                 Trad
        if(diag2 is not None):
            data2[ch2] = data2[ch2] / np.mean(data2[ch2][np.argmin(np.abs(t2 - (t_calib - 0.005))):np.argmin(np.abs(t2 - (t_calib + 0.005)))]) * \
                     Trad
    print(np.shape(t1), np.shape(data1))
    if(diag2 is not None):
        print(np.shape(t2), np.shape(data2))
    label_ECE = "1d ECE"  # "1d ECE"
    if(diag1.name == "CTA" or diag1.name == "CTC"):
        label1 = "s-ECE Danish"  # "1d ECE"
    elif(diag1.name == "IEC"):
        label1 = "s-ECE Dutch"
    elif(diag1.name == "ECN" or diag1.name == "ECO"):
        label1 = "ECEI"
    else:
        print("Unkown diagnostic 1")
        print(diag1.name, diag1.diag)
    if(diag2 is not None):
        if(diag2.name == "CTA" or diag2.name == "CTC"):
            label2 = "s-ECE Danish"  # "1d ECE"
        elif(diag2.name == "IEC"):
            label2 = "s-ECE Dutch"
        elif(diag2.name == "ECN" or diag2.name == "ECO"):
            label2 = "ECEI"
        else:
            print("Unkown diagnostic 1")
            print(diag2.name, diag2.diag)
    if(diag1.diag == "IEC" or diag1.diag == "CTA" or diag1.diag == "CTA"):
        if(diag1.diag == "CTA"):
            good_t1 = filter_CTA(shot, t1, "CTA", diag1.exp, diag1.ed)
        elif(diag2 is not None and diag2.diag == "CTA"):
            good_t1 = filter_CTA(shot, t1, "CTA", diag2.exp, diag2.ed)
        else:
            good_t1 = filter_CTA(shot, t1, "CTA", "AUGD", 0)
    else:
        good_t1 = np.zeros(len(t1), dtype=np.bool)
        good_t1[:] = True
    if(diag2 is not None):
        if(diag1.diag == "IEC" or diag1.diag == "CTA" or diag1.diag == "CTA"):
            if(diag1.diag == "CTA"):
                good_t2 = filter_CTA(shot, t2, "CTA", diag1.exp, diag1.ed)
            elif(diag2.diag == "CTA"):
                good_t2 = filter_CTA(shot, t2, "CTA", diag2.exp, diag2.ed)
            else:
                good_t2 = filter_CTA(shot, t2, "CTA", "AUGD", 0)
        else:
            good_t2 = np.zeros(len(t2, dtype=np.bool))
            good_t2[:] = True
    if(Trad is None):
        plt.plot(ECE_t, medfilt(ECE_data[ch_ECE], 5), label=label_ECE)
    plt.plot(t1[good_t1], medfilt(data1[ch1][good_t1], 5), label=label1)
    if(diag2 is not None):
        plt.plot(t2[good_t2], medfilt(data2[ch2][good_t2], 5), label=label2)
    plt.plot(t_calib, Trad_calib, "+k", label='$t_\mathrm{Calib}$')
    plt.gca().set_xlabel("$t$ [s]")
    plt.gca().set_ylabel("$T_\mathrm{rad}$ [eV]")
    # plt.gca().set_ylabel("a. u.")
    plt.legend()
    plt.show()
ECE_diag = Diag("ECE", "AUGD", "RMD", 0)
# diag2 = Diag("IEC", "AUGD", "IEC", 0)  # , diag2, 39
diag1 = Diag("CTA", "AUGD", "CTA", 0)  # "SDENK"
# diag1 = Diag("ECN", "AUGD", "TDI", 0)  # "SDENK"
check_ECE_correlation(33317, 6.45, ECE_diag, 56, diag1, 0)  # , 147  # , 0,  # 49 -1.756e+04 #-8.799e+04 #-8.112e+04,#-6.445e+04
# correlate_s_ECE_ECRH(32965, diag1, 39, t_match=2.4)

def Trad_vs_ne(shot, time, ida_ed, ECE_ch_no1, ECE_ch_no2):
    IDA_ed, time, plasma_dict = load_IDA_data(shot, np.linspace(time[0], time[1], 3000), ed=ida_ed)
    ECE_diag = Diag("ECE", "AUGD", "CEC", 0)
    std_dev, ECE_data = get_data_calib(diag=ECE_diag, shot=shot, time=time, t_smooth=1.e-3)
    ne = np.zeros(time.shape)
    Trad = np.zeros(time.shape)
    bins = 50
    for i in range(len(time)):
        ne[i] = np.mean(plasma_dict["ne"][i][plasma_dict["rhop"][i] < 0.2])
        Trad[i] = np.mean(ECE_data[1][i][ECE_ch_no1 - 1 : ECE_ch_no2 - 1]) / np.mean(plasma_dict["Te"][i][plasma_dict["rhop"][i] < 0.1]) * 1.e3
    Trad = np.mean(Trad.reshape((20, 150)), axis=1)
    ne = np.mean(ne.reshape((20, 150)), axis=1)
    plt.plot(ne * 1.e-19, Trad)

    # plt.plot(time, ne * 1.e-19)
    # plt.plot(time, Trad)
    plt.show()

# Trad_vs_ne(33705, [3.2, 5], 1, 1, 9)


