'''
Created on Dec 10, 2015
@author: sdenk
'''
import sys
from GlobalSettings import itm
if(not itm):
    sys.path.append('/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib')
else:
    sys.path.append('../lib')
import dd
import os
import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
libECRH = np.ctypeslib.load_library("libaug_ecrh_setmirrors", '/afs/ipp-garching.mpg.de/home/e/ecrh/sys/amd64_sles11/')
import datetime
# NOTE: This routine is not compatible with ECRH 3, yet.
# Contact S. Denk for a new version.
gy_pos_x = np.zeros(8, np.double)
gy_pos_y = np.zeros(8, np.double)
gy_sect = np.zeros(8, np.double)
gy_pos_z = np.zeros(8, np.double)
gy_curv_y_105 = np.zeros(8)
gy_curv_z_105 = np.zeros(8)
gy_curv_y_140 = np.zeros(8)
gy_curv_z_140 = np.zeros(8)
gy_width_y_105 = np.zeros(8)
gy_width_y_140 = np.zeros(8)
gy_width_z_105 = np.zeros(8)
gy_width_z_140 = np.zeros(8)
gy_pos_x[0] = 2.380
gy_pos_x[1] = 2.380
gy_pos_x[2] = 2.311
gy_pos_x[3] = 2.311
gy_pos_x[4] = 2.361
gy_pos_x[5] = 2.361
gy_pos_x[6] = 2.361
gy_pos_x[7] = 2.361
gy_pos_y[0] = 0.
gy_pos_y[1] = 0.
gy_pos_y[2] = -0.075
gy_pos_y[3] = 0.075
gy_pos_y[4] = -0.115
gy_pos_y[5] = 0.115
gy_pos_y[6] = 0.115
gy_pos_y[7] = -0.115
gy_sect[0] = 7.0
gy_sect[1] = 7.0
gy_sect[2] = 7.0
gy_sect[3] = 7.0
gy_sect[4] = 4.5
gy_sect[5] = 4.5
gy_sect[6] = 4.5
gy_sect[7] = 4.5
gy_pos_z[0] = 0.0
gy_pos_z[1] = 0.0
gy_pos_z[2] = 0.0
gy_pos_z[3] = 0.0
gy_pos_z[4] = 0.32025
gy_pos_z[5] = 0.32025
gy_pos_z[6] = -0.32025
gy_pos_z[7] = -0.32025
gy_curv_y_105[0] = 0.8793
gy_curv_y_105[1] = 0.8793
gy_curv_y_105[2] = 2.9664
gy_curv_y_105[3] = 2.9664
gy_curv_y_105[4:8] = 1.158
gy_curv_y_140[0] = 0.8793
gy_curv_y_140[1] = 0.8793
gy_curv_y_140[2] = 2.9664
gy_curv_y_140[3] = 2.9664
gy_curv_y_140[4:8] = 0.8551
gy_curv_z_105[0] = 0.8793
gy_curv_z_105[1] = 0.8793
gy_curv_z_105[2] = 2.9664
gy_curv_z_105[3] = 2.9664
gy_curv_z_105[4:8] = 1.158
gy_curv_z_140[0] = 0.8793
gy_curv_z_140[1] = 0.8793
gy_curv_z_140[2] = 2.9664
gy_curv_z_140[3] = 2.9664
gy_curv_z_140[4:8] = 0.8551
gy_width_y_105[0] = 0.0364
gy_width_y_105[1] = 0.0364
gy_width_y_105[2] = 0.0329
gy_width_y_105[3] = 0.0329
gy_width_y_105[4:8] = 0.0301
gy_width_y_140[0] = 0.0364
gy_width_y_140[1] = 0.0364
gy_width_y_140[2] = 0.0329
gy_width_y_140[3] = 0.0329
gy_width_y_140[4:8] = 0.0255
gy_width_z_105[0] = 0.0364
gy_width_z_105[1] = 0.0364
gy_width_z_105[2] = 0.0329
gy_width_z_105[3] = 0.0329
gy_width_z_105[4:8] = 0.0301
gy_width_z_140[0] = 0.0364
gy_width_z_140[1] = 0.0364
gy_width_z_140[2] = 0.0329
gy_width_z_140[3] = 0.0329
gy_width_z_140[4:8] = 0.0255
gy_name = np.array(['ECRH1_1', 'ECRH1_2', 'ECRH1_3', 'ECRH1_4', \
                    'ECRH2_1', 'ECRH2_2', 'ECRH2_3', 'ECRH2_4'])


class libECRH_wrapper:
    def __init__(self, shot):
        shot_folder = '/afs/ipp-garching.mpg.de/augd/shots/'
        path = shot_folder + '%d/' + 'L0/' + 'MAI' + '/%d'
        path = path % (shot / 10, shot)
        date_obj = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        self.date = ct.c_double(date_obj.date().year * 1.e4 + date_obj.date().month * 1.e2 + date_obj.date().day + 0.0)

    def setval2tp(self, gynum, a, beta):
        ct_gy = ct.c_int32(gynum)
        c_error = ct.c_int32(0)
        c_error.value = 0
        ct_theta_pol = ct.c_double(a)
        ct_phi_tor = ct.c_double(beta)
        # print(a, beta)
        error = 0
        libECRH.setval2tp_(ct.byref(c_error), ct.byref(ct_gy), ct.byref(ct_theta_pol), ct.byref(ct_phi_tor), ct.byref(self.date))
        if(c_error.value != 0):
            print("Encountered error ", c_error.value)
            if(np.abs(c_error.value) == 102 or np.abs(c_error.value) == 2):
                # print(a, beta)
                error = -1
        theta_pol = ct_theta_pol.value
        phi_tor = ct_phi_tor.value  # note - phi tor changes during poloidal sweep
        # print(theta_pol, phi_tor)
        return error, theta_pol, phi_tor

    def tp2setval(self, gynum, theta, phi):
        ct_gy = ct.c_int32(gynum)
        c_error = ct.c_int32(0)
        c_error.value = 0
        ct_a_pol = ct.c_double(theta)
        ct_beta_tor = ct.c_double(phi)
        # print(a, beta)
        error = 0
        libECRH.tp2setval_(ct.byref(c_error), ct.byref(ct_gy), ct.byref(ct_a_pol), ct.byref(ct_beta_tor), ct.byref(self.date))
        if(c_error.value != 0):
            print("Encountered error ", c_error.value)
            if(np.abs(c_error.value) == 102 or np.abs(c_error.value) == 2):
                # print(a, beta)
                error = -1
        a = ct_a_pol.value
        beta = ct_beta_tor.value  # note - phi tor changes during poloidal sweep
        # print(theta_pol, phi_tor)
        return error, a, beta

class AUGLauncher:
    def __init__(self, shot, N, view=False, view_140=True):  # N is the gyrotron number from 1-8
        # All units are SI units (i.e. m, W, Hz)
        # Angles are given in degrees
        self.x = gy_pos_x[N - 1]
        self.y = gy_pos_y[N - 1]
        self.z = gy_pos_z[N - 1]
        sect = gy_sect[N - 1]
        self.name = gy_name[N - 1]
        self.R = np.sqrt(self.x ** 2 + self.y ** 2)
        self.phi = (np.arctan2(self.y, self.x) + sect / 8.0 * np.pi) / np.pi * 180.e0
        self.x = np.cos(np.deg2rad(self.phi)) * self.R
        self.y = np.sin(np.deg2rad(self.phi)) * self.R
        self.error = 0
        self.curv_y = gy_curv_y_140[N - 1]  # This we can only correct if we know the frequency
        self.curv_z = gy_curv_z_140[N - 1]
        self.width_y = gy_width_z_140[N - 1]
        self.width_z = gy_width_z_140[N - 1]
        if(view and not view_140):
            self.curv_y = gy_curv_y_105[N - 1]
            self.curv_z = gy_curv_z_105[N - 1]
            self.width_y = gy_width_z_105[N - 1]
            self.width_z = gy_width_z_105[N - 1]
        self.theta_pol = 0.0  # Will be overwritten later
        self.phi_tor = 0.0  # Will be overwritten later
        self.f = 140.e9
        if(view):
            self.avail = True
        else:
            self.avail = False

class gyrotron(AUGLauncher):
    def __init__(self, shot, N, ECS, ECN, view=False, view_140=True):  # N is the gyrotron number from 1-8
        AUGLauncher.__init__(self, shot, N, view=view, view_140=view_140)
        # All units are SI units (i.e. m, W, Hz)
        # Angles are given in degrees
        libECRH_obj = libECRH_wrapper(shot)
        self.error = 0
        self.N =N
        if(N > 4):
            N_2 = N - 4
        else:
            N_2 = N
        # print(N, 'P_sy{0:n}'.format(int(N / 5.0) + 1) + '_g{0:n}'.format(N_2))
        if(N <= 4):
            if(shot < 33725):
                self.f = ECS.getParameter('P_sy{0:d}'.format(int(N / 5.0) + 1) + '_g{0:d}'.format(N_2), 'gyr_freq').data
            else:
                self.f = ECS.getParameter('P_sy3' + '_g{0:d}'.format(N_2), 'gyr_freq').data
        else:
            self.f = ECS.getParameter('P_sy{0:d}'.format(int(N / 5.0) + 1) + '_g{0:d}'.format(N_2), 'gyr_freq').data
        if(np.abs(self.f - 1.40e11) < 5.e9 or (view and view_140)):
            self.curv_y = gy_curv_y_140[N - 1]
            self.curv_z = gy_curv_z_140[N - 1]
            self.width_y = gy_width_z_140[N - 1]
            self.width_z = gy_width_z_140[N - 1]
        elif(np.abs(self.f - 1.05e11) < 5.e9 or (view and not view_140)):
            self.curv_y = gy_curv_y_105[N - 1]
            self.curv_z = gy_curv_z_105[N - 1]
            self.width_y = gy_width_z_105[N - 1]
            self.width_z = gy_width_z_105[N - 1]
        elif(self.f < 5.e9 and not view):
            self.avail = False
            print("Gyrotron " + self.name + " not available")
            self.error = -1
            return
        elif(not view):
            print("Found gyrotron with f = {0:1.3e}".format(self.f) , ", which is currently not supported")
            self.error = -1
            return
        if(self.curv_y == 0.e0):
                print("Zero encountered in curvature")
                print("Error!: Gyrotron data not properly read")
        self.avail = True
        if(N <= 4):
            if(shot < 33725):
                self.a = ECS.getParameter('P_sy{0:d}'.format(int(N / 5.0) + 1) + '_g{0:d}'.format(N_2), 'GPolPos').data * 1000.0
                self.beta = ECS.getParameter('P_sy{0:d}'.format(int(N / 5.0) + 1) + '_g{0:d}'.format(N_2), 'GTorPos').data
            else:
                self.a = ECS.getParameter('P_sy{0:d}'.format(3) + '_g{0:d}'.format(N_2), 'GPolPos').data
                self.beta = ECS.getParameter('P_sy{0:d}'.format(3) + '_g{0:d}'.format(N_2), 'GTorPos').data
        else:
            self.a = ECS.getParameter('P_sy{0:d}'.format(int(N / 5.0) + 1) + '_g{0:d}'.format(N_2), 'GPolPos').data * 1000.0
            self.beta = ECS.getParameter('P_sy{0:d}'.format(int(N / 5.0) + 1) + '_g{0:d}'.format(N_2), 'GTorPos').data
        if(N > 4):
            try:
                SUCOMDAT = ECN.getSignal('D-SIMPRE')  # ECN D-SIMPRE alternative source
            except Exception as e:
                SUCOMDAT = ECN.getSignal('SUCOMDAT')  # ECN D-SIMPRE alternative source
                print("Failed to load D-SIMPRE", e)
                print("Falling back to SUCOMDAT")
            if(N == 5):
                beta = SUCOMDAT[26] / 10.0
                a = SUCOMDAT[27] / 10.0
            elif(N == 6):
                beta = SUCOMDAT[71] / 10.0
                a = SUCOMDAT[72] / 10.0
            elif(N == 7):
                beta = SUCOMDAT[116] / 10.0
                a = SUCOMDAT[117] / 10.0
            elif(N == 8):
                beta = SUCOMDAT[161] / 10.0
                a = SUCOMDAT[162] / 10.0
            a_ECS = ECS.getParameter('P_sy{0:n}'.format(int(N / 5.0) + 1) + '_g{0:n}'.format(N_2), 'GPolPos').data * 1000.0
            beta_ECS = ECS.getParameter('P_sy{0:n}'.format(int(N / 5.0) + 1) + '_g{0:n}'.format(N_2), 'GTorPos').data
            if(np.abs(a - a_ECS) > 0.1):
                print("WARNING ECS and SUCOMDAT do not hold same spindle position")
                print("ECS, SUCOMDAT", a, ",", a_ECS)
                raise ValueError
            if(np.abs(beta - beta_ECS) > 0.1):
                print("WARNING ECS and SUCOMDAT do not hold same toroidal launching angle")
                print("ECS, SUCOMDAT", a, ",", a_ECS)
                raise ValueError
        # print("a and beta", a, beta)
        if(N <= 4):
            gy = 100 + N
        else:
            gy = 200 + N - 4
        if(N < 5):
            self.time = ECS.getTimeBase('T-B')
            self.PW = ECS.getSignal("PG{0:d}".format(N_2), dtype=np.double)
            if(shot < 33725):
                self.error, self.theta_pol, self.phi_tor = libECRH_obj.setval2tp(gy, self.a, self.beta)
            else:
                self.theta_pol = self.a
                self.phi_tor = self.beta
        else:
            time = ECS.getTimeBase('T-B')
            self.time = ECN.getTimeBase('T-Base')
            self.a_t = ECN.getSignalCalibrated("G{0:n}POL".format(N_2))[0] * 10.0
            # self.a_t = self.a_t - self.a_t[0] + a  # Treat last a_t as offset and replace it with a from SUCOMDAT
            a_t_0 = self.a_t[0]
            self.a_t = self.a_t - self.a_t[0] + a_ECS  # Treat last a_t as offset and replace it with a from ECS
            print("Offset correction:", (a_ECS - a_t_0))
            # beta_t = ECN.getSignalCalibrated("G{0:n}TOR".format(N_2))[0]  # should not change
            # self.time = DDS.getTimeBase('ProtTime')
            # a_t = DDS.getSignal("XECRH{0:n}G{1:n}".format(N / 2, N_2)) * 1000.0
            # According to M. Reich this is most likely not necessary, but it shouldn't be wrong either
            # According to M. Schubert the offset might also drift. Hence, it might be a good idea to get the value at the beginning and at the end of the
            # discharge
            self.beta_t = np.zeros(len(self.a_t))
            # Changes is beta during discharge not supported
            self.beta_t[:] = beta
            # Treat inital error as offset beta_t - beta_t[0] +
            PW = ECS.getSignal("PG{0:d}N".format(N_2), dtype=np.double)
            pw_spline = InterpolatedUnivariateSpline(time, PW)
            self.PW = pw_spline(self.time)
            if(view):
                self.avail = True
            else:
                self.avail = np.any(self.PW > 5.e3)
            if(not self.avail):
                print("Gyrotron " + self.name + " not active")
                return
            self.theta_pol = np.zeros(len(self.time))
            self.phi_tor = np.zeros(len(self.time))
            for i in (range(len(self.time))):
                self.errpr, self.theta_pol[i], self.phi_tor[i] = libECRH_obj.setval2tp(gy, self.a_t[i], self.beta_t[i])
            # plt.plot(self.time, self.theta_pol)

def load_all_ECRH(shot, output=False):
    ECS = dd.shotfile("ECS", int(shot), experiment="AUGD")
    ECN = dd.shotfile("ECN", int(shot), experiment="AUGD")
    gy_list = []
    for N in range(1, 9):
        gy_list.append(gyrotron(shot, N, ECS, ECN, False))
#        if(gy_list[-1].avail):
#            print(np.mean(gy_list[-1].PW))
#            print(np.mean(gy_list[-1].theta_pol))
#            print(np.mean(gy_list[-1].phi_tor))
    return gy_list

def load_all_active_ECRH(shot, time=None, P_active=1.e3, delta=1.e-3):
    avt = delta + delta * 0.02  # add a little bit extra to remove extreme cases
    num_beams = 0
    active_gys = []
    gy_list = load_all_ECRH(shot)
    for gy in gy_list:
        if(gy.avail):
            if(time is not None):
                t1 = np.argmin(np.abs(gy.time - (time - avt)))
                t2 = np.argmin(np.abs(gy.time - (time + avt)))
                if(t1 == t2):
                    t2 += 1
                P_ECRH = np.mean(gy.PW[t1: t2])
            else:
                P_ECRH = np.max(gy.PW)
            if(P_ECRH > P_active):
                num_beams += 1
                active_gys.append(gy)
    return active_gys

def identify_ECRH_on_phase(shot, time, av_tim=1.e-3, level=5.e3):  # 1 kW trigger level
    gy_list = load_all_ECRH(shot)
    P_ECRH = np.zeros(len(time))
    avt = av_tim / 2.0 + av_tim * 0.01  # add a little bit extra to remove extreme cases
    for gy in gy_list:
        if(gy.avail):
            for i in range(len(time)):
                t = time[i]
                P_ECRH[i] += np.mean(gy.PW[np.argmin(np.abs(gy.time - (t - avt))):np.argmin(np.abs(gy.time - (t + avt)))])
    return np.where(P_ECRH < level)[0]


def get_ECRH_viewing_angles(shot, LOS_no, view_freq_140):
    ECS = dd.shotfile("ECS", int(shot), experiment="AUGD")
    # DDS = dd.shotfile("DDS", int(shot), experiment="AUGD")
    ECN = dd.shotfile("ECN", int(shot), experiment="AUGD")
    return gyrotron(shot, LOS_no, ECS, ECN, True, view_freq_140)

def get_ECRH_launcher(shot, LOS_no, view_freq_140):
    return AUGLauncher(shot, LOS_no, True, view_freq_140)
# get_ECRH_viewing_angles(33117, 6)

def get_discharge_config(shot, time, OECE_list):
    gys = load_all_active_ECRH(shot, time=time, P_active=1.e3, delta=1.e-3)
    for gy in gys:
        print(gy.name)
        itime = np.argmin(np.abs(gy.time - time))
        print(gy.PW[itime]) / 1.e6
        if(np.isscalar(gy.theta_pol)):
            print("{0:2.1f} {1:2.1f}".format(gy.theta_pol, gy.phi_tor))
        else:
            theta_pol = gy.theta_pol[itime]
            phi_tor = gy.phi_tor[itime]
            print("{0:2.1f} {1:2.1f}".format(theta_pol, phi_tor))
    for OECE in OECE_list:
        if(shot >= 33723):
            if(OECE == "CTA"):
                OECE_launch_num = 7
            elif(OECE == "CTC"):
                OECE_launch_num = 8
            else:
                print("Unknown oblique ECE: " + OECE)
                raise ValueError
        else:
            if(OECE == "CTA"):
                OECE_launch_num = 6
            elif(OECE == "CTC"):
                OECE_launch_num = 5
            else:
                print("Unknown oblique ECE: " + OECE)
                raise ValueError
        gy = get_ECRH_viewing_angles(shot, OECE_launch_num, True)
        print(gy.name)
        if(np.isscalar(gy.theta_pol)):
            print("{0:2.1f} {1:2.1f}".format(gy.theta_pol, gy.phi_tor))
        else:
            theta_pol = gy.theta_pol[np.argmin(np.abs(gy.time - time))]
            phi_tor = gy.phi_tor[np.argmin(np.abs(gy.time - time))]
            print("{0:2.1f} {1:2.1f}".format(theta_pol, phi_tor))

if(__name__ == "__main__"):
    shot = 33697
    time = 4.8
    get_discharge_config(shot, time, ["CTA"])
