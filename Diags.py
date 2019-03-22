'''
Created on Feb 2, 2017

@author: sdenk
'''
import numpy as np
import os

class Diag:
    def __init__(self, name, exp, diag_str, ed, t_smooth=1.e-3, N_ray=1, N_freq=1, \
                 waist_scale=1.0, waist_shift=0.0, mode_filter=False, mode_harmonics=1, \
                 mode_width=100.0, freq_cut_off=100.0):
        self.name = name
        self.exp = exp
        self.diag = diag_str
        self.ed = ed
        self.t_smooth = t_smooth
        self.N_ray = N_ray
        self.N_freq = N_freq
        self.waist_scale = waist_scale
        self.waist_shift = waist_shift
        self.mode_filter = mode_filter
        self.mode_harmonics = mode_harmonics
        self.mode_width = mode_width
        self.freq_cut_off = freq_cut_off

class ECI_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, Rz_exp, Rz_diag, Rz_ed, t_smooth=1.e-3):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        self.Rz_exp = Rz_exp
        self.Rz_diag = Rz_diag
        self.Rz_ed = Rz_ed

class ECRH_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, beamline, pol_coeff_X, base_freq_140, t_smooth=1.e-3):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        self.beamline = beamline
        self.pol_coeff_X = pol_coeff_X
        self.base_freq_140 = base_freq_140

class EXT_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, launch_geo=None, pol_coeff_X= -1, t_smooth=1.e-3, append=False):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        # exp diag_str and ed not used
        if(launch_geo is not None):
            self.set_from_launch_geo(launch_geo, pol_coeff_X, append)

    def load_launch_geo_from_file(self, filename, old_ray_launch=False):
        # ray_launch refers to the old ray_launch files written by older versions of ECRad
        new_launch = np.loadtxt(filename, skiprows=1)
        launch_geo = []
        length_scale = 1.0
        if(old_ray_launch):
            length_scale = 1.e-2
        if(len(new_launch.T) < 10):
            print("Old launch file detected")
            launch_geo.append(new_launch.T[0])  # f
            launch_geo.append(new_launch.T[1])  # df
            launch_geo.append(np.zeros(len((new_launch.T[0])), dtype=np.int))  # N_freq
            launch_geo[-1][:] = 1
            launch_geo.append(np.zeros(len((new_launch.T[0])), dtype=np.int))  # N_ray
            launch_geo[-1][:] = 1
            launch_geo.append(np.zeros(len((new_launch.T[0]))))  # waist_scale
            launch_geo[-1][:] = 1.0
            launch_geo.append(np.zeros(len((new_launch.T[0]))))  # waist_shift
            launch_geo[-1][:] = 0.0
            if(old_ray_launch):
                launch_geo.append(np.sqrt(new_launch.T[2] ** 2 + new_launch.T[3] ** 2) * length_scale)  # R
                launch_geo.append(np.rad2deg(np.arctan2(new_launch.T[3], new_launch.T[2])))  # phi
            else:
                launch_geo.append(new_launch.T[2])
                launch_geo.append(new_launch.T[3])
            launch_geo.append(new_launch.T[4] * length_scale)  # z
            launch_geo.append(new_launch.T[6])  # theta_pol
            launch_geo.append(new_launch.T[5])  # phi_tor
            if(old_ray_launch):
                launch_geo.append(new_launch.T[7] * length_scale)  # dist_focus
                launch_geo.append(new_launch.T[8] * length_scale)  # width
            else:
                launch_geo.append(new_launch.T[8] * length_scale)  # dist_focus
                launch_geo.append(new_launch.T[7] * length_scale)  # width
        else:
            launch_geo.append(new_launch.T[0])  # f
            launch_geo.append(new_launch.T[1])  # df
            launch_geo.append(new_launch.T[2])  # N_freq
            launch_geo.append(new_launch.T[3])  # N_ray
            launch_geo.append(new_launch.T[4])  # waist_scale
            launch_geo.append(new_launch.T[5])  # waist_shift
            if(old_ray_launch):
                launch_geo.append(np.sqrt(new_launch.T[6] ** 2 + new_launch.T[7] ** 2) * length_scale)  # R
                launch_geo.append(np.rad2deg(np.arctan2(new_launch.T[7], new_launch.T[6])))  # phi
            else:
                launch_geo.append(new_launch.T[6])  # R
                launch_geo.append(new_launch.T[7])  # Phi
            launch_geo.append(new_launch.T[8] * length_scale)  # z
            launch_geo.append(new_launch.T[9])  # theta_pol
            launch_geo.append(new_launch.T[10])  # phi_tor
            launch_geo.append(new_launch.T[11] * length_scale)  # dist_focus
            launch_geo.append(new_launch.T[12] * length_scale)  # width
        try:
            polfilename = os.path.join(os.path.dirname(filename), \
                                       os.path.basename(filename)[0:3] + "_pol_coeff.dat")
            pol = np.loadtxt(polfilename)
        except:
            pol = -1
        self.set_from_launch_geo(launch_geo, pol)

    def get_launch_geo(self):
        launch_geo = np.zeros((13, len(self.f)))
        launch_geo[0] = self.f
        launch_geo[1] = self.df
        launch_geo[2] = self.N_freq
        launch_geo[3] = self.N_ray
        launch_geo[4] = self.waist_scale
        launch_geo[5] = self.waist_shift
        launch_geo[6] = self.R
        launch_geo[7] = self.phi
        launch_geo[8] = self.z
        launch_geo[9] = self.theta_pol
        launch_geo[10] = self.phi_tor
        launch_geo[11] = self.dist_focus
        launch_geo[12] = self.width
        return launch_geo, self.pol_coeff_X

    def set_from_launch_geo(self, launch_geo, pol_coeff_X, append=False):
        if(append):
            self.N_ch += len(launch_geo[0])
            self.f = np.concatenate([self.f, launch_geo[0]])
            self.df = np.concatenate([self.df, launch_geo[1]])
            self.N_freq = np.concatenate([self.N_freq, launch_geo[2]])
            self.N_ray = np.concatenate([self.N_ray, launch_geo[3]])
            self.waist_scale = np.concatenate([self.waist_scale, launch_geo[4]])
            self.waist_shift = np.concatenate([self.waist_shift, launch_geo[5]])
            self.R = np.concatenate([self.R, launch_geo[6]])
            self.phi = np.concatenate([self.phi, launch_geo[7]])
            self.z = np.concatenate([self.z, launch_geo[8]])
            self.theta_pol = np.concatenate([self.theta_pol, launch_geo[9]])
            self.phi_tor = np.concatenate([self.phi_tor, launch_geo[10]])
            self.dist_focus = np.concatenate([self.dist_focus, launch_geo[11]])
            self.width = np.concatenate([self.width, launch_geo[12]])
            if(np.isscalar(pol_coeff_X)):
                pol_coeff_X_append = np.zeros(self.N_ch)
                pol_coeff_X_append[:] = pol_coeff_X
                self.pol_coeff_X[:] = np.concatenate([self.f, pol_coeff_X_append])
            else:
                self.pol_coeff_X = np.concatenate([self.f, pol_coeff_X])
        else:
            if(np.isscalar(launch_geo[0])):
                self.N_ch = 1
                self.f = np.array([launch_geo[0]])
                self.df = np.array([launch_geo[1]])
                self.N_freq = np.array([launch_geo[2]])
                self.N_ray = np.array([launch_geo[3]])
                self.waist_scale = np.array([launch_geo[4]])
                self.waist_shift = np.array([launch_geo[5]])
                self.R = np.array([launch_geo[6]])
                self.phi = np.array([launch_geo[7]])
                self.z = np.array([launch_geo[8]])
                self.theta_pol = np.array([launch_geo[9]])
                self.phi_tor = np.array([launch_geo[10]])
                self.dist_focus = np.array([launch_geo[11]])
                self.width = np.array([launch_geo[12]])
            else:
                self.N_ch = len(launch_geo[0])
                self.f = launch_geo[0]
                self.df = launch_geo[1]
                self.N_freq = launch_geo[2]
                self.N_ray = launch_geo[3]
                self.waist_scale = launch_geo[4]
                self.waist_shift = launch_geo[5]
                self.R = launch_geo[6]
                self.phi = launch_geo[7]
                self.z = launch_geo[8]
                self.theta_pol = launch_geo[9]
                self.phi_tor = launch_geo[10]
                self.dist_focus = launch_geo[11]
                self.width = launch_geo[12]
            if(np.isscalar(pol_coeff_X)):
                self.pol_coeff_X = np.zeros(self.N_ch)
                self.pol_coeff_X[:] = pol_coeff_X
            else:
                self.pol_coeff_X = pol_coeff_X

class TCV_CCE_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, time, launch_geo, t_smooth=1.e-3):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        # exp diag_str and ed not used
        # launch_geo first dimension channel, second dimension time
        self.time = time
        self.N_ch = len(launch_geo.T[1])
        self.f = launch_geo[0]
        self.df = launch_geo[1]
        self.R = launch_geo[2]
        self.phi = launch_geo[3]
        self.z = launch_geo[4]
        self.theta_pol = launch_geo[5]
        self.phi_tor = launch_geo[6]
        self.dist_focus = launch_geo[7]
        self.width = launch_geo[8]

    def get_launch_geo(self, time):
        launch_geo = np.zeros((9, len(self.f)))
        i_closest = np.argmin(np.abs(time - self.time))
        if(np.abs(self.time[i_closest] - time) > 1.e-6):
            print("Large discrepancy between Diagnostic time and plasma time!!!")
            raise ValueError
        launch_geo[0] = self.f[i_closest]
        launch_geo[1] = self.df[i_closest]
        launch_geo[2] = self.R[i_closest]
        launch_geo[3] = self.phi[i_closest]
        launch_geo[4] = self.z[i_closest]
        launch_geo[5] = self.theta_pol[i_closest]
        launch_geo[6] = self.phi_tor[i_closest]
        launch_geo[7] = self.dist_focus[i_closest]
        launch_geo[8] = self.width[i_closest]
        return launch_geo

class TCV_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, R_scale, z_scale, t_smooth=1.e-3):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        self.R_scale = R_scale
        self.z_scale = z_scale
