'''
Created on Feb 2, 2017

@author: sdenk
'''
import numpy as np
import os

class BasicDiag:
    def __init__(self, name):
        self.name = name  # Must not be changed -> Does not get a widget
        self.properties = []
        self.descriptions_dict = {}
        self.data_types_dict = {}
        self.scale_dict = {}


class Diag(BasicDiag):
    def __init__(self, name, exp, diag_str, ed, t_smooth=1.e-3, N_ray=1, N_freq=1, \
                 waist_scale=1.0, waist_shift=0.0, mode_filter=False, mode_harmonics=1, \
                 mode_width=100.0, freq_cut_off=100.0):
        BasicDiag.__init__(self, name)
        self.exp = exp
        self.properties.append("exp")
        self.descriptions_dict["exp"] = "Experiment"
        self.data_types_dict["exp"] = "string"
        self.diag = diag_str
        self.properties.append("diag")
        self.descriptions_dict["diag"] = "Diagnostic"
        self.data_types_dict["diag"] = "string"
        self.ed = ed
        self.properties.append("ed")
        self.descriptions_dict["ed"] = "Edition"
        self.data_types_dict["ed"] = "integer"
        self.t_smooth = t_smooth
        self.properties.append("t_smooth")
        self.descriptions_dict["t_smooth"] = "time window for smoothing"
        self.data_types_dict["t_smooth"] = "real"
#        self.N_ray = N_ray
#        self.properties.append("N_ray")
#        self.descriptions_dict["N_ray"] = "Number of rays"
#        self.N_freq = N_freq
#        self.properties.append("N_freq")
#        self.descriptions_dict["N_freq"] = "Number of frequencies"
#        self.waist_scale = waist_scale
#        self.properties.append("waist_scale")
#        self.descriptions_dict["waist_scale"] = "Scale beam waist"
#        self.waist_shift = waist_shift
#        self.properties.append("waist_shift")
#        self.descriptions_dict["waist_shift"] = "Shift beam waist"
        self.mode_filter = mode_filter
        self.mode_width = mode_width
        self.freq_cut_off = freq_cut_off
#        self.properties.append("mode_filter")
#        self.descriptions_dict["mode_filter"] = "Shift beam waist"
#        self.mode_harmonics = mode_harmonics
#        


class ECI_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, Rz_exp, Rz_diag, Rz_ed, t_smooth=1.e-3):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        self.Rz_exp = Rz_exp
        self.properties.append("Rz_exp")
        self.descriptions_dict["Rz_exp"] = "Experiment for launch shot file"
        self.data_types_dict["Rz_exp"] = "string"
        self.Rz_diag = Rz_diag
        self.properties.append("Rz_diag")
        self.descriptions_dict["Rz_diag"] = "Diagnostic for launch shot file"
        self.data_types_dict["Rz_diag"] = "string"
        self.Rz_ed = Rz_ed
        self.properties.append("Rz_ed")
        self.descriptions_dict["Rz_ed"] = "Edition for launch shot file"
        self.data_types_dict["Rz_ed"] = "integer"


class ECRH_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, beamline, pol_coeff_X, base_freq_140, t_smooth=1.e-3):
        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
        self.beamline = beamline
        self.properties.append("beamline")
        self.descriptions_dict["beamline"] = "Used beamline"
        self.data_types_dict["beamline"] = "integer"
        self.pol_coeff_X = pol_coeff_X
        self.properties.append("pol_coeff_X")
        self.data_types_dict["pol_coeff_X"] = "real"
        self.descriptions_dict["pol_coeff_X"] = "Efficiency of polarizer for X-mode"
        self.base_freq_140 = base_freq_140
        self.properties.append("base_freq_140")
        self.descriptions_dict["base_freq_140"] = "140 GHz central frequency"
        self.data_types_dict["base_freq_140"] = "bool"

class EXT_diag(BasicDiag):  #  Makes no sense to inherit properties we do not want -> Own class
    def __init__(self, name, launch_geo=None, pol_coeff_X= -1, t_smooth=1.e-3):
        BasicDiag.__init__(self, name)
        # Set diag to an example diagnostic
        self.name = name
        self.properties = []
        self.descriptions_dict = {}
        self.data_types_dict = {}
        self.scale_dict = {}  # Scaled quantities get an entry here
        self.N_ch = 1
        self.properties.append("N_ch")
        self.descriptions_dict["N_ch"] = "Number of channels"
        self.data_types_dict["N_ch"] = "integer"
        self.f = np.array([140.e0])
        self.properties.append("f")
        self.descriptions_dict["f"] = "frequency [GHz]"
        self.data_types_dict["f"] = "real"
        self.scale_dict["f"] = 1.e-9  # This is the scale used for I/O
        self.df = np.array([1.e9])
        self.properties.append("df")
        self.descriptions_dict["df"] = "frequency bandwidth [GHz]"
        self.data_types_dict["df"] = "real"
        self.scale_dict["df"] = 1.e-9  # This is the scale used for I/O
        self.R = np.array([2.6])
        self.properties.append("R")
        self.descriptions_dict["R"] = "Antenna position: R [m]"
        self.data_types_dict["R"] = "real"
        self.phi = np.array([45.0])
        self.properties.append("phi")
        self.descriptions_dict["phi"] = "Antenna position: phi [deg.]"
        self.data_types_dict["phi"] = "real"
        self.z = np.array([0.0])
        self.properties.append("z")
        self.descriptions_dict["z"] = "Antenna position: z [m]"
        self.data_types_dict["z"] = "real"
        self.theta_pol = np.array([0.5])
        self.properties.append("theta_pol")
        self.descriptions_dict["theta_pol"] = "Poloidal launch angle [deg.]"
        self.data_types_dict["theta_pol"] = "real"
        self.phi_tor = np.array([2.0])
        self.properties.append("phi_tor")
        self.descriptions_dict["phi_tor"] = "Toroidal launch angle [deg.]"
        self.data_types_dict["phi_tor"] = "real"
        self.properties.append("dist_focus")
        self.dist_focus = np.array([1.0])
        self.descriptions_dict["dist_focus"] = "Distance to focus [m]"
        self.data_types_dict["dist_focus"] = "real"
        self.width = np.array([0.3])
        self.properties.append("width")
        self.descriptions_dict["width"] = "Beam width at antenna"
        self.data_types_dict["width"] = "real"
        self.pol_coeff_X = np.array([0.3])
        self.properties.append("pol_coeff_X")
        self.descriptions_dict["pol_coeff_X"] = "Polarizer efficiency for X-mode"
        self.data_types_dict["pol_coeff_X"] = "real"
        # Now overwrite if actual values is provided
        if(launch_geo is not None):
            self.set_from_launch_geo(launch_geo, pol_coeff_X, False)

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
# Deprecated
# class TCV_CCE_diag(Diag):
#    def __init__(self, name, exp, diag_str, ed, time, launch_geo, t_smooth=1.e-3):
#        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
#        # exp diag_str and ed not used
#        # launch_geo first dimension channel, second dimension time
#        self.time = time
#        self.N_ch = len(launch_geo.T[1])
#        self.f = launch_geo[0]
#        self.df = launch_geo[1]
#        self.R = launch_geo[2]
#        self.phi = launch_geo[3]
#        self.z = launch_geo[4]
#        self.theta_pol = launch_geo[5]
#        self.phi_tor = launch_geo[6]
#        self.dist_focus = launch_geo[7]
#        self.width = launch_geo[8]
#
#    def get_launch_geo(self, time):
#        launch_geo = np.zeros((9, len(self.f)))
#        i_closest = np.argmin(np.abs(time - self.time))
#        if(np.abs(self.time[i_closest] - time) > 1.e-6):
#            print("Large discrepancy between Diagnostic time and plasma time!!!")
#            raise ValueError
#        launch_geo[0] = self.f[i_closest]
#        launch_geo[1] = self.df[i_closest]
#        launch_geo[2] = self.R[i_closest]
#        launch_geo[3] = self.phi[i_closest]
#        launch_geo[4] = self.z[i_closest]
#        launch_geo[5] = self.theta_pol[i_closest]
#        launch_geo[6] = self.phi_tor[i_closest]
#        launch_geo[7] = self.dist_focus[i_closest]
#        launch_geo[8] = self.width[i_closest]
#        return launch_geo
#
# class TCV_diag(Diag):
#    def __init__(self, name, exp, diag_str, ed, R_scale, z_scale, t_smooth=1.e-3):
#        Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
#        self.R_scale = R_scale
#        self.z_scale = z_scale
