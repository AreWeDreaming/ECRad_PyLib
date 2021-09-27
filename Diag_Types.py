'''
Created on Feb 2, 2017

@author: sdenk
'''
import numpy as np
from collections import OrderedDict as od
from scipy.io import loadmat
from netCDF4 import Dataset

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
        # self.t_smooth = t_smooth
        # self.properties.append("t_smooth")
        # self.descriptions_dict["t_smooth"] = "time window for smoothing"
        # self.data_types_dict["t_smooth"] = "real"
        # self.mode_filter = mode_filter
        # self.properties.append("mode_filter")
        # self.descriptions_dict["mode_filter"] = "Filter MHD modes"
        # self.data_types_dict["mode_filter"] = "bool"
        # self.mode_width = mode_width
        # self.properties.append("mode_width")
        # self.descriptions_dict["mode_width"] = "width of MHD mode [Hz]"
        # self.data_types_dict["mode_width"] = "real"
        # self.freq_cut_off = freq_cut_off
        # self.properties.append("freq_cut_off")
        # self.descriptions_dict["freq_cut_off"] = "low frequency cut of Fourier filter [Hz]"
        # self.data_types_dict["freq_cut_off"] = "real"
        # self.mode_harmonics = mode_harmonics
        # self.properties.append("mode_harmonics")
        # self.descriptions_dict["mode_harmonics"] = "harmonics to consider in filter"
        # self.data_types_dict["mode_harmonics"] = "integer"

class CECE_diag(Diag):
    def __init__(self, name, exp, diag_str, ed, t_smooth=1.e-3):
         Diag.__init__(self, name, exp, diag_str, ed, t_smooth)
         self.f = None
         self.df = None
         # CECE sits on wg 8
         self.wg = 8
         # distance is 55 mm
         self.dtoECESI = 0.055

    def set_f_info(self, f, df):
        self.f = f
        self.df = df

    def get_f_info(self):
        return self.f, self.df

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
    def __init__(self, name, exp, diag_str, ed, beamline, pol_coeff_X, base_freq_140, \
                 t_smooth=1.e-3, pol_angle_cor= 0.0, tor_angle_cor = 0.0):
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
        self.pol_angle_cor = pol_angle_cor
        self.properties.append("pol_angle_cor")
        self.descriptions_dict["pol_angle_cor"] = "Correction to be applied on the pol. launch angle"
        self.data_types_dict["pol_angle_cor"] = "real"
        self.tor_angle_cor = tor_angle_cor
        self.properties.append("tor_angle_cor")
        self.descriptions_dict["tor_angle_cor"] = "Correction to be applied on the tor. launch angle"
        self.data_types_dict["tor_angle_cor"] = "real"

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
        self.scale_dict["phi"] = 1.0  # This is the scale used for I/O
        self.z = np.array([0.0])
        self.properties.append("z")
        self.descriptions_dict["z"] = "Antenna position: z [m]"
        self.data_types_dict["z"] = "real"
        self.scale_dict["z"] = 1.0  # This is the scale used for I/O
        self.theta_pol = np.array([0.5])
        self.properties.append("theta_pol")
        self.descriptions_dict["theta_pol"] = "Poloidal launch angle [deg.]"
        self.data_types_dict["theta_pol"] = "real"
        self.scale_dict["theta_pol"] = 1.0  # This is the scale used for I/O
        self.phi_tor = np.array([2.0])
        self.properties.append("phi_tor")
        self.descriptions_dict["phi_tor"] = "Toroidal launch angle [deg.]"
        self.data_types_dict["phi_tor"] = "real"
        self.scale_dict["phi_tor"] = 1.0  # This is the scale used for I/O
        self.properties.append("dist_focus")
        self.dist_focus = np.array([1.0])
        self.descriptions_dict["dist_focus"] = "Distance to focus [m]"
        self.data_types_dict["dist_focus"] = "real"
        self.scale_dict["dist_focus"] = 1.0  # This is the scale used for I/O
        self.width = np.array([0.3])
        self.properties.append("width")
        self.descriptions_dict["width"] = "Beam width at antenna [m]"
        self.data_types_dict["width"] = "real"
        self.scale_dict["width"] = 1.0  # This is the scale used for I/O
        self.pol_coeff_X = np.array([0.3])
        self.properties.append("pol_coeff_X")
        self.descriptions_dict["pol_coeff_X"] = "Polarizer efficiency for X-mode"
        self.data_types_dict["pol_coeff_X"] = "real"
        self.scale_dict["pol_coeff_X"] = 1.0  # This is the scale used for I/O
        # Now overwrite if actual values is provided
        if(launch_geo is not None):
            self.set_from_launch_geo(launch_geo, pol_coeff_X, False)

    def get_launch_geo(self):
        launch_geo = np.zeros((9, len(self.f)))
        launch_geo[0] = self.f
        launch_geo[1] = self.df
        launch_geo[2] = self.R
        launch_geo[3] = self.phi
        launch_geo[4] = self.z
        launch_geo[5] = self.theta_pol
        launch_geo[6] = self.phi_tor
        launch_geo[7] = self.dist_focus
        launch_geo[8] = self.width
        return launch_geo, self.pol_coeff_X
    
    def get_launch(self):
        launch = {}
        launch["diag_name"] = np.zeros(len(self.f), dtype="|U3")
        launch["diag_name"][:] = "EXT"
        launch["f"] = np.copy(self.f)
        launch["df"] = np.copy(self.df)
        launch["R"] = np.copy(self.R)
        launch["phi"] = np.copy(self.phi)
        launch["z"] = np.copy(self.z)
        launch["theta_pol"] = np.copy(self.theta_pol)
        launch["phi_tor"] = np.copy(self.phi_tor)
        launch["dist_focus"] = np.copy(self.dist_focus)
        launch["width"] = np.copy(self.width)
        launch["pol_coeff_X"] = np.copy(self.pol_coeff_X)
        return launch
    
    def set_from_ray_launch(self, ray_launch, itime, set_only_EXT=True):
        if(set_only_EXT):
            mask = ray_launch[itime]["diag_name"] == "EXT"
        else:
            mask = np.ones(len(ray_launch[itime]["diag_name"]), dtype=np.bool)
        self.f = ray_launch[itime]["f"][mask]
        self.N_ch = len(self.f)
        self.df = ray_launch[itime]["df"][mask]
        self.R = ray_launch[itime]["R"][mask]
        self.phi = ray_launch[itime]["phi"][mask]
        self.z = ray_launch[itime]["z"][mask]
        self.theta_pol = ray_launch[itime]["theta_pol"][mask]
        self.phi_tor = ray_launch[itime]["phi_tor"][mask]
        self.dist_focus = ray_launch[itime]["dist_focus"][mask]
        self.width = ray_launch[itime]["width"][mask]
        self.pol_coeff_X = ray_launch[itime]["pol_coeff_X"][mask]
        
    def set_from_scenario_diagnostic(self, ray_launch, itime, set_only_EXT=True):
        if(set_only_EXT):
            mask = ray_launch["diag_name"][itime] == "EXT"
        else:
            mask = np.ones(len(ray_launch["diag_name"][itime]), dtype=np.bool)
        self.f = ray_launch["f"][itime][mask]
        self.N_ch = len(self.f)
        self.df = ray_launch["df"][itime][mask]
        self.R = ray_launch["R"][itime][mask]
        self.phi = ray_launch["phi"][itime][mask]
        self.z = ray_launch["z"][itime][mask]
        self.theta_pol = ray_launch["theta_pol"][itime][mask]
        self.phi_tor = ray_launch["phi_tor"][itime][mask]
        self.dist_focus = ray_launch["dist_focus"][itime][mask]
        self.width = ray_launch["width"][itime][mask]
        self.pol_coeff_X = ray_launch["pol_coeff_X"][itime][mask]
        
    def set_from_NETCDF(self, filename, itime=0):
        rootgrp = Dataset(filename, "r", format="NETCDF4")
        for sub_key in ["f", "df", "R", "phi", "z", "theta_pol", \
                        "phi_tor", "dist_focus", "width", "pol_coeff_X"]:
            setattr(self, sub_key, np.array(rootgrp["Diagnostic"]["diagnostic_" + sub_key])[itime])
        self.N_ch = len(self.f)
        
        
    def set_from_mat(self, ray_launch_file):
        mdict = loadmat(ray_launch_file, squeeze_me=True)
        if(np.ndim(mdict["launch_f"]) == 1):
            self.f = mdict["launch_f"]
            self.df = mdict["launch_df"]
            self.R = mdict["launch_R"]
            self.phi = mdict["launch_phi"]
            self.z = mdict["launch_z"]
            self.theta_pol = mdict["launch_pol_ang"]
            self.phi_tor = mdict["launch_tor_ang"]
            self.dist_focus = mdict["launch_dist_focus"]
            self.width = mdict["launch_width"]
            self.pol_coeff_X = mdict["launch_pol_coeff_X"]
        else:
            self.f = mdict["launch_f"][0]
            self.df = mdict["launch_df"][0]
            self.R = mdict["launch_R"][0]
            self.phi = mdict["launch_phi"][0]
            self.z = mdict["launch_z"][0]
            self.theta_pol = mdict["launch_pol_ang"][0]
            self.phi_tor = mdict["launch_tor_ang"][0]
            self.dist_focus = mdict["launch_dist_focus"][0]
            self.width = mdict["launch_width"][0]
            self.pol_coeff_X = mdict["launch_pol_coeff_X"][0]
        self.N_ch = len(self.f)
        

    def set_from_launch_geo(self, launch_geo, pol_coeff_X, append=False):
        if(append):
            self.N_ch += len(launch_geo[0])
            self.f = np.concatenate([self.f, launch_geo[0]])
            self.df = np.concatenate([self.df, launch_geo[1]])
            self.R = np.concatenate([self.R, launch_geo[2]])
            self.phi = np.concatenate([self.phi, launch_geo[3]])
            self.z = np.concatenate([self.z, launch_geo[4]])
            self.theta_pol = np.concatenate([self.theta_pol, launch_geo[5]])
            self.phi_tor = np.concatenate([self.phi_tor, launch_geo[6]])
            self.dist_focus = np.concatenate([self.dist_focus, launch_geo[7]])
            self.width = np.concatenate([self.width, launch_geo[8]])
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
                self.R = np.array([launch_geo[2]])
                self.phi = np.array([launch_geo[3]])
                self.z = np.array([launch_geo[4]])
                self.theta_pol = np.array([launch_geo[5]])
                self.phi_tor = np.array([launch_geo[6]])
                self.dist_focus = np.array([launch_geo[7]])
                self.width = np.array([launch_geo[8]])
            else:
                self.N_ch = len(launch_geo[0])
                self.f = launch_geo[0]
                self.df = launch_geo[1]
                self.R = launch_geo[2]
                self.phi = launch_geo[3]
                self.z = launch_geo[4]
                self.theta_pol = launch_geo[5]
                self.phi_tor = launch_geo[6]
                self.dist_focus = launch_geo[7]
                self.width = launch_geo[8]
            if(np.isscalar(pol_coeff_X)):
                self.pol_coeff_X = np.zeros(self.N_ch)
                self.pol_coeff_X[:] = pol_coeff_X
            else:
                self.pol_coeff_X = pol_coeff_X

# Single entry diag dict with just EXT diag
                
DefaultDiagDict = od()
launch_geo = np.zeros((13, 1))
launch_geo[0, 0] = 140.e9
launch_geo[1, 0] = 0.2e9
launch_geo[2, 0] = 2.3
launch_geo[3, 0] = 104.0
launch_geo[4, 0] = 0.33
launch_geo[5, 0] = -0.824
launch_geo[6, 0] = -8.24
launch_geo[7, 0] = 1.1850
launch_geo[8, 0] = 0.0865
DefaultDiagDict.update({"EXT":  EXT_diag("EXT", launch_geo)})                
                
if(__name__ == "__main__"):
    ext_diag = EXT_diag("EXT")
    ext_diag.set_from_NETCDF("/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_Scenario.nc")
    print(ext_diag.get_launch())
