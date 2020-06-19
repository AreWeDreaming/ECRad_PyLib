'''
Created on Jun 18, 2020

@author: denk
'''
import numpy as np
from scipy.io import loadmat

class BasicDiag:
    def __init__(self, name, exp=None, diag=None, ed=None):
        self.name = name
        self.exp = exp
        self.diag = diag
        self.ed = ed
        
class ECEDiag(BasicDiag):
    '''
    classdocs
    '''

    def __init__(self, name, launch_geo=None, pol_coeff_X= -1):
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
        
    def set_from_mat(self, ray_launch_file):
        mdict = loadmat(ray_launch_file, squeeze_me=True)
        itime = 0 # Only first time point imported -> limitation of current ext diag
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
