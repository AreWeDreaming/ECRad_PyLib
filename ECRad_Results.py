'''
Created on Dec 17, 2015

@author: sdenk
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
np.set_printoptions(threshold=10)
import os
from GlobalSettings import globalsettings
from scipy.io import savemat, loadmat
from scipy import constants as cnst
from ECRad_Config import ECRadConfig
from ECRad_Scenario import ECRadScenario
from ndarray_helper import ndarray_math_operation, ndarray_check_for_None


class ECRadResults:
    def __init__(self):
        self.Config = ECRadConfig()
        self.Scenario = ECRadScenario()
        self.reset()

    def reset(self):
        # Does not reset Config or Scenario
        self.status = 0
        self.edition = 0
        self.init = False
        # Edition of this result
        # Remains zero until the result is saved
        # Time of the results, should be identical to self.Scenario.plasma_dict["time"]
        self.time = []
        # This holds the same information as the scenario
        self.modes = None
        self.Trad = []
        self.XTrad = []
        self.OTrad = []
        self.tau = []
        self.Xtau = []
        self.Otau = []
        self.X_mode_frac = []
        self.Trad_X_reflec = []
        self.tau_X_reflec = []
        self.Trad_O_reflec = []
        self.tau_O_reflec = []
        self.Trad_comp = []
        self.XTrad_comp = []
        self.OTrad_comp = []
        self.tau_comp = []
        self.Xtau_comp = []
        self.Otau_comp = []
        self.X_mode_frac_comp = []
        self.BPD = {}
        self.BPD["rhopX"] = []
        self.BPD["BPDX"] = []
        self.BPD["BPD_secondX"] = []
        self.BPD["rhopO"] = []
        self.BPD["BPDO"] = []
        self.BPD["BPD_secondO"] = []
        self.calib = {}  # First index diagnostic, second index channel
        self.calib_mat = {}  # First index diagnostic, second index time, third index channel
        self.std_dev_mat = {}  # First index diagnostic, second index time, third index channel
        self.rel_dev = {}  # First index diagnostic, second index channel
        self.sys_dev = {}  # First index diagnostic, second index channel
        self.masked_time_points = {}
        # self.los = {}  # First index quantity, second index time , third index channel, 4th index LOS
        self.ray = {}  # First index quantity, second index time , third index channel, 4th index ray, 5th index LOS
        self.resonance = {}  # First index quantity, second index time , third index channel
        self.resonance["s_cold"] = []
        self.resonance["R_cold"] = []
        self.resonance["z_cold"] = []
        self.resonance["rhop_cold"] = []
        self.resonance["s_warm"] = []
        self.resonance["R_warm"] = []
        self.resonance["z_warm"] = []
        self.resonance["rhop_warm"] = []
        self.resonance["s_warm_secondary"] = []
        self.resonance["R_warm_secondary"] = []
        self.resonance["z_warm_secondary"] = []
        self.resonance["rhop_warm_secondary"] = []
        self.ray["sX"] = []
        self.ray["xX"] = []
        self.ray["yX"] = []
        self.ray["zX"] = []
        self.ray["rhopX"] = []
        self.ray["HX"] = []
        self.ray["NX"] = []
        self.ray["NcX"] = []
        self.ray["XX"] = []
        self.ray["YX"] = []
        self.ray["thetaX"] = []
        self.ray["BPDX"] = []
        self.ray["BPD_secondX"] = []
        self.ray["emX"] = []
        self.ray["em_secondX"] = []
        self.ray["abX"] = []
        self.ray["ab_secondX"] = []
        self.ray["TX"] = []
        self.ray["T_secondX"] = []
        self.ray["TeX"] = []
        self.ray["sO"] = []
        self.ray["xO"] = []
        self.ray["yO"] = []
        self.ray["zO"] = []
        self.ray["rhopO"] = []
        self.ray["HO"] = []
        self.ray["NO"] = []
        self.ray["NcO"] = []
        self.ray["XO"] = []
        self.ray["YO"] = []
        self.ray["thetaO"] = []
        self.ray["BPDO"] = []
        self.ray["BPD_secondO"] = []
        self.ray["emO"] = []
        self.ray["em_secondO"] = []
        self.ray["abO"] = []
        self.ray["ab_secondO"] = []
        self.ray["TO"] = []
        self.ray["T_secondO"] = []
        self.ray["TeO"] = []
        self.weights = {}
        self.weights["ray"] = []
        self.weights["freq"] = []
        self.ray_launch = {}
        self.ray_launch["x"] = []
        self.ray_launch["y"] = []
        self.ray_launch["z"] = []
        self.ray_launch["f"] = []
        self.ray_launch["df"] = []
        self.ray_launch["pol_ang"] = []  # Degrees!
        self.ray_launch["tor_ang"] = []  # Degrees!
        self.ray_launch["dist_focus"] = []
        self.ray_launch["width"] = []
        self.ray_launch["pol_coeff_X"] = []
        self.comment = ""

    def append_new_results(self, time):
        # Open files first to check for any missing files
        too_few_rays = False
        if(self.modes == None):
            if(self.Config.considered_modes == 1):
                self.modes = ["X"]
            elif(self.Config.considered_modes == 2):
                self.modes = ["O"]
            elif(self.Config.considered_modes == 3):
                self.modes = ["X", "O"]
            else:
                print("Only mode = 1, 2, 3 supported")
                print("Selected mode: ", self.Config.considered_modes)
        if(self.status != 0):
            return False
        if(self.Config is None):
            print("Error -  config was not yet parsed into result instance")
            print("First parse config before appending results")
        if(self.Config.dstf == "Th"):
            Tradfilename = "TRadM_therm.dat"
        elif(self.Config.dstf == "Re"):
            Tradfilename = "TRadM_RELAX.dat"
        elif(self.Config.dstf == "Lu"):
            Tradfilename = "TRadM_LUKE.dat"
        elif(self.Config.dstf == "Ge" or self.Config.dstf == "GB"):
            Tradfilename = "TRadM_GENE.dat"
        Trad_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", Tradfilename))
        sres_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "sres.dat"))
        self.Trad.append(Trad_file.T[1])
        self.tau.append(Trad_file.T[2])
        if(not self.Config.extra_output):
            self.resonance["s_cold"].append(sres_file.T[0])
            self.resonance["R_cold"].append(sres_file.T[1])
            self.resonance["z_cold"].append(sres_file.T[2])
            self.resonance["rhop_cold"].append(sres_file.T[3])
            return
        if(self.Config.considered_modes == 3):
            XTrad_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "X_" + Tradfilename))
            OTrad_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "O_" + Tradfilename))
            self.XTrad.append(XTrad_file.T[1])
            self.OTrad.append(OTrad_file.T[1])
            self.Xtau.append(XTrad_file.T[2])
            self.Otau.append(OTrad_file.T[2])
            self.X_mode_frac.append(XTrad_file.T[3])
        sres_rel_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "sres_rel.dat"))
        self.resonance["rhop_warm_secondary"].append(sres_rel_file.T[7])
        if(self.Config.dstf == "Th"):
            Ich_folder = "Ich" + self.Config.dstf
            Trad_old_name = "TRadM_Farina.dat"
        elif(self.Config.dstf == "Re"):
            Ich_folder = "Ich" + self.Config.dstf
            Trad_old_name = "TRadM_therm.dat"
        elif(self.Config.dstf == "Ge"):
            Ich_folder = "Ich" + self.Config.dstf
            Trad_old_name = "TRadM_therm.dat"
        elif(self.Config.dstf == "GB"):
            Ich_folder = "Ich" + "Ge"
            Trad_old_name = "TRadM_therm.dat"
        else:
            print("Currently only Alabajar and Gray comparison are supported for saving and loading")
            print("Supported dstf are: \"Th\" and \"TB\":")
            print("Chosen dstf:", self.Config.dstf)
            return
        Trad_comp_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", Trad_old_name))
        Ich_folder = os.path.join(self.Config.working_dir, "ECRad_data", Ich_folder)
        # Ray_folder = os.path.join(self.Config.working_dir, "ECRad_data", "ray")
        # Append new empty list for this time point
        if("X" in self.modes):
            for key in self.resonance.keys():
                if(key.endswith("X")):
                    self.resonance[key].append([])
            for key in self.ray.keys():
                if(key.endswith("X")):
                    self.ray[key].append([])
            for key in self.BPD.keys():
                if(key.endswith("X")):
                    self.BPD[key].append([])
        if("O" in self.modes):
            for key in self.resonance.keys():
                if(key.endswith("O")):
                    self.resonance[key].append([])
            for key in self.ray.keys():
                if(key.endswith("O")):
                    self.ray[key].append([])
            for key in self.BPD.keys():
                if(key.endswith("O")):
                    self.BPD[key].append([])
        # Now add los info into this list for each channel
        if(np.isscalar(Trad_file.T[0])):
            N_ch = 1
        else:
            N_ch = len(Trad_file.T[0])
        for i in range(N_ch):
            if("X" in self.modes):
                try:
                    BDOP_X = np.loadtxt(os.path.join(Ich_folder, "BPDX{0:03n}.dat".format(i + 1)))
                except IOError:
                    raise IndexError("Failed to load BPD")  # Raise this as an Index error to communicate that results no l
                if(len(BDOP_X) > 0):
                    self.BPD["rhopX"][-1].append(BDOP_X.T[0])
                    self.BPD["BPDX"][-1].append(BDOP_X.T[1])
                    self.BPD["BPD_secondX"][-1].append(BDOP_X.T[2])
                    if(self.Config.N_ray > 1):
                        for key in self.ray.keys():
                            if(key.endswith("X")):
                                self.ray[key][-1].append([])
                        try:
                            Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(1, i + 1)))
                        except IOError:
                            raise IndexError("Failed to load Ray")
                        if(len(Ray_file) > 0):
                            for i_ray in range(self.Config.N_ray):
                                if(i_ray >= 1 and not os.path.isfile(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(i_ray + 1, i + 1)))):
                                    too_few_rays = True
                                    break
                                # Ray_file = np.loadtxt(os.path.join(Ray_folder, "Ray{0:03n}ch{1:03n}_X.dat".format(i_ray + 1, i + 1)).replace(",", ""))
                                try:
                                    Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(i_ray + 1, i + 1)))
                                except IOError:
                                    raise IndexError("Failed to load Ray")
                                self.ray["sX"][-1][-1].append(Ray_file.T[0])
                                self.ray["xX"][-1][-1].append(Ray_file.T[1])
                                self.ray["yX"][-1][-1].append(Ray_file.T[2])
                                self.ray["zX"][-1][-1].append(Ray_file.T[3])
                                self.ray["rhopX"][-1][-1].append(Ray_file.T[4])
                                self.ray["BPDX"][-1][-1].append(Ray_file.T[5])
                                self.ray["BPD_secondX"][-1][-1].append(Ray_file.T[6])
                                self.ray["emX"][-1][-1].append(Ray_file.T[7])
                                self.ray["em_secondX"][-1][-1].append(Ray_file.T[8])
                                self.ray["abX"][-1][-1].append(Ray_file.T[9])
                                self.ray["ab_secondX"][-1][-1].append(Ray_file.T[10])
                                self.ray["TX"][-1][-1].append(Ray_file.T[11])
                                self.ray["T_secondX"][-1][-1].append(Ray_file.T[12])
                                self.ray["TeX"][-1][-1].append(Ray_file.T[13])
                                self.ray["HX"][-1][-1].append(Ray_file.T[14])
                                self.ray["NX"][-1][-1].append(Ray_file.T[15])
                                self.ray["NcX"][-1][-1].append(Ray_file.T[16])
                                self.ray["thetaX"][-1][-1].append(Ray_file.T[17])
                                Bx = Ray_file.T[21]
                                By = Ray_file.T[22]
                                Bz = Ray_file.T[23]
                                itime = np.argmin(np.abs(self.Scenario.plasma_dict["time"] - time))
                                omega = 2.0 * np.pi * self.Scenario.ray_launch[itime]["f"][i]
                                self.ray["YX"][-1][-1].append(cnst.e * np.sqrt(Bx**2 + By**2 + Bz**2) / \
                                                              (cnst.m_e * omega))
                                if(self.Scenario.profile_dimension == 1):
                                    ne_spl = InterpolatedUnivariateSpline(self.Scenario.plasma_dict["rhop_prof"][itime], \
                                                                          np.log(self.Scenario.plasma_dict["ne"][itime]), ext=3)
                                    self.ray["XX"][-1][-1].append(cnst.e**2 * np.exp(ne_spl(self.ray["rhopX"][-1][-1][-1]))/ \
                                                                  (cnst.m_e * cnst.epsilon_0))
                                else:
                                    ne_spl = RectBivariateSpline(self.Scenario.plasma_dict["eq_data"][itime].R, \
                                                                 self.Scenario.plasma_dict["eq_data"][itime].R, \
                                                                 np.log(self.Scenario.plasma_dict["ne"][itime]), ext=3)
                                    R_ray = self.ray["R"][-1][-1][-1]
                                    z_ray = self.ray["z"][-1][-1][-1]
                                    R_ray[np.logical_or(R_ray > np.max(self.Scenario.plasma_dict["eq_data"][itime].R), \
                                                        R_ray < np.min(self.Scenario.plasma_dict["eq_data"][itime].R) )] = \
                                         np.max(self.Scenario.plasma_dict["eq_data"][itime].R)
                                    z_ray[np.logical_or(z_ray > np.max(self.Scenario.plasma_dict["eq_data"][itime].z), \
                                                        z_ray < np.min(self.Scenario.plasma_dict["eq_data"][itime].z) )] = \
                                         np.max(self.Scenario.plasma_dict["eq_data"][itime].z)
                                    self.ray["XX"][-1][-1].append(cnst.e**2 * np.exp(ne_spl(R_ray, z_ray, grid=False))/ \
                                                                  (cnst.m_e * cnst.epsilon_0))
                                    self.ray["XX"][-1][-1][-1][np.logical_or(R_ray >= np.max(self.Scenario.plasma_dict["eq_data"][itime].R), \
                                                                             z_ray >= np.max(self.Scenario.plasma_dict["eq_data"][itime].z) )] = 0.0
                                                                                
                    else:
                        try:
                            Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(1, i + 1)))
                        except IOError:
                            raise IndexError("Failed to load Ray")  # Raise this as an Index error to communicate that results no
                        if(len(Ray_file) > 0):  # Can be empty if mode is skipped because of ext. pol. coeff.= 0
                            self.ray["sX"][-1].append(Ray_file.T[0])
                            self.ray["xX"][-1].append(Ray_file.T[1])
                            self.ray["yX"][-1].append(Ray_file.T[2])
                            self.ray["zX"][-1].append(Ray_file.T[3])
                            self.ray["rhopX"][-1].append(Ray_file.T[4])
                            self.ray["BPDX"][-1].append(Ray_file.T[5])
                            self.ray["BPD_secondX"][-1].append(Ray_file.T[6])
                            self.ray["emX"][-1].append(Ray_file.T[7])
                            self.ray["em_secondX"][-1].append(Ray_file.T[8])
                            self.ray["abX"][-1].append(Ray_file.T[9])
                            self.ray["ab_secondX"][-1].append(Ray_file.T[10])
                            self.ray["TX"][-1].append(Ray_file.T[11])
                            self.ray["T_secondX"][-1].append(Ray_file.T[12])
                            self.ray["TeX"][-1].append(Ray_file.T[13])
                            self.ray["HX"][-1].append(Ray_file.T[14])
                            self.ray["NX"][-1].append(Ray_file.T[15])
                            self.ray["NcX"][-1].append(Ray_file.T[16])
                            self.ray["thetaX"][-1].append(Ray_file.T[17])
                            Bx = Ray_file.T[21]
                            By = Ray_file.T[22]
                            Bz = Ray_file.T[23]
                            itime = np.argmin(np.abs(self.Scenario.plasma_dict["time"] - time))
                            omega = 2.0 * np.pi * self.Scenario.ray_launch[itime]["f"][i]
                            self.ray["YX"][-1].append(cnst.e * np.sqrt(Bx**2 + By**2 + Bz**2) / \
                                                          (cnst.m_e * omega))
                            if(self.Scenario.profile_dimension == 1):
                                ne_spl = InterpolatedUnivariateSpline(self.Scenario.plasma_dict["rhop_prof"][itime], \
                                                                      np.log(self.Scenario.plasma_dict["ne"][itime]), ext=1)
                                self.ray["XX"][-1].append(cnst.e**2 * np.exp(ne_spl(self.ray["rhopX"][-1][-1]))/ \
                                                              (cnst.m_e * cnst.epsilon_0* omega**2))
                            elif(self.Scenario.profile_dimension == 3):
                                ne_spl = RectBivariateSpline(self.Scenario.plasma_dict["eq_data"][itime].R, \
                                                             self.Scenario.plasma_dict["eq_data"][itime].z, \
                                                             np.log(self.Scenario.plasma_dict["ne"][itime]))
                                R_ray = self.ray["R"][-1][-1]
                                z_ray = self.ray["z"][-1][-1]
                                R_ray[np.logical_or(R_ray > np.max(self.Scenario.plasma_dict["eq_data"][itime].R), \
                                                    R_ray < np.min(self.Scenario.plasma_dict["eq_data"][itime].R) )] = \
                                     np.max(self.Scenario.plasma_dict["eq_data"][itime].R)
                                z_ray[np.logical_or(z_ray > np.max(self.Scenario.plasma_dict["eq_data"][itime].z), \
                                                    z_ray < np.min(self.Scenario.plasma_dict["eq_data"][itime].z) )] = \
                                     np.max(self.Scenario.plasma_dict["eq_data"][itime].z)
                                self.ray["XX"][-1].append(cnst.e**2 * np.exp(ne_spl(R_ray, z_ray, grid=False))/ \
                                                              (cnst.m_e * cnst.epsilon_0 * omega**2))
                                self.ray["XX"][-1][-1][np.logical_or(R_ray >= np.max(self.Scenario.plasma_dict["eq_data"][itime].R), \
                                                                         z_ray >= np.max(self.Scenario.plasma_dict["eq_data"][itime].z) )] = 0.0
                            else:
                                print("Three dimensional profiles not supported for Stix parameters X and Y")
                                break
            if("O" in self.modes):
                try:
                    BDOP_O = np.loadtxt(os.path.join(Ich_folder, "BPDO{0:03d}.dat".format(i + 1)))
                except IOError:
                    raise IndexError("Failed to load BPD")  # Raise this as an Index error to communicate that results no l
                self.BPD["rhopO"][-1].append(BDOP_O.T[0])
                self.BPD["BPDO"][-1].append(BDOP_O.T[1])
                self.BPD["BPD_secondO"][-1].append(BDOP_O.T[2])
                if(self.Config.N_ray > 1):
                    for key in self.ray.keys():
                        if(key.endswith("O")):
                            self.ray[key][-1].append([])
                    try:
                        Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_O.dat".format(1, i + 1)))
                    except IOError:
                        raise IndexError("Failed to load Ray")  # Raise this as an Index error to communicate that results no l
                    if(len(Ray_file) > 0):  # Can be empty if mode is skipped because of ext. pol. coeff.= 0
                        for i_ray in range(self.Config.N_ray):
                            if(i_ray >= 1 and not os.path.isfile(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(i_ray + 1, i + 1)))):
                                too_few_rays = True
                                break
                            try:
                                Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_O.dat".format(i_ray + 1, i + 1)))
                            except IOError:
                                raise IndexError("Failed to load Ray")
                            self.ray["sO"][-1][-1].append(Ray_file.T[0])
                            self.ray["xO"][-1][-1].append(Ray_file.T[1])
                            self.ray["yO"][-1][-1].append(Ray_file.T[2])
                            self.ray["zO"][-1][-1].append(Ray_file.T[3])
                            self.ray["rhopO"][-1][-1].append(Ray_file.T[4])
                            self.ray["BPDO"][-1][-1].append(Ray_file.T[5])
                            self.ray["BPD_secondO"][-1][-1].append(Ray_file.T[6])
                            self.ray["emO"][-1][-1].append(Ray_file.T[7])
                            self.ray["em_secondO"][-1][-1].append(Ray_file.T[8])
                            self.ray["abO"][-1][-1].append(Ray_file.T[9])
                            self.ray["ab_secondO"][-1][-1].append(Ray_file.T[10])
                            self.ray["TO"][-1][-1].append(Ray_file.T[11])
                            self.ray["T_secondO"][-1][-1].append(Ray_file.T[12])
                            self.ray["TeO"][-1][-1].append(Ray_file.T[7])
                            self.ray["HO"][-1][-1].append(Ray_file.T[8])
                            self.ray["NO"][-1][-1].append(Ray_file.T[9])
                            self.ray["NcO"][-1][-1].append(Ray_file.T[10])
                            self.ray["thetaO"][-1][-1].append(Ray_file.T[11])
                else:
                    try:
                        Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_O.dat".format(1, i + 1)))
                    except IOError:
                        raise IndexError("Failed to load Ray")
                    if(len(Ray_file) > 0):
                        self.ray["sO"][-1].append(Ray_file.T[0])
                        self.ray["xO"][-1].append(Ray_file.T[1])
                        self.ray["yO"][-1].append(Ray_file.T[2])
                        self.ray["zO"][-1].append(Ray_file.T[3])
                        self.ray["rhopO"][-1].append(Ray_file.T[4])
                        self.ray["BPDO"][-1].append(Ray_file.T[5])
                        self.ray["BPD_secondO"][-1].append(Ray_file.T[6])
                        self.ray["emO"][-1].append(Ray_file.T[7])
                        self.ray["em_secondO"][-1].append(Ray_file.T[8])
                        self.ray["abO"][-1].append(Ray_file.T[9])
                        self.ray["ab_secondO"][-1].append(Ray_file.T[10])
                        self.ray["TO"][-1].append(Ray_file.T[11])
                        self.ray["T_secondO"][-1].append(Ray_file.T[12])
                        self.ray["TeO"][-1].append(Ray_file.T[13])
                        self.ray["HO"][-1].append(Ray_file.T[14])
                        self.ray["NO"][-1].append(Ray_file.T[15])
                        self.ray["NcO"][-1].append(Ray_file.T[16])
                        self.ray["thetaO"][-1].append(Ray_file.T[17])
        self.weights["ray"].append(np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "ray_weights.dat"), ndmin=2))
        self.weights["freq"].append(np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "freq_weights.dat"), ndmin=2))
        self.resonance["s_cold"].append(sres_file.T[0])
        self.resonance["R_cold"].append(sres_file.T[1])
        self.resonance["z_cold"].append(sres_file.T[2])
        self.resonance["rhop_cold"].append(sres_file.T[3])
        self.resonance["s_warm"].append(sres_rel_file.T[0])
        self.resonance["R_warm"].append(sres_rel_file.T[1])
        self.resonance["z_warm"].append(sres_rel_file.T[2])
        self.resonance["rhop_warm"].append(sres_rel_file.T[3])
        self.resonance["s_warm_secondary"].append(sres_rel_file.T[4])
        self.resonance["R_warm_secondary"].append(sres_rel_file.T[5])
        self.resonance["z_warm_secondary"].append(sres_rel_file.T[6])
        self.Trad_comp.append(Trad_comp_file.T[1])
        self.tau_comp.append(Trad_comp_file.T[2])
        if(self.Config.considered_modes == 3):
            XTrad_comp_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "X_" + Trad_old_name))
            OTrad_comp_file = np.loadtxt(os.path.join(self.Config.working_dir, "ECRad_data", "O_" + Trad_old_name))
            self.XTrad_comp.append(XTrad_comp_file.T[1])
            self.OTrad_comp.append(OTrad_comp_file.T[1])
            self.Xtau_comp.append(XTrad_comp_file.T[2])
            self.Otau_comp.append(OTrad_comp_file.T[2])
            self.X_mode_frac_comp.append(XTrad_comp_file.T[3])
        # Store the launch geometry of this run for future runs
        if(too_few_rays):
            print("Warning! Only a single ray file was produced for this multi ray calculation!")
            print("Even though Trad and BDOP account for multiple rays, only central ray geometry is available for the plots!")
        self.time.append(time)

    def tidy_up(self, autosave=True):
        if(self.status != 0):
            return
        # Put everything into numpy arrays
        self.Trad = np.array(self.Trad)
        self.tau = np.array(self.tau)
        self.Trad_comp = np.array(self.Trad_comp)
        self.tau_comp = np.array(self.tau_comp)
        self.XTrad = np.array(self.XTrad)
        self.OTrad = np.array(self.OTrad)
        self.Xtau = np.array(self.Xtau)
        self.Otau = np.array(self.Otau)
        self.X_mode_frac = np.array(self.X_mode_frac)
        self.Trad_comp = np.array(self.Trad_comp)
        self.tau_comp = np.array(self.tau_comp)
        self.XTrad_comp = np.array(self.XTrad_comp)
        self.OTrad_comp = np.array(self.OTrad_comp)
        self.Xtau_comp = np.array(self.Xtau_comp)
        self.Otau_comp = np.array(self.Otau_comp)
        self.X_mode_frac_comp = np.array(self.X_mode_frac_comp)
        for key in self.resonance.keys():
            self.resonance[key] = np.array(self.resonance[key])
        for key in self.ray_launch:
            self.ray_launch[key] = np.array(self.ray_launch[key])
        for key in self.ray.keys():
            self.ray[key] = np.array(self.ray[key])
        for key in self.BPD.keys():
            self.BPD[key] = np.array(self.BPD[key])
        if(not self.init):
            self.init = True
        for key in self.weights.keys():
            self.weights[key] = np.array(self.weights[key])
        self.time = np.array(self.time)
        # Autosave results
        if(autosave):
            self.to_mat_file(comment=self.comment)

    def UpdateCalib(self, diag, calib, calib_mat, std_dev_mat, rel_dev, sys_dev, masked_time_points):
        self.calib[diag.name] = calib
        self.calib_mat[diag.name] = calib_mat
        self.std_dev_mat[diag.name] = std_dev_mat
        self.rel_dev[diag.name] = rel_dev
        self.sys_dev[diag.name] = sys_dev
        self.masked_time_points[diag.name] = masked_time_points

    def from_mat_file(self, filename):
        try:
            mdict = loadmat(filename, chars_as_strings=True, squeeze_me=True)
        except IOError as e:
            print(e)
            print("Error: " + filename + " does not exist")
            return False
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["calib_diags", "time"]
        at_least_2d_keys = ["Trad", "Trad_comp", "tau", "tau_comp", \
                            "XTrad", "OTrad", "Xtau", "Otau", "X_mode_frac", \
                            "XTrad_comp", "OTrad_comp", "Xtau_comp", "Otau_comp", "X_mode_frac_comp", \
                             "ne", "calib", "rel_dev", "sys_dev", "Te", "rhop", "masked_time_points", \
                             "ECE_rhop", "ECE_dat", "eq_data", \
                             "eq_R", "eq_z", "launch_diag_name", "launch_f", "launch_df", "launch_R", "launch_phi", \
                             "launch_z", "launch_tor_ang" , "launch_pol_ang", "launch_dist_focus", \
                             "launch_width", "eq_special", "eq_special_complete"  ] + self.resonance.keys()
        at_least_3d_keys = self.BPD.keys()
        at_least_3d_keys[at_least_3d_keys.index("rhopX")] = "BPDrhopX"
        at_least_3d_keys[at_least_3d_keys.index("rhopO")] = "BPDrhopO"
        at_least_3d_keys += self.ray.keys() + ["ray_BPDX", "ray_BPDO", "ray_BPD_secondX", "ray_BPD_secondO", "ray_emX", "ray_emO", \
                                               "ray_abX", "ray_abO", "ray_TX", "ray_TO", "ray_em_secondX", "ray_em_secondO", \
                                               "ray_absecondX", "ray_absecondO", "ray_TsecondX", "ray_TsecondO", "freq_weights", \
                                               "ray_weights"]
        at_least_3d_keys += ["std_dev_mat", "calib_mat"]
        self.Config.from_mat_file(mdict=mdict)
        self.Scenario.from_mat(mdict=mdict, load_plasma_dict=True)
        increase_time_dim = False
        if(np.isscalar(mdict["time"])):
            increase_time_dim = True
        elif(len(mdict["time"]) == 1):
            increase_time_dim = True
        increase_diag_dim = False
        try:
            if(len(np.unique(mdict["diag_name"])) == 1):
                increase_diag_dim = True
        except KeyError:
            if(len(np.unique(mdict["launch_diag_name"])) == 1):
                increase_diag_dim = True
        for key in mdict.keys():
            if(not key.startswith("_")):  # throw out the .mat specific information
                try:
                    if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                        mdict[key] = np.atleast_1d(mdict[key])
                    elif(key in at_least_2d_keys):
                        if(key in ["calib", "rel_dev", "sys_dev"]):
                            if(increase_diag_dim):
                                mdict[key] = np.array([mdict[key]])
                        else:
                            mdict[key] = np.atleast_2d(mdict[key])
                    elif(key in at_least_3d_keys):
                        if(increase_time_dim):
                            if(key == "std_dev_mat" or key == "calib_mat"):
                                for i in range(len(mdict[key])):
                                    mdict[key][i] = np.array([mdict[key][i]])
                            else:
                                mdict[key] = np.array([mdict[key]])
                        elif((key == "std_dev_mat" or key == "calib_mat") and increase_diag_dim):
                            mdict[key] = np.atleast_3d([mdict[key]])
                except Exception as e:
                    print(key)
                    print(e)
        self.edition = mdict["edition"]
        if(mdict["considered_modes"] == 1):
            self.modes = ["X"]
        elif(mdict["considered_modes"] == 2):
            self.modes = ["O"]
        elif(mdict["considered_modes"] == 3):
            self.modes = ["X", "O"]
        if("comment" in mdict.keys()):
            self.comment = mdict["comment"]
            try:
                if(type(self.comment) == np.ndarray):
                    if(len(self.comment) > 0):
                        self.comment = self.comment[0]
                    else:
                        self.comment =  ""
            except Exception as e:
                print("Failed to parse comment")
                print(e)
                self.comment = ""
        self.time = mdict["time"]
        self.Trad = mdict["Trad"]
        self.tau = mdict["tau"]
        self.resonance["s_cold"] = mdict["s_cold"]
        self.resonance["R_cold"] = mdict["R_cold"]
        self.resonance["z_cold"] = mdict["z_cold"]
        self.resonance["rhop_cold"] = mdict["rhop_cold"]
        if(self.Config.extra_output):
            if(self.Config.considered_modes == 3):
                try:
                    self.XTrad = mdict["XTrad"]
                    self.OTrad = mdict["OTrad"]
                    self.Xtau = mdict["Xtau"]
                    self.Otau = mdict["Otau"]
                    self.X_mode_frac = mdict["X_mode_frac"]
                except KeyError:
                    print("Could not find X and O-mode specific Trad and tau info for primary model, skipping this")
                    self.XTrad = []
                    self.OTrad = []
                    self.Xtau = []
                    self.Otau = []
                    self.X_mode_frac = []
            self.resonance["s_warm"] = mdict["s_warm"]
            self.resonance["R_warm"] = mdict["R_warm"]
            self.resonance["z_warm"] = mdict["z_warm"]
            self.resonance["rhop_warm"] = mdict["rhop_warm"]
            try:
                self.resonance["s_warm_secondary"] = mdict["s_warm_secondary"]
                self.resonance["R_warm_secondary"] = mdict["R_warm_secondary"]
                self.resonance["z_warm_secondary"] = mdict["z_warm_secondary"]
                self.resonance["rhop_warm_secondary"] = mdict["rhop_warm_secondary"]
            except KeyError:
                print("Only the warm resonances of the primary model could be found!")
                print("Is this an old file?")
        try:
            if("calib_diags" in mdict.keys()  and len(mdict["calib"]) > 0):
                if(len(mdict["calib_diags"]) == 1):
                    self.calib[mdict["calib_diags"][0]] = mdict["calib"][0]
                    self.calib_mat[mdict["calib_diags"][0]] = mdict["calib_mat"][0].T
                    self.std_dev_mat[mdict["calib_diags"][0]] = mdict["std_dev_mat"][0].T
                    self.rel_dev[mdict["calib_diags"][0]] = mdict["rel_dev"][0]
                    try:
                        self.sys_dev[mdict["calib_diags"][0]] = mdict["sys_dev"][0]
                    except KeyError:
                        print("No systematic errors in .mat file")
                        self.sys_dev[mdict["calib_diags"][0]] = np.zeros(self.rel_dev[mdict["calib_diags"][0]].shape)
                    try:
                        self.masked_time_points[mdict["calib_diags"][0]] = np.bool8(mdict["masked_time_points"][0])
                    except KeyError:
                        print("Masked time points for calibration not specified")
                        self.masked_time_points[mdict["calib_diags"][0]] = np.zeros(self.time.shape, dtype=np.bool8)
                        self.masked_time_points[mdict["calib_diags"][0]][:] = True
                else:
                    for i in range(len(mdict["calib_diags"])):
                        self.calib[mdict["calib_diags"][i]] = mdict["calib"][i]
                        self.calib_mat[mdict["calib_diags"][i]] = mdict["calib_mat"][i].T
                        self.std_dev_mat[mdict["calib_diags"][i]] = mdict["std_dev_mat"][i].T
                        self.rel_dev[mdict["calib_diags"][i]] = mdict["rel_dev"][i]
                        try:
                            self.sys_dev[mdict["calib_diags"][i]] = mdict["sys_dev"][i]
                        except KeyError:
                            print("No systematic errors in .mat file")
                            self.sys_dev[mdict["calib_diags"][i]] = np.zeros(self.rel_dev[mdict["calib_diags"][i]].shape)
                        try:
                            self.masked_time_points[mdict["calib_diags"][i]] = np.bool8(mdict["masked_time_points"][i])
                        except KeyError:
                            print("Masked time points for calibration not specified")
                            self.masked_time_points[mdict["calib_diags"][i]] = np.zeros(self.time.shape, dtype=np.bool8)
                            self.masked_time_points[mdict["calib_diags"][i]][:] = True
        except TypeError:
            print("Error loading calibration factors - please recalculate")
        if(not self.Config.extra_output):
            self.init = True
            return True
        self.Trad_comp = mdict["Trad_comp"]
        self.tau_comp = mdict["tau_comp"]
        try:
            self.weights["freq"] = mdict["freq_weights"]
            self.weights["ray"] = mdict["ray_weights"]
            if(self.Config.N_ray == 1):
                for key in self.weights.keys():
                    self.weights[key] = self.weights[key].reshape((len(self.time), len(self.weights[key][0]), 1))
        except:
            self.weights["freq"] = None
            self.weights["ray"] = None
            print("Warning frequency and ray weights not found in .mat file!")
        if(self.Config.considered_modes == 3):
            try:
                self.XTrad_comp = mdict["XTrad_comp"]
                self.OTrad_comp = mdict["OTrad_comp"]
                self.Xtau_comp = mdict["Xtau_comp"]
                self.Otau_comp = mdict["Otau_comp"]
                self.X_mode_frac_comp = mdict["X_mode_frac_comp"]
            except KeyError:
                print("Could not find X and O-mode specific Trad and tau info for 2nd model, skipping this")
                self.XTrad_comp = []
                self.OTrad_comp = []
                self.Xtau_comp = []
                self.Otau_comp = []
                self.X_mode_frac_comp = []
        if("X" in self.modes):
            try:
                self.BPD["BPDX"] = mdict["BPDX"]
            except KeyError:
                self.BPD["BPDX"] = mdict["BDP_X"]
            try:
                self.BPD["BPD_secondX"] = mdict["BPD_secondX"]
            except KeyError:
                self.BPD["BPD_secondX"] = mdict["BDP_X_comp"]
            try:
                self.BPD["rhopX"] = mdict["BPDrhopX"]
            except KeyError:
                self.BPD["rhopX"] = mdict["LOS-rhopX"]
#            self.los["sX"] = mdict["LOS-sX"]
#            self.los["RX"] = mdict["LOS-RX"]
#            self.los["zX"] = mdict["LOS-zX"]
#            self.los["rhopX"] = mdict["LOS-rhopX"]
#            self.los["TeX"] = mdict["LOS-TeX"]
            try:
                self.ray["sX"] = mdict["sX"]
                self.ray["xX"] = mdict["xX"]
                self.ray["yX"] = mdict["yX"]
                self.ray["zX"] = mdict["zX"]
                self.ray["HX"] = mdict["HX"]
                self.ray["NX"] = mdict["NX"]
                self.ray["NcX"] = mdict["NcX"]
                try:
                    self.ray["rhopX"] = mdict["rhopX"]
                    self.ray["TeX"] = mdict["TeX"]
                    self.ray["BPDX"] = mdict["ray_BPDX"]
                    self.ray["BPD_secondX"] = mdict["ray_BPD_secondX"]
                except KeyError:
                    print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                try:
                    self.ray["XX"] = mdict["XX"]
                    self.ray["YX"] = mdict["YX"]
                except KeyError:
                    print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                try:
                    self.ray["emX"] = mdict["ray_emX"]
                    self.ray["em_secondX"] = mdict["ray_em_secondX"]
                    self.ray["abX"] = mdict["ray_abX"]
                    self.ray["ab_secondX"] = mdict["ray_ab_secondX"]
                    self.ray["TX"] = mdict["ray_TX"]
                    self.ray["T_secondX"] = mdict["ray_T_secondX"]
                    self.ray["TeX"] = mdict["TeX"]
                except KeyError:
                    print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
            except KeyError:
                print("Warning! No Ray information in loaded .mat file. Is this an old file?")
            try:
                self.ray["thetaX"] = mdict["thetaX"]
            except KeyError:
                print("Warning! theta not found in loaded .mat file. Is this an old file?")
        if("O" in self.modes):
            try:
                self.BPD["BPDO"] = mdict["BPDO"]
            except KeyError:
                self.BPD["BPDO"] = mdict["BDP_O"]
            try:
                self.BPD["BPD_secondO"] = mdict["BPD_secondO"]
            except KeyError:
                self.BPD["BPD_secondO"] = mdict["BPD_O_comp"]
            try:
                self.BPD["rhopO"] = mdict["BPDrhopO"]
            except KeyError:
                self.BPD["rhopO"] = mdict["LOS-rhopO"]
#            self.los["sO"] = mdict["LOS-sO"]
#            self.los["RO"] = mdict["LOS-RO"]
#            self.los["zO"] = mdict["LOS-zO"]
#            self.los["rhopO"] = mdict["LOS-rhopO"]
#            self.los["TeO"] = mdict["LOS-TeO"]
            try:
                self.ray["sO"] = mdict["sO"]
                self.ray["xO"] = mdict["xO"]
                self.ray["yO"] = mdict["yO"]
                self.ray["zO"] = mdict["zO"]
                self.ray["HO"] = mdict["HO"]
                self.ray["NO"] = mdict["NO"]
                self.ray["NcO"] = mdict["NcO"]
                try:
                    self.ray["rhopO"] = mdict["rhopO"]
                    self.ray["TeO"] = mdict["TeO"]
                    self.ray["BPDO"] = mdict["ray_BPDO"]
                    self.ray["BPD_secondO"] = mdict["ray_BPD_secondO"]
                except KeyError:
                    print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                try:
                    self.ray["XO"] = mdict["XO"]
                    self.ray["YO"] = mdict["YO"]
                except KeyError:
                    print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                try:
                    self.ray["emO"] = mdict["ray_emO"]
                    self.ray["em_secondO"] = mdict["ray_em_secondO"]
                    self.ray["abO"] = mdict["ray_abO"]
                    self.ray["ab_secondO"] = mdict["ray_ab_secondO"]
                    self.ray["TO"] = mdict["ray_TO"]
                    self.ray["T_secondO"] = mdict["ray_T_secondO"]
                except KeyError:
                    print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
            except KeyError:
                print("Warning! No Ray information in loaded .mat file. Is this an old file?")
            try:
                self.ray["thetaO"] = mdict["thetaO"]
            except KeyError:
                print("Warning! theta not found in loaded .mat file. Is this an old file?")
        self.init = True
        return True

    def to_mat_file(self, ext_filename=None, comment=None, quasi_linear_beam=None, dist_obj=None, linear_beam=None):
        ed = 1
        diag_str = ""
        for key in self.Scenario.used_diags_dict.keys():
            diag_str += key
        if(ext_filename is not None):
            filename = ext_filename
        elif(len(self.calib.keys()) > 0):
            filename = os.path.join(self.Config.working_dir, "ECRad_{0:5d}_{1:s}_w_calib_ed{2:d}.mat".format(self.Scenario.shot, diag_str, ed))
        else:
            filename = os.path.join(self.Config.working_dir, "ECRad_{0:5d}_{1:s}_ed{2:d}.mat".format(self.Scenario.shot, diag_str, ed))
        while(os.path.exists(filename) and ext_filename is None):
            ed += 1
            if(len(self.calib.keys()) > 0):
                filename = os.path.join(self.Config.working_dir, "ECRad_{0:5d}_{1:s}_w_calib_ed{2:d}.mat".format(self.Scenario.shot, diag_str, ed))
            else:
                filename = os.path.join(self.Config.working_dir, "ECRad_{0:5d}_{1:s}_ed{2:d}.mat".format(self.Scenario.shot, diag_str, ed))
        mdict = {}
        mdict["edition"] = ed
        if(comment is not None):
            mdict["comment"] = comment
        elif(self.comment is not None):
            mdict["comment"] = self.comment
        mdict["ECRad_git_tag"] = np.loadtxt(os.path.join(globalsettings.ECRadRoot, "id"),dtype=np.str)
        try:
            mdict["ECRadGUI_git_tag"] =  np.loadtxt(os.path.join(globalsettings.GUI_folder, "id"),dtype=np.str)
        except IOError:
            mdict["ECRadGUI_git_tag"] = "Unknown"
        try:
            mdict["ECRadPylib_git_tag"] = np.loadtxt(os.path.join(globalsettings.pylib_folder, "id"),dtype=np.str)
        except IOError:
            mdict["ECRadPylib_git_tag"] = "Unknown"
        mdict["time"] = self.time
        mdict["Trad"] = self.Trad
        mdict["tau"] = self.tau
        mdict = self.Config.saveconfig(mdict)
        mdict = self.Scenario.to_mat_file(mdict_in=mdict)
        if(self.Config.extra_output):
            mdict["Trad_comp"] = self.Trad_comp
            mdict["tau_comp"] = self.tau_comp
            if(self.Config.considered_modes == 3):
                mdict["XTrad"] = self.XTrad
                mdict["OTrad"] = self.OTrad
                mdict["Xtau"] = self.Xtau
                mdict["Otau"] = self.Otau
                mdict["X_mode_frac"] = self.X_mode_frac
                mdict["XTrad_comp"] = self.XTrad_comp
                mdict["OTrad_comp"] = self.OTrad_comp
                mdict["Xtau_comp"] = self.Xtau_comp
                mdict["Otau_comp"] = self.Otau_comp
                mdict["X_mode_frac_comp"] = self.X_mode_frac_comp
        mdict["s_cold"] = self.resonance["s_cold"]
        mdict["R_cold"] = self.resonance["R_cold"]
        mdict["z_cold"] = self.resonance["z_cold"]
        mdict["rhop_cold"] = self.resonance["rhop_cold"]
        if(self.Config.extra_output):
            mdict["s_warm"] = self.resonance["s_warm"]
            mdict["R_warm"] = self.resonance["R_warm"]
            mdict["z_warm"] = self.resonance["z_warm"]
            mdict["rhop_warm"] = self.resonance["rhop_warm"]
            mdict["s_warm_secondary"] = self.resonance["s_warm_secondary"]
            mdict["R_warm_secondary"] = self.resonance["R_warm_secondary"]
            mdict["z_warm_secondary"] = self.resonance["z_warm_secondary"]
            mdict["rhop_warm_secondary"] = self.resonance["rhop_warm_secondary"]
        if("X" in self.modes and self.Config.extra_output):
            mdict["BPDX"] = self.BPD["BPDX"]
            mdict["BPD_secondX"] = self.BPD["BPD_secondX"]
            mdict["BPDrhopX"] = self.BPD["rhopX"]
#            mdict["LOS-sX"] = self.los["sX"]
#            # print(mdict["LOS-sX"])
#            mdict["LOS-RX"] = self.los["RX"]
#            mdict["LOS-zX"] = self.los["zX"]
#            mdict["LOS-rhopX"] = self.los["rhopX"]
#            mdict["LOS-TeX"] = self.los["TeX"]
            mdict["sX"] = self.ray["sX"]
            mdict["xX"] = self.ray["xX"]
            mdict["yX"] = self.ray["yX"]
            mdict["zX"] = self.ray["zX"]
            mdict["rhopX"] = self.ray["rhopX"]
            mdict["HX"] = self.ray["HX"]
            mdict["NX"] = self.ray["NX"]
            mdict["NcX"] = self.ray["NcX"]
            mdict["thetaX"] = self.ray["thetaX"]
            mdict["XX"] = self.ray["XX"]
            mdict["YX"] = self.ray["YX"]
            mdict["TeX"] = self.ray["TeX"]
            mdict["ray_BPDX"] = self.ray["BPDX"]
            mdict["ray_BPD_secondX"] = self.ray["BPD_secondX"]
            mdict["ray_emX"] = self.ray["emX"]
            mdict["ray_em_secondX"] = self.ray["em_secondX"]
            mdict["ray_abX"] = self.ray["abX"]
            mdict["ray_ab_secondX"] = self.ray["ab_secondX"]
            mdict["ray_TX"] = self.ray["TX"]
            mdict["ray_T_secondX"] = self.ray["T_secondX"]
        if("O" in self.modes and self.Config.extra_output):
            mdict["BPDO"] = self.BPD["BPDO"]
            mdict["BPD_secondO"] = self.BPD["BPD_secondO"]
            mdict["BPDrhopO"] = self.BPD["rhopO"]
#            mdict["LOS-sO"] = self.los["sO"]
#            mdict["LOS-RO"] = self.los["RO"]
#            mdict["LOS-zO"] = self.los["zX"]
#            mdict["LOS-rhopO"] = self.los["rhopO"]
#            mdict["LOS-TeO"] = self.los["TeO"]
            mdict["sO"] = self.ray["sO"]
            mdict["xO"] = self.ray["xO"]
            mdict["yO"] = self.ray["yO"]
            mdict["zO"] = self.ray["zO"]
            mdict["rhopO"] = self.ray["rhopO"]
            mdict["HO"] = self.ray["HO"]
            mdict["NO"] = self.ray["NO"]
            mdict["NcO"] = self.ray["NcO"]
            mdict["thetaO"] = self.ray["thetaO"]
            mdict["XO"] = self.ray["XO"]
            mdict["YO"] = self.ray["YO"]
            mdict["TeO"] = self.ray["TeO"]
            mdict["ray_BPDO"] = self.ray["BPDO"]
            mdict["ray_BPD_secondO"] = self.ray["BPD_secondO"]
            mdict["ray_emO"] = self.ray["emO"]
            mdict["ray_em_secondO"] = self.ray["em_secondO"]
            mdict["ray_abO"] = self.ray["abO"]
            mdict["ray_ab_secondO"] = self.ray["ab_secondO"]
            mdict["ray_TO"] = self.ray["TO"]
            mdict["ray_T_secondO"] = self.ray["T_secondO"]
        if(len(self.calib.keys()) > 0):
            mdict["calib"] = []
            mdict["calib_mat"] = []
            mdict["std_dev_mat"] = []
            mdict["rel_dev"] = []
            mdict["sys_dev"] = []
            mdict["masked_time_points"] = []
            mdict["calib_diags"] = []
            for diag in self.calib.keys():
                mdict["calib"].append(self.calib[diag])
                # Needs to be transposed to avoid numpy error related to the different channel count of the individual diagnostics
                mdict["calib_mat"].append(self.calib_mat[diag].T)
                mdict["std_dev_mat"].append(self.std_dev_mat[diag].T)
                mdict["rel_dev"].append(self.rel_dev[diag])
                mdict["sys_dev"].append(self.sys_dev[diag])
                mdict["masked_time_points"].append(self.masked_time_points[diag])
                mdict["calib_diags"].append(diag)
            mdict["calib"] = np.array(mdict["calib"])
            mdict["calib_mat"] = np.array(mdict["calib_mat"])
            mdict["std_dev_mat"] = np.array(mdict["std_dev_mat"])
            mdict["rel_dev"] = np.array(mdict["rel_dev"])
            mdict["sys_dev"] = np.array(mdict["sys_dev"])
            mdict["masked_time_points"] = np.array(mdict["masked_time_points"])
            mdict["calib_diags"] = np.array(mdict["calib_diags"])
        mdict["freq_weights"] = self.weights["freq"]
        mdict["ray_weights"] = self.weights["ray"]
        if(quasi_linear_beam is not None):
            try:
                mdict["dist_rhot_prof"] = quasi_linear_beam.rhot
                mdict["dist_rhop_prof"] = quasi_linear_beam.rhop
                mdict["dist_PW_prof"] = quasi_linear_beam.PW * 1.e6
                mdict["dist_j_prof"] = quasi_linear_beam.j * 1.e6
                mdict["dist_PW_tot"] = quasi_linear_beam.PW_tot * 1.e6
                mdict["dist_j_tot"] = quasi_linear_beam.j_tot * 1.e6
                mdict["dist_rhot_1D_profs"] = dist_obj.rhot_1D_profs
                mdict["dist_rhop_1D_profs"] = dist_obj.rhop_1D_profs
                mdict["dist_Te_init"] = dist_obj.Te_init
                mdict["dist_ne_init"] = dist_obj.ne_init
                f = dist_obj.f
                mdict["dist_u"] = dist_obj.u
                mdict["dist_pitch"] = dist_obj.pitch
                mdict["dist_f"] = f
            except Exception as e:
                print("Failed to write dist to mat")
                print(e)
        if(linear_beam is not None):
            try:
                mdict["wave_rhot_prof"] = linear_beam.rhot
                mdict["wave_rhop_prof"] = linear_beam.rhop
                mdict["wave_PW_prof"] = linear_beam.PW * 1.e6
                mdict["wave_j_prof"] = linear_beam.j * 1.e6
                mdict["wave_PW_tot"] = linear_beam.PW_tot * 1.e6
                mdict["wave_j_tot"] = linear_beam.j_tot * 1.e6
                mdict["wave_PW_beam"] = linear_beam.PW_beam
                mdict["wave_j_beam"] = linear_beam.j_beam
                for key in linear_beam.rays[0][0].keys():
                    mdict["wave_" + key] = []
                    for ibeam in range(len(linear_beam.rays)):
                        mdict["wave_" + key].append([])
                        for iray in range(len(linear_beam.rays[ibeam])):
                            mdict["wave_" + key][-1].append(linear_beam.rays[ibeam][iray][key])
            except Exception as e:
                print("Failed to write wave to mat")
                print(e)
        try:
            savemat(filename, mdict, appendmat=False)
            print("Successfully created: ", filename)
        except TypeError as e:
            print("Failed to save to .mat")
            print(e)
            print(mdict)
            for key in mdict.keys():
                if(ndarray_check_for_None(mdict[key])): # Recursive seach of each element for None
                    print("Entry for " + key + "is None or contains None")
                    print(mdict[key])

    def extract_field(self, field):
        if(field=="Trad"):
            x = self.resonance["rhop_cold"]
            y = self.Trad
        elif(field =="RayXRz"):
            x = ndarray_math_operation(self.ray["xX"]**2 + self.ray["yX"]**2, np.sqrt)
            y = self.ray["zX"]
        elif(field =="RayORz"):
            x = ndarray_math_operation(self.ray["xO"]**2 + self.ray["yO"]**2, np.sqrt)
            y = self.ray["zO"]
        elif(field =="RayXxy"):
            x = self.ray["xX"]
            y = self.ray["yX"]
        elif(field =="RayOxy"):
            x = self.ray["xO"]
            y = self.ray["yO"]
        return x,y
