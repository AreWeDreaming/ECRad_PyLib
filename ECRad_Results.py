'''
Created on Dec 17, 2015

@author: sdenk
'''
import numpy as np
np.set_printoptions(threshold=10)
import os
from Global_Settings import globalsettings
from scipy.io import savemat, loadmat
from scipy import constants as cnst
from ECRad_Config import ECRadConfig
from ECRad_Scenario import ECRadScenario
from Ndarray_Helper import ndarray_math_operation, ndarray_check_for_None


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
        for mode in ["X", "O"]:
            self.ray["s" + mode] = []
            self.ray["x" + mode] = []
            self.ray["y" + mode] = []
            self.ray["z" + mode] = []
            self.ray["Nx" + mode] = []
            self.ray["Ny" + mode] = []
            self.ray["Nz" + mode] = []
            self.ray["Bx" + mode] = []
            self.ray["By" + mode] = []
            self.ray["Bz" + mode] = []
            self.ray["rhop" + mode] = []
            self.ray["H" + mode] = []
            self.ray["N" + mode] = []
            self.ray["Nc" + mode] = []
            self.ray["X" + mode] = []
            self.ray["Y" + mode] = []
            self.ray["theta" + mode] = []
            self.ray["BPD" + mode] = []
            self.ray["BPD_second" + mode] = []
            self.ray["em" + mode] = []
            self.ray["em_second" + mode] = []
            self.ray["ab" + mode] = []
            self.ray["ab_second" + mode] = []
            self.ray["T" + mode] = []
            self.ray["T_second" + mode] = []
            self.ray["Te" + mode] = []
            self.ray["ne" + mode] = []
            self.ray["v_g_perp" + mode] = []
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
        self.dist_obj = None
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
        Trad_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", Tradfilename))
        sres_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "sres.dat"))
        self.Trad.append(Trad_file.T[1])
        self.tau.append(Trad_file.T[2])
        if(not self.Config.extra_output):
            self.resonance["s_cold"].append(sres_file.T[0])
            self.resonance["R_cold"].append(sres_file.T[1])
            self.resonance["z_cold"].append(sres_file.T[2])
            self.resonance["rhop_cold"].append(sres_file.T[3])
            return
        if(self.Config.considered_modes == 3):
            XTrad_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "X_" + Tradfilename))
            OTrad_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "O_" + Tradfilename))
            self.XTrad.append(XTrad_file.T[1])
            self.OTrad.append(OTrad_file.T[1])
            self.Xtau.append(XTrad_file.T[2])
            self.Otau.append(OTrad_file.T[2])
            self.X_mode_frac.append(XTrad_file.T[3])
        sres_rel_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "sres_rel.dat"))
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
        Trad_comp_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", Trad_old_name))
        Ich_folder = os.path.join(self.Config.scratch_dir, "ECRad_data", Ich_folder)
        # Ray_folder = os.path.join(self.Config.scratch_dir, "ECRad_data", "ray")
        # Append new empty list for this time point
        for mode in self.modes:
            for key in self.resonance:
                if(key.endswith(mode)):
                    self.resonance[key].append([])
            for key in self.ray:
                if(key.endswith(mode)):
                    self.ray[key].append([])
            for key in self.BPD:
                if(key.endswith(mode)):
                    self.BPD[key].append([])
        # Now add los info into this list for each channel
        if(np.isscalar(Trad_file.T[0])):
            N_ch = 1
        else:
            N_ch = len(Trad_file.T[0])
        for i in range(N_ch):
            for mode in self.modes:
                try:
                    BDOP = np.loadtxt(os.path.join(Ich_folder, "BPD" + mode + "{0:03d}.dat".format(i + 1)))
                except IOError:
                    raise IndexError("Failed to load BPD")  # Raise this as an Index error to communicate that results no l
                if(len(BDOP) > 0):
                    self.BPD["rhop" + mode][-1].append(BDOP.T[0])
                    self.BPD["BPD" + mode][-1].append(BDOP.T[1])
                    self.BPD["BPD_second" + mode][-1].append(BDOP.T[2])
                    if(self.Config.N_ray > 1):
                        for key in self.ray:
                            if(key.endswith(mode)):
                                self.ray[key][-1].append([])
                        try:
                            Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_{2:s}.dat".format(1, i + 1, mode)))
                        except IOError:
                            raise IndexError("Failed to load Ray")
                        if(len(Ray_file) > 0):
                            for i_ray in range(self.Config.N_ray):
                                if(i_ray >= 1 and not os.path.isfile(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_{2:s}.dat".format(i_ray + 1, i + 1, mode)))):
                                    too_few_rays = True
                                    break
                                # Ray_file = np.loadtxt(os.path.join(Ray_folder, "Ray{0:03n}ch{1:03n}_X.dat".format(i_ray + 1, i + 1)).replace(",", ""))
                                try:
                                    Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_{2:s}.dat".format(i_ray + 1, i + 1, mode)))
                                except IOError:
                                    raise IndexError("Failed to load Ray")
                                self.ray["s" + mode][-1][-1].append(Ray_file.T[0])
                                self.ray["x" + mode][-1][-1].append(Ray_file.T[1])
                                self.ray["y" + mode][-1][-1].append(Ray_file.T[2])
                                self.ray["z" + mode][-1][-1].append(Ray_file.T[3])
                                self.ray["rhop" + mode][-1][-1].append(Ray_file.T[4])
                                self.ray["BPD" + mode][-1][-1].append(Ray_file.T[5])
                                self.ray["BPD_second" + mode][-1][-1].append(Ray_file.T[6])
                                self.ray["em" + mode][-1][-1].append(Ray_file.T[7])
                                self.ray["em_second" + mode][-1][-1].append(Ray_file.T[8])
                                self.ray["ab" + mode][-1][-1].append(Ray_file.T[9])
                                self.ray["ab_second" + mode][-1][-1].append(Ray_file.T[10])
                                self.ray["T" + mode][-1][-1].append(Ray_file.T[11])
                                self.ray["T_second" + mode][-1][-1].append(Ray_file.T[12])
                                self.ray["ne" + mode][-1][-1].append(Ray_file.T[13])
                                self.ray["Te" + mode][-1][-1].append(Ray_file.T[14])
                                self.ray["H" + mode][-1][-1].append(Ray_file.T[15])
                                self.ray["N" + mode][-1][-1].append(Ray_file.T[16])
                                self.ray["Nc" + mode][-1][-1].append(Ray_file.T[17])
                                self.ray["theta" + mode][-1][-1].append(Ray_file.T[18])
                                self.ray["Nx" + mode][-1][-1].append(Ray_file.T[19])
                                self.ray["Ny" + mode][-1][-1].append(Ray_file.T[20])
                                self.ray["Nz" + mode][-1][-1].append(Ray_file.T[21])
                                self.ray["Bx" + mode][-1][-1].append(Ray_file.T[22])
                                self.ray["By" + mode][-1][-1].append(Ray_file.T[23])
                                self.ray["Bz" + mode][-1][-1].append(Ray_file.T[24])
                                self.ray["v_g_perp" + mode][-1][-1].append(Ray_file.T[25])
                                itime = np.argmin(np.abs(self.Scenario.plasma_dict["time"] - time))
                                omega = 2.0 * np.pi * self.Scenario.ray_launch[itime]["f"][i]
                                self.ray["Y" + mode][-1][-1].append(cnst.e * np.sqrt(self.ray["Bx" + mode][-1][-1][-1]**2 + \
                                                                                     self.ray["By" + mode][-1][-1][-1]**2 + \
                                                                                     self.ray["Bz" + mode][-1][-1][-1]**2) / \
                                                                                     (cnst.m_e * omega))
                                self.ray["X" + mode][-1][-1].append(cnst.e**2 * self.ray["ne" + mode][-1][-1][-1] / \
                                                              (cnst.m_e * cnst.epsilon_0 * omega**2))
                                                                                
                    else:
                        try:
                            Ray_file = np.loadtxt(os.path.join(Ich_folder, "BPD_ray{0:03d}ch{1:03d}_{2:s}.dat".format(1, i + 1, mode)))
                        except (IOError,ValueError) as e:
                            print(e)
                            raise IndexError("Failed to load Ray")  # Raise this as an Index error to communicate that results no
                        if(len(Ray_file) > 0):
                            self.ray["s" + mode][-1].append(Ray_file.T[0])
                            self.ray["x" + mode][-1].append(Ray_file.T[1])
                            self.ray["y" + mode][-1].append(Ray_file.T[2])
                            self.ray["z" + mode][-1].append(Ray_file.T[3])
                            self.ray["rhop" + mode][-1].append(Ray_file.T[4])
                            self.ray["BPD" + mode][-1].append(Ray_file.T[5])
                            self.ray["BPD_second" + mode][-1].append(Ray_file.T[6])
                            self.ray["em" + mode][-1].append(Ray_file.T[7])
                            self.ray["em_second" + mode][-1].append(Ray_file.T[8])
                            self.ray["ab" + mode][-1].append(Ray_file.T[9])
                            self.ray["ab_second" + mode][-1].append(Ray_file.T[10])
                            self.ray["T" + mode][-1].append(Ray_file.T[11])
                            self.ray["T_second" + mode][-1].append(Ray_file.T[12])
                            self.ray["ne" + mode][-1].append(Ray_file.T[13])
                            self.ray["Te" + mode][-1].append(Ray_file.T[14])
                            self.ray["H" + mode][-1].append(Ray_file.T[15])
                            self.ray["N" + mode][-1].append(Ray_file.T[16])
                            self.ray["Nc" + mode][-1].append(Ray_file.T[17])
                            self.ray["theta" + mode][-1].append(Ray_file.T[18])
                            self.ray["Nx" + mode][-1].append(Ray_file.T[19])
                            self.ray["Ny" + mode][-1].append(Ray_file.T[20])
                            self.ray["Nz" + mode][-1].append(Ray_file.T[21])
                            self.ray["Bx" + mode][-1].append(Ray_file.T[22])
                            self.ray["By" + mode][-1].append(Ray_file.T[23])
                            self.ray["Bz" + mode][-1].append(Ray_file.T[24])
                            self.ray["v_g_perp" + mode][-1].append(Ray_file.T[25])
                            itime = np.argmin(np.abs(self.Scenario.plasma_dict["time"] - time))
                            omega = 2.0 * np.pi * self.Scenario.ray_launch[itime]["f"][i]
                            self.ray["Y" + mode][-1].append(cnst.e * np.sqrt(self.ray["Bx" + mode][-1][-1]**2 + \
                                                                                     self.ray["By" + mode][-1][-1]**2 + \
                                                                                     self.ray["Bz" + mode][-1][-1]**2) / \
                                                                                     (cnst.m_e * omega))
                            self.ray["X" + mode][-1].append(cnst.e**2 * self.ray["ne" + mode][-1][-1] / \
                                                          (cnst.m_e * cnst.epsilon_0 * omega**2))
        self.weights["ray"].append(np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "ray_weights.dat"), ndmin=2))
        self.weights["freq"].append(np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "freq_weights.dat"), ndmin=2))
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
            XTrad_comp_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "X_" + Trad_old_name))
            OTrad_comp_file = np.loadtxt(os.path.join(self.Config.scratch_dir, "ECRad_data", "O_" + Trad_old_name))
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
        for key in self.resonance:
            self.resonance[key] = np.array(self.resonance[key])
        for key in self.ray_launch:
            self.ray_launch[key] = np.array(self.ray_launch[key])
        for key in self.ray:
            self.ray[key] = np.array(self.ray[key])
        for key in self.BPD:
            self.BPD[key] = np.array(self.BPD[key])
        if(not self.init):
            self.init = True
        for key in self.weights:
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
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["calib_diags", "time"]
        at_least_2d_keys = ["Trad", "Trad_comp", "tau", "tau_comp", \
                            "XTrad", "OTrad", "Xtau", "Otau", "X_mode_frac", \
                            "XTrad_comp", "OTrad_comp", "Xtau_comp", "Otau_comp", "X_mode_frac_comp", \
                             "ne", "calib", "rel_dev", "sys_dev", "Te", "rhop", "masked_time_points", \
                             "ECE_rhop", "ECE_dat", "eq_data", \
                             "eq_R", "eq_z", "launch_diag_name", "launch_f", "launch_df", "launch_R", "launch_phi", \
                             "launch_z", "launch_tor_ang" , "launch_pol_ang", "launch_dist_focus", \
                             "launch_width", "eq_special", "eq_special_complete"  ] + list(self.resonance)
        at_least_3d_keys = list(self.BPD)
        at_least_3d_keys[at_least_3d_keys.index("rhopX")] = "BPDrhopX"
        at_least_3d_keys[at_least_3d_keys.index("rhopO")] = "BPDrhopO"
        at_least_3d_keys += list(self.ray) + ["ray_BPDX", "ray_BPDO", "ray_BPD_secondX", "ray_BPD_secondO", "ray_emX", "ray_emO", \
                                               "ray_abX", "ray_abO", "ray_TX", "ray_TO", "ray_em_secondX", "ray_em_secondO", \
                                               "ray_ab_secondX", "ray_ab_secondO", "ray_T_secondX", "ray_T_secondO", "freq_weights", \
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
        for key in mdict:
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
        if("comment" in mdict):
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
            if("calib_diags" in mdict  and len(mdict["calib"]) > 0):
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
                for key in self.weights:
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
                for mode in self.modes:
                    self.ray["s" + mode] = mdict["s" + mode]
                    self.ray["x" + mode] = mdict["x" + mode]
                    self.ray["y" + mode] = mdict["y" + mode]
                    self.ray["z" + mode] = mdict["z" + mode]
                    self.ray["H" + mode] = mdict["H" + mode]
                    self.ray["N" + mode] = mdict["N" + mode]
                    self.ray["Nc" + mode] = mdict["Nc" + mode]
                    try:
                        self.ray["rhop" + mode] = mdict["rhop" + mode]
                        self.ray["Te" + mode] = mdict["Te" + mode]
                        self.ray["BPD" + mode] = mdict["ray_BPD" + mode]
                        self.ray["BPD_second" + mode] = mdict["ray_BPD_second" + mode]
                    except KeyError:
                        print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                    try:
                        self.ray["X" + mode] = mdict["X" + mode]
                        self.ray["Y" + mode] = mdict["Y" + mode]
                    except KeyError:
                        print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                    try:
                        self.ray["ne" + mode] = mdict["ne" + mode]
                    except KeyError:
                        print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                    try:
                        self.ray["em" + mode] = mdict["ray_em" + mode]
                        self.ray["em_second" + mode] = mdict["ray_em_second" + mode]
                        self.ray["ab" + mode] = mdict["ray_ab" + mode]
                        self.ray["ab_second" + mode] = mdict["ray_ab_second" + mode]
                        self.ray["T" + mode] = mdict["ray_T" + mode]
                        self.ray["T_second" + mode] = mdict["ray_T_second" + mode]
                        self.ray["Te" + mode] = mdict["Te" + mode]
                    except KeyError:
                        print("Warning! Some BPD Ray information mssing in loaded .mat file. Is this an old file?")
                    try:
                        self.ray["theta" + mode] = mdict["theta" + mode]
                    except KeyError:
                        print("Warning! theta not found in loaded .mat file. Is this an old file?")
                    try:
                        self.ray["ne" + mode] = mdict["ne" + mode]
                        self.ray["Nx" + mode] = mdict["Nx" + mode]
                        self.ray["Ny" + mode] = mdict["Ny" + mode]
                        self.ray["Nz" + mode] = mdict["Nz" + mode]
                        self.ray["Bx" + mode] = mdict["Bx" + mode]
                        self.ray["By" + mode] = mdict["By" + mode]
                        self.ray["Bz" + mode] = mdict["Bz" + mode]
                        self.ray["v_g_perp" + mode] = mdict["v_g_perp" + mode]
                    except KeyError:
                        print("Warning! v_g_perp not found in loaded .mat file. Is this an old file?")
            except KeyError:
                print("Warning! No Ray information in loaded .mat file. Is this an old file?")
        self.init = True
        return True

    def to_mat_file(self, ext_filename=None, comment=None, quasi_linear_beam=None, dist_obj=None, linear_beam=None):
        ed = 1
        diag_str = ""
        for key in self.Scenario.used_diags_dict:
            diag_str += key
        if(ext_filename is not None):
            filename = ext_filename
        elif(len(self.calib) > 0):
            filename = os.path.join(self.Config.working_dir, "ECRad_{0:5d}_{1:s}_w_calib_ed{2:d}.mat".format(self.Scenario.shot, diag_str, ed))
        else:
            filename = os.path.join(self.Config.working_dir, "ECRad_{0:5d}_{1:s}_ed{2:d}.mat".format(self.Scenario.shot, diag_str, ed))
        while(os.path.exists(filename) and ext_filename is None):
            ed += 1
            if(len(self.calib) > 0):
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
            mdict["ECRadGUI_git_tag"] =  np.loadtxt(os.path.join(globalsettings.ECRadGUIRoot, "id"),dtype=np.str)
        except IOError:
            mdict["ECRadGUI_git_tag"] = "Unknown"
        try:
            mdict["ECRadPylib_git_tag"] = np.loadtxt(os.path.join(globalsettings.ECRadPylibRoot, "id"),dtype=np.str)
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
        if(self.Config.extra_output):
            for mode in self.modes:
                mdict["BPD" + mode] = self.BPD["BPD" + mode]
                mdict["BPD_second" + mode] = self.BPD["BPD_second" + mode]
                mdict["BPDrhop" + mode] = self.BPD["rhop" + mode]
                mdict["s" + mode] = self.ray["s" + mode]
                mdict["x" + mode] = self.ray["x" + mode]
                mdict["y" + mode] = self.ray["y" + mode]
                mdict["z" + mode] = self.ray["z" + mode]
                mdict["Nx" + mode] = self.ray["Nx" + mode]
                mdict["Ny" + mode] = self.ray["Ny" + mode]
                mdict["Nz" + mode] = self.ray["Nz" + mode]
                mdict["Bx" + mode] = self.ray["Bx" + mode]
                mdict["By" + mode] = self.ray["By" + mode]
                mdict["Bz" + mode] = self.ray["Bz" + mode]
                mdict["rhop" + mode] = self.ray["rhop" + mode]
                mdict["H" + mode] = self.ray["H" + mode]
                mdict["N" + mode] = self.ray["N" + mode]
                mdict["Nc" + mode] = self.ray["Nc" + mode]
                mdict["theta" + mode] = self.ray["theta" + mode]
                mdict["v_g_perp" + mode] = self.ray["v_g_perp" + mode]
                mdict["X" + mode] = self.ray["X" + mode]
                mdict["Y" + mode] = self.ray["Y" + mode]
                mdict["Te" + mode] = self.ray["Te" + mode]
                mdict["ne" + mode] = self.ray["ne" + mode]
                mdict["ray_BPD" + mode] = self.ray["BPD" + mode]
                mdict["ray_BPD_second" + mode] = self.ray["BPD_second" + mode]
                mdict["ray_em" + mode] = self.ray["em" + mode]
                mdict["ray_em_second" + mode] = self.ray["em_second" + mode]
                mdict["ray_ab" + mode] = self.ray["ab" + mode]
                mdict["ray_ab_second" + mode] = self.ray["ab_second" + mode]
                mdict["ray_T" + mode] = self.ray["T" + mode]
                mdict["ray_T_second" + mode] = self.ray["T_second" + mode]
        if(len(self.calib) > 0):
            mdict["calib"] = []
            mdict["calib_mat"] = []
            mdict["std_dev_mat"] = []
            mdict["rel_dev"] = []
            mdict["sys_dev"] = []
            mdict["masked_time_points"] = []
            mdict["calib_diags"] = []
            for diag in self.calib:
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
                for key in linear_beam.rays[0][0]:
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
            for key in mdict:
                if(ndarray_check_for_None(mdict[key])): # Recursive seach of each element for None
                    print("Entry for " + key + "is None or contains None")
                    print(mdict[key])

    def extract_field(self, field):
        if(field=="Trad"):
            if(self.Config.extra_output):
                x = [self.resonance["rhop_cold"],self.resonance["rhop_warm"]]
            else:
                x = [self.resonance["rhop_cold"],self.resonance["rhop_cold"]]
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
        
        
    
if(__name__ == "__main__"):
    res = ECRadResults()
    res.from_mat_file("/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed2.mat")
    res.Scenario.profile_dimension = 2
    res.to_mat_file("/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed3.mat")
    
    
    
    
    
    
