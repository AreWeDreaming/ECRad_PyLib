'''
Created on Dec 7, 2020
@author: Severin Denk
Restructuring of the old ECRadresults class. Uses the OMFit style approach where the parent class is a dictionary
'''

from ecrad_pylib.Distribution_Classes import Distribution
import numpy as np
np.set_printoptions(threshold=10)
import os
from ecrad_pylib.Global_Settings import globalsettings
from scipy.io import loadmat
from scipy import constants as cnst
from ecrad_pylib.ECRad_Config import ECRadConfig
from ecrad_pylib.ECRad_Scenario import ECRadScenario
from netCDF4 import Dataset


class ECRadResults(dict):
    def __init__(self, lastused=False):
        self.result_keys = ["Trad", "resonance", "ray", "BPD", "weights", "dimensions"]
        self.shapes = {}
        self.units = {}
        self.scales = {}
        self.xaxis_link = {}
        self.legend_entries = {}
        self.labels = {}
        self.sub_keys = {}
        self.graph_style = {}
        self.failed_keys = {}
        # Mode order is mixed, X, O
        self.sub_keys["Trad"] = ["Trad", "tau", "T", \
                                 "Trad_second", "tau_second", "T_second"]
        self.shapes["Trad"] = ["N_time", "N_mode_mix", "N_ch"]
        self.units["Trad"] = {"Trad":"keV", "tau":"", "T":"", \
                              "Trad_second":"keV", "tau_second":"", "T_second":""}
        self.scales["Trad"] = {"Trad":1.e-3, "tau":1.0, "T":1.0, \
                              "Trad_second":1.e-3, "tau_second":1.0, "T_second":1.0}
        self.xaxis_link["Trad"] = ["resonance" ,"Trad"]
        self.legend_entries["Trad"] = {"Trad":r"$T_" + globalsettings.mathrm + r"{rad}$", "tau":r"$\tau$", "T":r"$T$", \
                                       "Trad_second":r"$T_" + globalsettings.mathrm + r"{rad,2nd\,model}$", \
                                       "tau_second":r"$\tau_" + globalsettings.mathrm + r"{2nd\,model}$", "T_second":r"$T_" + globalsettings.mathrm + r"{2nd\,model}$"}
        self.labels["Trad"] = {"Trad":r"$T_" + globalsettings.mathrm + r"{rad}$", "tau":r"$\tau$", "T":r"$T$", \
                              "Trad_second":r"$T_" + globalsettings.mathrm + r"{rad}$", "tau_second":r"$\tau$", "T_second":r"$T$"}
        self.graph_style["Trad"] = "point"
        self.sub_keys["resonance"] = ["s_cold", "R_cold", "z_cold", \
                                      "rhop_cold", "rhot_cold", "s_warm", "R_warm", \
                                      "z_warm", "rhop_warm", "rhot_warm", \
                                      "s_warm_second", "R_warm_second", \
                                      "z_warm_second", "rhop_warm_second", \
                                      "rhot_warm_second"]
        # You want the channel as the inner most index since you want to plot as a function of channel
        self.shapes["resonance"] = ["N_time", "N_mode_mix", "N_ch"]
        self.units["resonance"] = {}
        self.scales["resonance"] = {}
        for sub_key in self.sub_keys["resonance"]:
            if(sub_key.startswith("rho")):
                self.units["resonance"][sub_key] = ""
            else:
                self.units["resonance"][sub_key] = "m"
            self.scales["resonance"][sub_key] = 1.0
        self.xaxis_link["resonance"] = ["Trad", "resonance"]
        self.legend_entries["resonance"] = {"s_cold":r"$s_" + globalsettings.mathrm + r"{cold}$", \
                                   "R_cold":r"$R_" + globalsettings.mathrm + r"{cold}$", \
                                   "z_cold":r"$z_" + globalsettings.mathrm + r"{cold}$", \
                                   "rhop_cold":r"$\rho_" + globalsettings.mathrm + r"{pol,cold}$", \
                                   "rhot_cold":r"$\rho_" + globalsettings.mathrm + r"{tor,cold}$", \
                                   "s_warm":r"$s_" + globalsettings.mathrm + r"{warm}$", \
                                   "R_warm":r"$R_" + globalsettings.mathrm + r"{warm}$", \
                                   "z_warm":r"$z_" + globalsettings.mathrm + r"{warm}$", \
                                   "rhop_warm":r"$\rho_" + globalsettings.mathrm + r"{pol,warm}$", \
                                   "rhot_warmd":r"$\rho_" + globalsettings.mathrm + r"{tor,warm}$", \
                                   "s_warm_second":r"$s_" + globalsettings.mathrm + r"{warm,\,2nd\,model}$", \
                                   "R_warm_second":r"$R_" + globalsettings.mathrm + r"{warm,\,2nd\,model}$", \
                                   "z_warm_second":r"$z_" + globalsettings.mathrm + r"{warm,\,2nd\,model}$", \
                                   "rhop_warm_second":r"$\rho_" + globalsettings.mathrm + r"{pol,warm,\,2nd\,model}$", \
                                   "rhot_warm_second":r"$\rho_" + globalsettings.mathrm + r"{tor,warm,\,2nd\,model}$"}
        self.labels["resonance"] = {"s_cold":r"$s_" + globalsettings.mathrm + r"{cold}$", \
                                   "R_cold":r"$R_" + globalsettings.mathrm + r"{cold}$", \
                                   "z_cold":r"$z_" + globalsettings.mathrm + r"{cold}$", \
                                   "rhop_cold":r"$\rho_" + globalsettings.mathrm + r"{pol,cold}$", \
                                   "rhot_cold":r"$\rho_" + globalsettings.mathrm + r"{tor,cold}$", \
                                   "s_warm":r"$s_" + globalsettings.mathrm + r"{warm}$", \
                                   "R_warm":r"$R_" + globalsettings.mathrm + r"{warm}$", \
                                   "z_warm":r"$z_" + globalsettings.mathrm + r"{warm}$", \
                                   "rhop_warm":r"$\rho_" + globalsettings.mathrm + r"{pol,warm}$", \
                                   "rhot_warmd":r"$\rho_" + globalsettings.mathrm + r"{tor,warm}$", \
                                   "s_warm_second":r"$s_" + globalsettings.mathrm + r"{warm}$", \
                                   "R_warm_second":r"$R_" + globalsettings.mathrm + r"{warm}$", \
                                   "z_warm_second":r"$z_" + globalsettings.mathrm + r"{warm}$", \
                                   "rhop_warm_second":r"$\rho_" + globalsettings.mathrm + r"{pol,warm}$", \
                                   "rhot_warm_second":r"$\rho_" + globalsettings.mathrm + r"{tor,warm}$"}
        self.graph_style["resonance"] = "point"
        self.sub_keys["ray"] = ["s", "x", "y",  "R", "z", \
                                "Nx", "Ny", "Nz", \
                                "Bx", "By", "Bz", \
                                "H", "N", "Nc", \
                                "X", "Y", "rhop", "rhot" ,\
                                "Te", "ne", \
                                "theta", "BPD", \
                                "BPD_second", 
                                "Trad", "Trad_second", "em", \
                                "em_second", "ab", \
                                "ab_second", \
                                "T", "T_second", \
                                "v_g_perp"]
        self.shapes["ray"] = ["N_time", "N_ch", "N_mode", "N_ray", "N_LOS"]
        self.legend_entries["ray"] = {"s":r"$s$", "x":r"$x$", "y":r"$y$",  "R":r"$R$", "z":r"$z$", \
                                      "Nx":r"$N_" + globalsettings.mathrm + r"{x}$", "Ny":r"$N_" + globalsettings.mathrm + r"{y}$", \
                                      "Nz":r"$N_" + globalsettings.mathrm + r"{z}$", "Bx":r"$B_" + globalsettings.mathrm + r"{x}$", \
                                      "By":r"$B_" + globalsettings.mathrm + r"{y}$", "Bz":r"$B_" + globalsettings.mathrm + r"{z}$", \
                                      "H":r"Hamiltonian", "N":r"$N_" + globalsettings.mathrm + r"{ray}$", \
                                      "Nc":r"$N_" + globalsettings.mathrm + r"{disp}$", "X":r"Stix Parameter $X$", \
                                      "Y":r"Stix Parameter $Y$", "rhop":r"$\rho_" + globalsettings.mathrm + r"{pol}$", \
                                      "rhot":r"$\rho_" + globalsettings.mathrm + r"{tor}$", "Te":r"$T_" + globalsettings.mathrm + r"{e}$", \
                                      "ne":r"$n_" + globalsettings.mathrm + r"{e}$", "theta":r"$\theta$", \
                                      "BPD":"BPD", "BPD_second":"BPD$_" + globalsettings.mathrm + r"{2nd\,model}$", \
                                      "Trad":r"$T_" + globalsettings.mathrm + r"{rad}$", "Trad_second":r"$T_" + globalsettings.mathrm + r"{rad,\,2nd\,model}}$", \
                                      "em":r"$j$", \
                                      "em_second":r"$j_{" + globalsettings.mathrm + r"{2nd\,model}}$", "ab":r"$\alpha$", \
                                      "ab_second":r"$\alpha_{" + globalsettings.mathrm + r"{2nd\,model}}$", "T":r"$\mathcal{T}$", \
                                      "T_second":r"$\mathcal{T}_{" + globalsettings.mathrm + r"{2nd\,model}}$", "v_g_perp":r"$v_{" + globalsettings.mathrm + r"{g},\perp}$"}
        self.labels["ray"] = {"s":r"$s$", "x":r"$x$", "y":r"$y$",  "R":r"$R$", "z":r"$z$", \
                              "Nx":r"$N_" + globalsettings.mathrm + r"{x}$", "Ny":r"$N_" + globalsettings.mathrm + r"{y}$", \
                             "Nz":r"$N_" + globalsettings.mathrm + r"{z}$", "Bx":r"$B_" + globalsettings.mathrm + r"{x}$", \
                             "By":r"$B_" + globalsettings.mathrm + r"{y}$", "Bz":r"$B_" + globalsettings.mathrm + r"{z}$", \
                             "H":r"Hamiltonian", "N":r"$N_" + globalsettings.mathrm + r"{ray}$", \
                             "Nc":r"$N_" + globalsettings.mathrm + r"{disp}$", "X":r"Stix Parameter $X$", \
                             "Y":r"Stix Parameter $Y$", "rhop":r"$\rho_" + globalsettings.mathrm + r"{pol}$", \
                             "rhot":r"$\rho_" + globalsettings.mathrm + r"{tor}$", "Te":r"$T_" + globalsettings.mathrm + r"{e}$", \
                             "ne":r"$n_" + globalsettings.mathrm + r"{e}$", "theta":r"$\theta$", \
                             "BPD":"BPD", "BPD_second":"BPD", \
                             "Trad":r"$T_" + globalsettings.mathrm + r"{rad}$", "Trad_second":r"$T_" + globalsettings.mathrm + r"{rad}$", \
                             "em":r"$j_\omega$", \
                             "em_second":r"$j_\omega$", "ab":r"$\alpha_\omega$", \
                             "ab_second":r"$\alpha_\omega$", "T":r"$\mathcal{T}_\omega$", \
                             "T_second":r"$\mathcal{T}_\omega$", "v_g_perp":r"$v_{" + globalsettings.mathrm + r"{g},\perp}$"}
        self.units["ray"] = {"s":"m", "x":"m", "y":"m", "R":"m", "z":"m", \
                             "Nx":"", "Ny":"", "Nz":"", \
                             "Bx":"T", "By":"T", "Bz":"T", \
                             "H":"", "N":"", "Nc":"", \
                             "X":"", "Y":"", \
                             "rhop":"", "rhot":"",\
                             "Te":r"keV", "ne":r"$10^{19}$m$^{-3}$", \
                             "theta":r"$^\circ$", "BPD":r"m$^{-1}$", \
                             "BPD_second":r"m$^{-1}$", \
                             "Trad":r"keV", "Trad_second":r"keV", \
                             "em":r"nW m$^{-3}$", \
                             "em_second":r"nW m$^{-3}$", "ab":r"m$^{-1}$", \
                             "ab_second":r"m$^{-1}$", \
                             "T":"", "T_second":"", \
                             "v_g_perp":""}
        self.scales["ray"] = {"s":1, "x":1, "y":1, "R":1, "z":1, \
                             "Nx":1, "Ny":1, "Nz":1, \
                             "Bx":1, "By":1, "Bz":1, \
                             "H":1, "N":1, "Nc":1, \
                             "X":1, "Y":1, \
                             "rhop":1, "rhot":1, \
                             "Te":1.e-3, "ne":1.e-19, \
                             "theta":np.rad2deg(1), "BPD":1, \
                             "BPD_second":1, \
                             "Trad":1.e-3, "Trad_second":1.e-3, \
                             "em":1.e9, \
                             "em_second":1.e9, "ab":1, \
                             "ab_second":1, \
                             "T":1, "T_second":1, \
                             "v_g_perp":1.0/cnst.speed_of_light}
        self.xaxis_link["ray"] = ["ray"]
        self.graph_style["ray"] = "line"
        self.sub_keys["BPD"] = ["rhop", "rhot", "BPD", "BPD_second"]
        self.scales["BPD"] = {"rhop":1.0, "rhot":1.0, "BPD":1.0, "BPD_second":1.0}
        self.labels["BPD"] = {"rhop":r"$\rho_" + globalsettings.mathrm + r"{pol}$", "rhot":r"$\rho_" + globalsettings.mathrm + r"{tor}$", \
                             "BPD":"BPD","BPD_second":"BPD"}
        self.legend_entries["BPD"] = {"rhop":r"$\rho_" + globalsettings.mathrm + r"{pol}$", "rhot":r"$\rho_" + globalsettings.mathrm + r"{tor}$", \
                                      "BPD":"BPD","BPD_second":"BPD$_" + globalsettings.mathrm + r"{2nd\,model}$"}
        self.units["BPD"] = {"rhop":"", "rhot":"", "BPD":"m$^{-1}$", \
                              "BPD_second":"m$^{-1}$"}
        self.xaxis_link["BPD"] = ["BPD"]
        self.shapes["BPD"] = ["N_time", "N_ch", "N_mode_mix", "N_BPD"]
        self.graph_style["BPD"] = "line"
        self.sub_keys["weights"] = ["mode_frac", "mode_frac_second", "ray_weights", "freq_weights"]
        self.units["weights"] = {}
        self.scales["weights"] = {}
        self.graph_style["weights"] = "point"
        for sub_key in self.sub_keys["weights"]:
            self.units["weights"][sub_key] = r'$\%$'
            self.scales["weights"][sub_key] = 1.e2
        self.labels["weights"] = {"mode_frac":r"Mode contribution", "mode_frac_second":"Mode contribution", \
                                 "ray_weights":"ray weight","freq_weights":"frequency weight"}
        self.legend_entries["weights"] = {"mode_frac":r"Mode contribution", "mode_frac_second":"Mode contribution", \
                                          "ray_weights":"ray weight","freq_weights":"frequency weight"}
        # The weights have a lot of different shapes need to store those by sub_key
        self.shapes["mode_frac"] = ["N_time", "N_mode", "N_ch"]
        self.shapes["mode_frac_second"] = ["N_time", "N_mode", "N_ch"]
        self.shapes["ray_weights"] = ["N_time", "N_ch", "N_ray"]
        self.shapes["freq_weights"] = ["N_time", "N_ch", "N_freq"]
        self.xaxis_link["weights"] = ["weights"]
        self.graph_style["BPD"] = "line"
        self["git"] = {"ECRad":"Unknown", "GUI":"Unknown", "Pylib":"Unknown"}
        self["types"] = {}
        for key in self.result_keys:
            if(key == 'dimensions'):
                self["types"][key] = "int"
            else:
                self["types"][key] = "float"
        for key in self.units.keys():
            for sub_key in self.units[key].keys():
                if(len(self.units[key][sub_key]) > 0):
                    self.units[key][sub_key] = "[" + self.units[key][sub_key] + "]"
        self.reset(not lastused)

    def reset(self, noLoad=True, light=False):
        # Does not reset Config or Scenario
        self.status = 0
        self.edition = 0
        self.init = False
        # Edition of this result
        # Remains zero until the result is saved
        # Time of the results, should be identical to self.Scenario.plasma_dict["time"]
        # This holds the same information as the scenario
        self.modes = None
        self.comment = ""
        self["dimensions"] = {}
        self["dimensions"]["N_LOS"] = []
        self["git"] = {}
        for key in self.result_keys:
            if(key in ["dimensions"]):
                continue
            self.failed_keys[key] = []
            self[key] = {}
            if(key in self.sub_keys):
                for sub_key in self.sub_keys[key]:
                    self[key][sub_key] = []
        if(not light):
            self.Config = ECRadConfig(noLoad=noLoad)
            self.Scenario = ECRadScenario(noLoad=noLoad)
        self.data_origin = None
        
    def load(self, filename):
        if(filename is not None):
            ext = os.path.splitext(filename)[1]
            if(ext == ".mat"):
                self.from_mat(filename=filename)
            elif(ext == ".nc"):
                self.from_netcdf(filename=filename)
            else:
                print("Extension " + ext + " is unknown")
                raise(ValueError)

    def tidy_up(self, autosave=True):
        if(self.status != 0):
            return
        # Put everything into numpy arrays
        for key in self.result_keys:
            if(key in ["dimensions", "git", "types"]):
                continue
            for sub_key in self.sub_keys[key]:
                self[key][sub_key] = np.array(self[key][sub_key])
        # Autosave results
        self["git"]["ECRad"] = np.genfromtxt(os.path.join(globalsettings.ECRadRoot, "id"), dtype=str).item()
        self["git"]["GUI"] = np.genfromtxt(os.path.join(globalsettings.ECRadGUIRoot, "id"), dtype=str).item()
        self["git"]["Pylib"] = np.genfromtxt(os.path.join(globalsettings.ECRadPylibRoot, "id"), dtype=str).item()
        self.data_origin = "ECRad"
        if(autosave):
            self.autosave()
            
    def autosave(self,filename=None):
        self.to_netcdf(filename)

    def get_shape(self, key, start=None, stop=None,i_time=None, \
                  i_ch=None, i_mode=None, i_ray=None):
        shape = ()
        for dim_ref in self.shapes[key][start:stop]:
            if(np.isscalar(self["dimensions"][dim_ref])):
                shape += (self["dimensions"][dim_ref],)
            else:
                # Cannot use numpy indexing here because the time dimension 
                # will be appended as we go
                shape += (self["dimensions"][dim_ref][i_time][i_ch, i_mode, i_ray],)
        return shape
    
    def get_index_reference(self, key, sub_key, ndim, index):
        # Retrieves the value this particular index refers to
        # used in plotting to label graphs.
        # The routine formats the quantity and returns a string
        # Note that the index should have the shape corresponding to the requested quantity
        if(key != "weights"):
            dim_ref = self.shapes[key][ndim]
        else:
            dim_ref = self.shapes[sub_key][ndim]
        if(dim_ref == "N_time"):
            return r"$t = $ " + "{0:1.3f}".format(self.Scenario["time"][index[ndim]]) + " s"
        elif(dim_ref == "N_ch"):
            return r"$f = $ " + "{0:3.1f}".format(self.Scenario["diagnostic"]["f"][index[0]][index[ndim]]/1.e9)  + " GHz" 
        elif(dim_ref in ["N_mode"]):
            if(self.Config["Physics"]["considered_modes"] == 1):
                return "X-mode"
            elif(self.Config["Physics"]["considered_modes"] == 2):
                return "O-mode"
            else:
                if(index[ndim] == 0):
                    return "X-mode"
                else:
                    return "O-mode"
        elif(dim_ref == "N_mode_mix"):
            if(self.Config["Physics"]["considered_modes"] < 3):
                return ""
            else:
                if(index[ndim] == 1):
                    return "X-mode"
                elif(index[ndim] == 2):
                    return "O-mode"
                else:
                    return ""
        elif(dim_ref == "N_ray"):
            if(index[ndim] == 0):
                return ""
            else:
                return r"ray \#" + str(index[ndim] + 1)
        else:
            raise ValueError("ECRadResults.get_index_reference could unexpected dim ref " + key + " " + sub_key + " " + str(ndim))
                           
    def set_dimensions(self):
        # Sets the dimensions from Scenario and Configself["dimensions"]["N_time"] = len(self.Scenario.plasma_dict["time"])
        self["dimensions"]["N_time"] = self.Scenario["dimensions"]["N_time"]
        self["dimensions"]["N_ray"] = self.Config["Physics"]["N_ray"]
        self["dimensions"]["N_freq"] = self.Config["Physics"]["N_freq"]
        self["dimensions"]["N_BPD"] = self.Config["Numerics"]["N_BPD"]
        self["dimensions"]["N_ch"] = self.Scenario["dimensions"]["N_ch"]
        if(self.Config["Physics"]["considered_modes"] > 2):
            self["dimensions"]["N_mode"] = 2
            self["dimensions"]["N_mode_mix"] = 3
            modes_mix = ["", "X", "O"]
            modes = ["X", "O"]
        else:
            self["dimensions"]["N_mode"] = 1
            self["dimensions"]["N_mode_mix"] = 1
            modes = ["X"]
            if(self.Config["Physics"]["considered_modes"] == 2):
                modes = ["O"]
            modes_mix = [""]
        self["dimensions"]["N_ch"] = self.Scenario["dimensions"]["N_ch"]
        return modes, modes_mix

    def from_mat(self, filename):
        try:
            mdict = loadmat(filename, chars_as_strings=True, squeeze_me=True)
        except IOError as e:
            print(e)
            print("Error: " + filename + " does not exist")
            return
        self.Config.from_mat(mdict=mdict)
        self.Scenario.from_mat(mdict=mdict, load_plasma_dict=True)
        self.edition = mdict["edition"]
        # We need to do this song and dance because
        # 3D equlibria use rho tor instead of rho pol
        # This was not clearly indicated in the old
        # .mat files, but the new result files distinguish this.
        # Further cases are marked with #3D rhot
        if(self.Scenario["plasma"]["eq_dim"] == 3):
            rho = "rhot"
        else:
            rho = "rhop"
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
        modes, modes_mix = self.set_dimensions()
        for key in ["Trad"]:
            for sub_key in self.sub_keys[key]:
                self[key][sub_key] = np.zeros(self.get_shape(key))
                for i_mode, mode in enumerate(modes_mix):
                    if(sub_key == "T"):
                        self[key][sub_key] = np.exp(-self["Trad"]["tau"])
                        continue
                    elif(sub_key == "T_second"):
                        self[key][sub_key] = np.exp(-self["Trad"]["tau_second"])
                        continue
                    mdict_key = sub_key.replace("second","comp")
                    self[key][sub_key][...,i_mode,:] = mdict[mode+mdict_key].reshape(self.get_shape(key,stop=1) + \
                                                                                     self.get_shape(key,start=2)) *1.e3
        if(self["dimensions"]["N_mode"] > 1):
            mode_info_printed = False
            # mode resolved resonances are not available in .mat files
            for key in ["resonance"]:
                for sub_key in self.sub_keys[key]:
                    self[key][sub_key] = np.zeros(self.get_shape(key))
                    if(sub_key.endswith("second")):
                        formatted_key = sub_key + "ary"
                    else:
                        formatted_key = sub_key
                    #3D rhot
                    if(rho == "rhot" and sub_key.startswith("rhot")):
                        formatted_key = formatted_key.replace("rhot","rhop")
                    elif(rho == "rhot" and formatted_key.startswith("rhop")):
                        continue
                    try:
                        self[key][sub_key][...,0,:] = mdict[formatted_key].reshape(self.get_shape(key,stop=1) + \
                                                                                   self.get_shape(key,start=2))
                        if(not mode_info_printed):
                            print("INFO:: No mode specific resonances in .mat files. Using mixed modes for all resonances.")
                            mode_info_printed = True
                        for imode in range(1,3):
                            self[key][sub_key][...,imode,:] = self[key][sub_key][...,0,:]
                    except KeyError:
                        print("INFO: Couldn't load " + sub_key + " from result file")
        else:
            for key in ["resonance"]:
                for sub_key in self.sub_keys[key]:
                    if(sub_key.endswith("second")):
                        formatted_key = sub_key + "ary"
                    else:
                        formatted_key = sub_key
                    #3D rhot
                    if(rho == "rhot" and sub_key.startswith("rhot")):
                        formatted_key = formatted_key.replace("rhot","rhop")
                    elif(rho == "rhot" and formatted_key.startswith("rhop")):
                        continue
                    try:
                        self[key][sub_key] = mdict[formatted_key].reshape(self.get_shape(key))
                    except KeyError:
                        print("INFO: Couldn't load " + sub_key + " from result file")
        self["dimensions"]["N_LOS"] = np.zeros(self.get_shape("ray", 0, -1),dtype=np.int)      
        # We need to fix the shape of the mdict ray info
        key = "ray"
        for sub_key in self.sub_keys[key]:
            for mode in modes:
                if(sub_key in ["em", "ab", "T", "BPD", "Trad", "Trad_second",\
                               "em_second", "ab_second", "T_second", "BPD_second"]):
                    mdict_key = "ray_" + sub_key
                #3D rhot
                elif(sub_key == "rhot" and rho == "rhot"):
                    mdict_key = "rhop"
                elif(sub_key == "rhop" and rho == "rhot"):
                    continue
                else:
                    mdict_key = sub_key
                if(mdict_key+mode not in mdict.keys()):
                    print("Cannot find " + key + "/" + sub_key + " in .mat")
                    if(sub_key not in self.failed_keys[key]):
                        self.failed_keys[key].append(sub_key)
                    continue
                if(self["dimensions"]["N_time"] == 1):
                    mdict[mdict_key+mode] = np.expand_dims(mdict[mdict_key+mode], 0)
                if(self["dimensions"]["N_ch"] == 1):
                        mdict[mdict_key+mode] = np.expand_dims(mdict[mdict_key+mode],1)
                if(self["dimensions"]["N_ray"] == 1):
                    mdict[mdict_key+mode] = np.expand_dims(mdict[mdict_key+mode],2)
        for key in ["ray", "BPD"]:
            for sub_key in self.sub_keys[key]:
                sub_key_error_printed = False
                if(key == "BPD"):
                    mdict_key = None
                    #3D rhot
                    if(sub_key == "rhot" and self.Scenario["plasma"]["eq_dim"] == 2):
                        continue
                    elif(sub_key == "rhop" and self.Scenario["plasma"]["eq_dim"] == 3):
                        continue
                    elif(sub_key == "rhot" and self.Scenario["plasma"]["eq_dim"] == 3):
                        # The BPD axis is mislabeled for 3D equilibria. It should be rhot
                        mdict_key = "BPD" + "rhop"
                    if(mdict_key is None):
                        if(sub_key in ["rhop", "rhot"]):
                            mdict_key = "BPD" + sub_key
                        else:
                            mdict_key = sub_key
                    if(mdict_key+mode not in mdict.keys()):
                        print("INFO: Cannot load " + key + "/" + sub_key)
                        if(sub_key not in self.failed_keys[key]):
                            self.failed_keys[key].append(sub_key)
                        continue
                    self[key][sub_key] = np.zeros(self.get_shape(key))
                    for i_mode, mode in enumerate(modes):
                        self[key][sub_key][...,i_mode,:] = mdict[mdict_key+mode].reshape(self.get_shape(key,stop=-2) + \
                                                                                         (self["dimensions"]["N_BPD"],))
                else:
                    if(sub_key in ["em", "ab", "T", "BPD", \
                                   "em_second", "ab_second", "T_second", "BPD_second"]):
                        mdict_key = "ray_" + sub_key
                    #3D rhot
                    elif(sub_key == "rhot" and rho == "rhot"):
                        mdict_key = "rhop"
                    elif(sub_key == "rhop" and rho == "rhot"):
                        continue
                    else:
                        mdict_key = sub_key
                    if(mdict_key+mode not in mdict.keys()):
                        if(sub_key !="R"):
                            print("INFO: Cannot load " + key + "/" + sub_key)
                        if(sub_key not in self.failed_keys[key]):
                            self.failed_keys[key].append(sub_key)
                        continue
                    self["ray"][sub_key] = []
                    for i_time in range(self["dimensions"]["N_time"]):
                        self["ray"][sub_key].append([])
                        for i_ch in range(self["dimensions"]["N_ch"]):
                            self["ray"][sub_key][i_time].append([])
                            for i_mode, mode in enumerate(modes):
                                self["ray"][sub_key][i_time][i_ch].append([])
                                for i_ray in range(self["dimensions"]["N_ray"]):
                                    try:
                                        self["ray"][sub_key][i_time][i_ch][i_mode].append( \
                                                mdict[mdict_key+mode][i_time][i_ch][i_ray])
                                        self["dimensions"]["N_LOS"][i_time][i_ch][i_mode][i_ray] = \
                                                len(self["ray"][sub_key][i_time][i_ch][i_mode][i_ray])
                                    except IndexError:
                                        if(not sub_key_error_printed):
                                            print("INFO: Failed to load {0:s} {1:s}".format(key,sub_key))
                                            print("INFO: For time index {0:d} channel {1:d} mode index {2:d} ray {3:d}".format(
                                                    i_time, i_ch, i_mode, i_ray))
                                            sub_key_error_printed = True
                                        self["ray"][sub_key][i_time][i_ch][i_mode].append([])
                                        self["dimensions"]["N_LOS"][i_time][i_ch][i_mode][i_ray] = 0
                    # Convert to ragged np array
                    self["ray"][sub_key] = np.array(self["ray"][sub_key], dtype=np.object)
        self["weights"]["ray_weights"] = mdict["ray_weights"]
        self["weights"]["freq_weights"] = mdict["ray_weights"]
        if(self.Config["Physics"]["considered_modes"] > 2):
            self["weights"]["mode_frac"] = np.zeros(self.get_shape("mode_frac"))
            self["weights"]["mode_frac_second"] = np.zeros(self.get_shape("mode_frac_second"))
            self["weights"]["mode_frac"][...,0,:] = mdict["X_mode_frac"].reshape(self.get_shape("mode_frac",stop=1) + \
                                                                                                 self.get_shape("mode_frac",start=2))
            self["weights"]["mode_frac_second"][...,0,:] = mdict["X_mode_frac_comp"].reshape(self.get_shape("mode_frac_second",stop=1) + \
                                                                                                 self.get_shape("mode_frac_second",start=2))
            self["weights"]["mode_frac"][...,1,:] = 1 - mdict["X_mode_frac"].reshape(self.get_shape("mode_frac",stop=1) + \
                                                                                                 self.get_shape("mode_frac",start=2))
            self["weights"]["mode_frac_second"][...,1,:] = 1 - mdict["X_mode_frac_comp"].reshape(self.get_shape("mode_frac_second",stop=1) + \
                                                                                                 self.get_shape("mode_frac_second",start=2))
        else:
            self["weights"]["mode_frac"] = np.ones(self.get_shape("mode_frac"))
            self["weights"]["mode_frac_second"] = np.ones(self.get_shape("mode_frac_second"))
        self["ray"]["R"] = np.zeros(self.get_shape("ray", stop=-1), dtype=np.object)
        print("INFO: Fixing missing ray/R.")
        for itime in range(self["dimensions"]["N_time"]):
            for ich in range(self["dimensions"]["N_ch"]):
                for imode in range(self["dimensions"]["N_mode"]):
                    for iray in range(self["dimensions"]["N_ray"]):
                        if(self["dimensions"]["N_LOS"][i_time][i_ch][i_mode][i_ray] > 0):
                            self["ray"]["R"][itime,ich,imode,iray] = np.sqrt(self["ray"]["x"][itime,ich,imode,iray]**2 + \
                                                                            self["ray"]["y"][itime,ich,imode,iray]**2)
        # We fix R later so we do not need to delete it
        self.failed_keys["ray"].remove("R")
        self["git"]["ECRad"] = mdict["ECRad_git_tag"]
        self["git"]["GUI"] = mdict["ECRadGUI_git_tag"]
        self["git"]["Pylib"] = mdict["ECRadPylib_git_tag"]
        self.data_origin = filename
        self.init = True
        return True
    
    def get_default_filename_and_edition(self, scratch=False, ed=None):
        if(scratch):
            dir = self.Config["Execution"]["scratch_dir"]
        else:
            dir = self.Config["Execution"]["working_dir"]
        diag_str = ""
        for key in self.Scenario["used_diags_dict"]:
            diag_str += key
        if(ed is None):
            ed = 1
            filename = os.path.join(dir, "ECRad_{0:5d}_{1:s}_ed{2:d}.nc".format(self.Scenario["shot"], diag_str, ed))
            while(os.path.exists(filename)):
                ed += 1
                filename = os.path.join(dir, "ECRad_{0:5d}_{1:s}_ed{2:d}.nc".format(self.Scenario["shot"], diag_str, ed))
            return filename, ed
        else:
            filename = os.path.join(dir, "ECRad_Results_{0:d}.nc".format(ed))
            return filename, 0

    def to_netcdf(self, filename=None, scratch=False, ed=None):
        if(filename is not None):
            rootgrp = Dataset(filename, "w", format="NETCDF4")
        else:
            filename, self.edition = self.get_default_filename_and_edition(scratch, ed=ed)
            rootgrp = Dataset(filename, "w", format="NETCDF4")
        rootgrp.createGroup("Results")
        self.Config.to_netcdf(rootgrp=rootgrp)
        self.Scenario.to_netcdf(rootgrp=rootgrp)
        for sub_key in self["dimensions"].keys():
            if(np.isscalar(self["dimensions"][sub_key])):
                rootgrp["Results"].createDimension(sub_key, self["dimensions"][sub_key])
            else:
                rootgrp["Results"].createDimension(sub_key, None)                    
        for key in self.result_keys:
            if(key == "dimensions" or key == "ray"):
                continue
            dtype = "f8"
            if(self["types"][key] != "float"):
                dtype = "i8"
            try:
                for sub_key in self.sub_keys[key]:
                    if(len(self[key][sub_key]) == 0):
                        print("INFO: Not saving " + key + " " + sub_key + " because there is no data.")
                        continue
                    if(sub_key in self.failed_keys[key]):
                        continue
                    if(sub_key == "rhot" and self.Scenario["plasma"]["eq_dim"] == 2):
                        continue
                    elif(sub_key == "rhop" and self.Scenario["plasma"]["eq_dim"] == 3):
                        continue
                    if(key != "weights"):
                        var = rootgrp["Results"].createVariable(key + "_" + sub_key,dtype, tuple(self.shapes[key]))
                    else:
                        var = rootgrp["Results"].createVariable(key + "_" + sub_key,dtype, tuple(self.shapes[sub_key]))
                    var[:] = self[key][sub_key]
            except Exception as e:
                print(key, sub_key)
                raise e
        if(self.Config["Execution"]["extra_output"]):
            key = "ray"
            for sub_key in self.sub_keys["ray"]:
                if(sub_key in self.failed_keys[key]):
                    continue
                elif(len(self[key][sub_key]) == 0):
                    continue
                var = rootgrp["Results"].createVariable(key + "_" + sub_key,dtype, tuple(self.shapes[key]))
                for i_time in range(self["dimensions"]["N_time"]):
                    for i_ch in range(self["dimensions"]["N_ch"]):
                        for i_mode in range(self["dimensions"]["N_mode"]):
                            for i_ray in range(self["dimensions"]["N_ray"]):
                                var[i_time,i_ch,i_mode,i_ray,:] =  self[key][sub_key][i_time,i_ch,i_mode,i_ray]
            # Get the shape information of the individual LOS length into the NETCDF file
            var = rootgrp["Results"].createVariable("dimensions" + "_" + "N_LOS", "i8", self.shapes["ray"][:-1])
            var[:] = self["dimensions"]["N_LOS"]
        rootgrp["Results"].comment = self.comment
        rootgrp["Results"].edition = self.edition
        rootgrp["Results"].ECRad_git_tag = self["git"]["ECRad"]
        rootgrp["Results"].ECRadGUI_git_tag = self["git"]["GUI"]
        rootgrp["Results"].ECRadPylib_git_tag = self["git"]["Pylib"]
        rootgrp.close()
        print("Created " + filename)
        
    def from_netcdf(self, filename):
        rootgrp = Dataset(filename, "r", format="NETCDF4")
        self.Config.from_netcdf(rootgrp=rootgrp)
        self.Scenario.from_netcdf(rootgrp=rootgrp)
        for sub_key in rootgrp["Results"].dimensions.keys():
            self["dimensions"][sub_key] = rootgrp["Results"].dimensions[sub_key].size
        for key in self.result_keys:
            if(key == "dimensions" or key == "ray"):
                continue
            for sub_key in self.sub_keys[key]:
                if(key + "_" + sub_key not in rootgrp["Results"].variables.keys()):
                    print("INFO: Could not find " + key + " " + sub_key + " in the result file.")
                    self.failed_keys[key].append(sub_key)
                    continue
                if(key == "BPD"):
                    if(sub_key == "rhot" and self.Scenario["plasma"]["eq_dim"] == 2):
                        continue
                    elif(sub_key == "rhop" and self.Scenario["plasma"]["eq_dim"] == 3):
                        continue
                self[key][sub_key] = np.array(rootgrp["Results"][key + "_" + sub_key])
        if(self.Config["Execution"]["extra_output"]):
            self["dimensions"]["N_LOS"] = np.array(rootgrp["Results"]["dimensions" + "_" + "N_LOS"])
            key = "ray"
            for sub_key in self.sub_keys["ray"]:
                self["ray"][sub_key] = []
                if(key + "_" + sub_key not in rootgrp["Results"].variables.keys()):
                    print("INFO: Cannot load " + key + "/" + sub_key)
                    self.failed_keys[key].append(sub_key)
                    continue
                for i_time in range(self["dimensions"]["N_time"]):
                    self["ray"][sub_key].append([])
                    for i_ch in range(self["dimensions"]["N_ch"]):
                        self["ray"][sub_key][i_time].append([])
                        for i_mode in range(self["dimensions"]["N_mode"]):
                            self["ray"][sub_key][i_time][i_ch].append([])
                            for i_ray in range(self["dimensions"]["N_ray"]):
                                self["ray"][sub_key][i_time][i_ch][i_mode].append( \
                                        rootgrp["Results"][key+ "_" +sub_key][i_time,i_ch,i_mode,i_ray,\
                                                                        :self["dimensions"]["N_LOS"][i_time,i_ch,i_mode,i_ray]])
                self["ray"][sub_key] = np.array(self["ray"][sub_key], dtype=np.object)
        # Get the shape information of the individual LOS length into the NETCDF file
        self.comment = rootgrp["Results"].comment 
        self.edition = rootgrp["Results"].edition
        self["git"]["ECRad"] = rootgrp["Results"].ECRad_git_tag
        self["git"]["GUI"] = rootgrp["Results"].ECRadGUI_git_tag
        self["git"]["Pylib"] = rootgrp["Results"].ECRadPylib_git_tag
        self.data_origin = filename
        rootgrp.close()
    
if(__name__ == "__main__"):
    res = ECRadResults()
#     res.from_mat("/mnt/c/Users/Severin/ECRad/ECRad_33585_EXT_ed1.mat")
#     res.to_netcdf("/mnt/c/Users/Severin/ECRad/ECRad_33585_EXT_ed1.nc")
#"/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc"
    # res.from_mat("/mnt/c/Users/Severin/ECRad/ECRad_35662_EXT_ed8.mat")
    # res.to_netcdf("/mnt/c/Users/Severin/ECRad/ECRad_35662_EXT_ed8.nc")
#     res.reset()
    res.from_netcdf("/mnt/c/Users/Severin/ECRad/HFS_LHCD/ECRad_147634_EXT_ed1.nc")
    res.Scenario["plasma"]["dist_obj"].plot_Te_ne()
#     res.reset()
#     res.from_mat("/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_ed19.mat")
#     res.to_netcdf("/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_Scenario.nc")
#     res.reset()
#     res.from_netcdf("/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_Scenario.nc")