'''
Created on Mar 20, 2019

@author: sdenk
'''
from collections import OrderedDict as od
import getpass
import os
from scipy.io import loadmat, savemat
import sys
from GlobalSettings import globalsettings
import numpy as np
from equilibrium_utils import EQDataSlice, special_points, EQDataExt
from Diags import Diag, ECRH_diag, ECI_diag, EXT_diag
from distribution_io import load_f_from_mat
from __builtin__ import True
if(globalsettings.AUG):
    from ECRad_DIAG_AUG import DefaultDiagDict
elif(globalsettings.TCV):
    from ECRad_DIAG_TCV import DefaultDiagDict
else:
    from Diags import DefaultDiagDict
# THis class holds all the input data provided to ECRad with the exception of the ECRad configuration

class ECRadScenario:
    def __init__(self, noLoad=False):
        if(not noLoad):
            self.scenario_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_last_scenario.mat")
            try:
                self.from_mat(path_in=self.scenario_file)
            except (IOError,KeyError):
                print("Failed to import last used Scenario")
                self.reset()
        else:
            self.reset()


    def reset(self):
        self.used_diags_dict = od()
        self.avail_diags_dict = DefaultDiagDict
        self.plasma_dict = {}
        self.plasma_dict["time"] = []
        self.plasma_dict["rhop_prof"] = []
        self.plasma_dict["Te"] = []
        self.plasma_dict["ne"] = []
        self.plasma_dict["eq_data"] = []
        self.ray_launch = []
        self.profile_dimension = 1
        self.diags_set = False
        self.plasma_set = False
        self.IDA_exp = "SDENK"
        self.IDA_ed = 0
        self.EQ_exp = "AUGD"
        self.EQ_diag = "EQH"
        self.EQ_ed = 0
        if(globalsettings.AUG):
            self.bt_vac_correction = 1.005
        else:
            self.bt_vac_correction = 1.000
        self.ne_rhop_scale = 1.e0
        self.Te_rhop_scale = 1.e0
        self.Te_scale = 1.0
        self.ne_scale = 1.0
        self.shot = 35662
        self.default_diag = "ECE"
        self.data_source = "aug_database"
        self.dist_obj = None

    def from_mat(self, mdict=None, path_in=None, load_plasma_dict=True):
        self.reset()
        if(mdict is None):
            if(path_in is None):
                filename = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_last_scenario.mat")
            else:
                filename = path_in
            try:
                mdict = loadmat(filename, chars_as_strings=True, squeeze_me=True, uint16_codec='ascii')
            except IOError as e:
                print(e)
                print("Error: " + filename + " does not exist")
                return False
            except TypeError as e:
                print(e)
                print("Error: File appears to be corrupted does not exist")
                return False
        self.profile_dimension = mdict["profile_dimension"]
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["diag", "time", "Diags_exp", "Diags_diag", "Diags_ed", "Extra_arg_1", "Extra_arg_2", "Extra_arg_3", \
                            "used_diags"]
        at_least_2d_keys = ["eq_R", "eq_z", "diag_name", "launch_f", "launch_df", "launch_R", "launch_phi", \
                             "launch_z", "launch_tor_ang" , "launch_pol_ang", "launch_dist_focus", "launch_diag_name", \
                             "launch_width", "launch_pol_coeff_X", "eq_special", "eq_special_complete"  ]
        at_least_3d_keys = ["eq_Psi", "eq_rhop", "eq_Br", "eq_Bt", "eq_Bz"]
        if(self.profile_dimension == 1):
            for key in ["rhop_prof", "Te", "ne"  ]:
                at_least_2d_keys.append(key)
        elif(self.profile_dimension == 2):
            for key in ["Te", "ne"]:
                at_least_3d_keys.append(key)
        self.shot = mdict["shot"]
        self.IDA_exp = mdict["IDA_exp"]
        self.IDA_ed = mdict["IDA_ed"]
        self.EQ_exp = mdict["EQ_exp"]
        self.EQ_diag = mdict["EQ_diag"]
        self.EQ_ed = mdict["EQ_ed"]
        try:
            self.bt_vac_correction = mdict["bt_vac_correction"]
        except KeyError:
            self.bt_vac_correction = 1.005
        try:
            self.Te_rhop_scale = mdict["Te_rhop_scale"]
            self.ne_rhop_scale = mdict["ne_rhop_scale"]
        except KeyError:
            self.ne_rhop_scale = 1.e0
            self.Te_rhop_scale = 1.e0
        try:
            self.Te_scale = mdict["Te_scale"]
            self.ne_scale = mdict["ne_scale"]
        except KeyError:
            self.Te_scale = 1.0
            self.ne_scale = 1.0
        increase_time_dim = False
        if(np.isscalar(mdict["time"])):
            self.plasma_dict["time"] = np.array([mdict["time"]])
            increase_time_dim = True
        else:
            self.plasma_dict["time"] = mdict["time"]
        for key in mdict.keys():
            if(not key.startswith("_")):  # throw out the .mat specific information
                try:
                    if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                        mdict[key] = np.atleast_1d(mdict[key])
                    elif(key in at_least_2d_keys):
                        mdict[key] = np.atleast_2d(mdict[key])
                    elif(key in at_least_3d_keys):
                        if(increase_time_dim):
                            mdict[key] = np.array([mdict[key]])
                except Exception as e:
                    print(key)
                    print(e)
        self.used_diags_dict = od()
        self.default_diag = mdict["used_diags"][0]
        for i in range(len(mdict["used_diags"])):
            diagname = mdict["used_diags"][i]
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI") and globalsettings.AUG):
                self.used_diags_dict.update({diagname: ECI_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                              mdict["Extra_arg_1"][i], mdict["Extra_arg_2"][i], int(mdict["Extra_arg_3"][i]))})
            elif(("CT" in diagname or "IEC" in diagname) and globalsettings.AUG):
                if(mdict["Extra_arg_3"][i] == "None"):
                    self.used_diags_dict.update({diagname: ECRH_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                              int(mdict["Extra_arg_1"][i]), float(mdict["Extra_arg_2"][i]), True)})
                else:
                    try:
                        self.used_diags_dict.update({diagname: ECRH_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                                  int(mdict["Extra_arg_1"][i]), float(mdict["Extra_arg_2"][i]), bool(int(mdict["Extra_arg_3"][i])))})
                    except ValueError:
                        if(mdict["Extra_arg_3"][i] == "True"):
                            extra_arg_3 = True
                        else:
                            extra_arg_3 = False
                        self.used_diags_dict.update({diagname: ECRH_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                                  int(mdict["Extra_arg_1"][i]), float(mdict["Extra_arg_2"][i]), extra_arg_3)})
            elif(diagname == "EXT"):
                if("Ext_launch_pol" in mdict.keys()):
                    self.used_diags_dict.update({diagname: EXT_diag(diagname, mdict["Ext_launch_geo"], mdict["Ext_launch_pol"])})
                else:
                    self.used_diags_dict.update({diagname: EXT_diag(diagname, mdict["Ext_launch_geo"], -1)})
            elif(globalsettings.AUG):
                self.used_diags_dict.update({diagname: \
                        Diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]))})
        if("launch_R" in mdict.keys()):
            for itime in range(len(self.plasma_dict["time"])):
                self.ray_launch.append({})
                self.ray_launch[-1]["f"] = mdict["launch_f"][itime]
                self.ray_launch[-1]["df"] = mdict["launch_df"][itime]
                self.ray_launch[-1]["R"] = mdict["launch_R"][itime]
                self.ray_launch[-1]["phi"] = mdict["launch_phi"][itime]
                self.ray_launch[-1]["z"] = mdict["launch_z"][itime]
                self.ray_launch[-1]["theta_pol"] = mdict["launch_pol_ang"][itime]
                self.ray_launch[-1]["phi_tor"] = mdict["launch_tor_ang"][itime]
                self.ray_launch[-1]["dist_focus"] = mdict["launch_dist_focus"][itime]
                self.ray_launch[-1]["width"] = mdict["launch_width"][itime]
                self.ray_launch[-1]["pol_coeff_X"] = mdict["launch_pol_coeff_X"][itime]
                try:
                    self.ray_launch[-1]["diag_name"] = mdict["launch_diag_name"][itime]
                except:
                    self.ray_launch[-1]["diag_name"] = mdict["diag_name"][itime]
            self.ray_launch = np.array(self.ray_launch)
            self.diags_set = True
        if(not load_plasma_dict):
            return
        if(self.profile_dimension == 1):
            self.plasma_dict["rhop_prof"] = mdict["rhop_prof"]
        self.plasma_dict["Te"] = mdict["Te"]
        self.plasma_dict["ne"] = mdict["ne"]
        self.plasma_dict["eq_data"] = []
        for i in range(len(self.plasma_dict["time"])):
            if("eq_special_complete" in mdict.keys()):
                entry = mdict["eq_special_complete"][i]
                spcl = special_points(entry[0], entry[1], entry[4], entry[2], entry[3], entry[5])
            else:
                entry = mdict["eq_special"][i]
                spcl = special_points(0.0, 0.0, entry[0], 0.0, 0.0, entry[1])
            self.plasma_dict["eq_data"].append(EQDataSlice(self.plasma_dict["time"][i], \
                                                                  mdict["eq_R"][i], mdict["eq_z"][i], \
                                                                  mdict["eq_Psi"][i], mdict["eq_Br"][i], \
                                                                  mdict["eq_Bt"][i], mdict["eq_Bz"][i], \
                                                                  spcl, rhop=mdict["eq_rhop"][i]))
        self.plasma_dict["eq_data"] = np.array(self.plasma_dict["eq_data"])
        self.plasma_dict["vessel_bd"] = mdict["vessel_bd"]
        for diag_key in self.avail_diags_dict:
            if(diag_key in list(self.used_diags_dict.keys())):
                self.avail_diags_dict.update({diag_key: self.used_diags_dict[diag_key]})
        if("data_source" in mdict.keys()):
            self.data_source = mdict["data_source"]
        else:
            self.data_source = "aug_database"
        self.plasma_set = True

    def autosave(self):
        self.to_mat_file(filename=self.scenario_file)

    def to_mat_file(self, filename=None, mdict_in=None):
        if(mdict_in is None):
            if(filename is None):
                print("Either mdict_in or filename need to be provided")
                raise ValueError
            mdict = {}
        else:
            mdict = mdict_in
        mdict["shot"] = self.shot
        mdict["IDA_exp"] = self.IDA_exp
        mdict["IDA_ed"] = self.IDA_ed
        mdict["EQ_exp"] = self.EQ_exp
        mdict["EQ_diag"] = self.EQ_diag
        mdict["EQ_ed"] = self.EQ_ed
        mdict["used_diags"] = list(self.used_diags_dict.keys()) # Cast ordered_dict_keys to list
        mdict["Diags_exp"] = []
        mdict["Diags_diag"] = []
        mdict["Diags_ed"] = []
        mdict["Extra_arg_1"] = []
        mdict["Extra_arg_2"] = []
        mdict["Extra_arg_3"] = []
        for diagname in list(self.used_diags_dict.keys()):
            if(hasattr(self.used_diags_dict[diagname], "exp")):
                mdict["Diags_exp"].append(self.used_diags_dict[diagname].exp)
                mdict["Diags_diag"].append(self.used_diags_dict[diagname].diag)
                mdict["Diags_ed"].append(self.used_diags_dict[diagname].ed)
            if(diagname == "ECN" or diagname == "ECO"):
                mdict["Extra_arg_1"].append(self.used_diags_dict[diagname].Rz_exp)
                mdict["Extra_arg_2"].append(self.used_diags_dict[diagname].Rz_diag)
                mdict["Extra_arg_3"].append(self.used_diags_dict[diagname].Rz_ed)
            elif(diagname in ["CTC", "IEC", "CTA"]):
                mdict["Extra_arg_1"].append("{0:d}".format(self.used_diags_dict[diagname].beamline))
                mdict["Extra_arg_2"].append("{0:1.3f}".format(self.used_diags_dict[diagname].pol_coeff_X))
                mdict["Extra_arg_3"].append(int(self.used_diags_dict[diagname].base_freq_140))
            elif(diagname == "EXT"):
                launch_geo, pol = self.used_diags_dict[diagname].get_launch_geo()
                mdict["Ext_launch_geo"] = launch_geo
                mdict["Ext_launch_pol"] = pol
            elif(diagname == "VCE" and globalsettings.AUG):
                mdict["Extra_arg_1"].append("{0:1.3f}".format(self.used_diags_dict[diagname].R_scale))
                mdict["Extra_arg_2"].append("{0:1.3f}".format(self.used_diags_dict[diagname].z_scale))
            else:
                mdict["Extra_arg_1"].append("None")
                mdict["Extra_arg_2"].append("None")
                mdict["Extra_arg_3"].append("None")
        mdict["launch_diag_name"] = []
        mdict["launch_R"] = []
        mdict["launch_phi"] = []
        mdict["launch_z"] = []
        mdict["launch_f"] = []
        mdict["launch_df"] = []
        mdict["launch_pol_ang"] = []
        mdict["launch_tor_ang"] = []
        mdict["launch_pol_coeff_X"] = []
        mdict["launch_dist_focus"] = []
        mdict["launch_width"] = []
        for itime in range(len(self.plasma_dict["time"])):
            mdict["launch_diag_name"].append(self.ray_launch[itime]["diag_name"])
            mdict["launch_R"].append(self.ray_launch[itime]["R"])
            mdict["launch_phi"].append(self.ray_launch[itime]["phi"])
            mdict["launch_z"].append(self.ray_launch[itime]["z"])
            mdict["launch_f"].append(self.ray_launch[itime]["f"])
            mdict["launch_df"].append(self.ray_launch[itime]["df"])
            mdict["launch_pol_ang"].append(self.ray_launch[itime]["theta_pol"])
            mdict["launch_tor_ang"].append(self.ray_launch[itime]["phi_tor"])
            mdict["launch_pol_coeff_X"].append(self.ray_launch[itime]["pol_coeff_X"])
            mdict["launch_dist_focus"].append(self.ray_launch[itime]["dist_focus"])
            mdict["launch_width"].append(self.ray_launch[itime]["width"])
        mdict["time"] = self.plasma_dict["time"]
        mdict["Te"] = self.plasma_dict["Te"]
        mdict["ne"] = self.plasma_dict["ne"]
        mdict["bt_vac_correction"] = self.bt_vac_correction
        mdict["Te_rhop_scale"] = self.Te_rhop_scale 
        mdict["ne_rhop_scale"] = self.ne_rhop_scale
        mdict["Te_scale"] = self.Te_scale
        mdict["ne_scale"] = self.ne_scale
        mdict["profile_dimension"] = self.profile_dimension
        if(mdict["profile_dimension"] == 1):
            mdict["rhop_prof"] = self.plasma_dict["rhop_prof"]
        mdict["eq_R"] = []
        mdict["eq_z"] = []
        mdict["eq_Psi"] = []
        mdict["eq_rhop"] = []
        mdict["eq_Br"] = []
        mdict["eq_Bt"] = []
        mdict["eq_Bz"] = []
        mdict["bt_vac_correction"] = self.bt_vac_correction
        if(self.plasma_dict["eq_data"][0].R_sep is not None):
            mdict["eq_special_complete"] = []
        mdict["eq_special"] = []
        for i in range(len(self.plasma_dict["time"])):
            mdict["eq_R"].append(self.plasma_dict["eq_data"][i].R)
            mdict["eq_z"].append(self.plasma_dict["eq_data"][i].z)
            mdict["eq_Psi"].append(self.plasma_dict["eq_data"][i].Psi)
            mdict["eq_rhop"].append(self.plasma_dict["eq_data"][i].rhop)
            mdict["eq_Br"].append(self.plasma_dict["eq_data"][i].Br)
            mdict["eq_Bt"].append(self.plasma_dict["eq_data"][i].Bt)
            mdict["eq_Bz"].append(self.plasma_dict["eq_data"][i].Bz)
            mdict["eq_special"].append(self.plasma_dict["eq_data"][i].special)
            if(self.plasma_dict["eq_data"][i].R_sep is not None):
                mdict["eq_special_complete"].append(np.array([self.plasma_dict["eq_data"][i].R_ax, \
                                                          self.plasma_dict["eq_data"][i].z_ax, \
                                                          self.plasma_dict["eq_data"][i].R_sep, \
                                                          self.plasma_dict["eq_data"][i].z_sep, \
                                                          self.plasma_dict["eq_data"][i].Psi_ax, \
                                                          self.plasma_dict["eq_data"][i].Psi_sep]))
        mdict["eq_R"] = np.array(mdict["eq_R"])
        mdict["eq_z"] = np.array(mdict["eq_z"])
        mdict["eq_Psi"] = np.array(mdict["eq_Psi"])
        mdict["eq_rhop"] = np.array(mdict["eq_rhop"])
        mdict["eq_Br"] = np.array(mdict["eq_Br"])
        mdict["eq_Bt"] = np.array(mdict["eq_Bt"])
        mdict["eq_Bz"] = np.array(mdict["eq_Bz"])
        mdict["eq_special"] = np.array(mdict["eq_special"])
        mdict["vessel_bd"] = self.plasma_dict["vessel_bd"]
        mdict["data_source"] = self.data_source
        if(filename is not None):
            try:
                savemat(filename, mdict, appendmat=False)
                print("Successfully created: ", filename)
            except TypeError as e:
                print("Failed to save to .mat")
                print(e)
                print(mdict)
        if(mdict_in is not None):
            return mdict
        else:
            return True

    def load_dist_obj(self, filename):
        self.dist_obj = load_f_from_mat(filename, use_dist_prefix=True)
        
if(__name__ == "__main__"):
    newScen = ECRadScenario(noLoad=True)
    newScen.from_mat( path_in="/tokp/work/sdenk/ECRad/ECRad_35662_ECECTCCTA_ed2.mat")