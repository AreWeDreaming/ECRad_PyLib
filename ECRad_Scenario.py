'''
Created on Mar 20, 2019

@author: sdenk
'''
from collections import OrderedDict as od
import os
from scipy.io import loadmat, savemat
from Global_Settings import globalsettings, GlobalSettingsAUG
import numpy as np
from Basic_Methods.Equilibrium_Utils import EQDataExt, EQDataSlice, special_points
from Diag_Types import Diag, ECRH_diag, ECI_diag, EXT_diag
from Distribution_IO import load_f_from_mat
from Distribution_Classes import Gene, GeneBiMax
from netCDF4 import Dataset
if(globalsettings.AUG):
    from ECRad_DIAG_AUG import DefaultDiagDict
else:
    from Diag_Types import DefaultDiagDict
# THis class holds all the input data provided to ECRad with the exception of the ECRad configuration


class ECRadScenario(dict):
    def __init__(self, noLoad=False):
        self.scenario_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_last_scenario.nc")
        self.reset()
        if(not noLoad):
            try:
                self.load(filename=self.scenario_file)
            except FileNotFoundError as e:
                print("Failed to import last used Scenario")
                print("Cause: " + str(e))
                self.reset()


    def reset(self):
        self["dimensions"] = {}
        self["dimensions"]["N_time"] = 0
        self["dimensions"]["N_profiles"] = 0
        self["dimensions"]["N_eq_2D_R"] = 0
        self["dimensions"]["N_eq_2D_z"] = 0
        self["dimensions"]["N_vessel_bd"] = 0
        self["dimensions"]["N_vessel_dim"] = 2
        self["dimensions"]["N_ch"] = 0
        self["dimensions"]["N_used_diags"] = 0
        self["time"] = []
        self["shot"] = 0
        self["plasma"] = {}
        self["plasma"]["rhop_prof"] = []
        self["plasma"]["rhot_prof"] = []
        self["plasma"]["prof_reference"] = "rhop_prof"
        self["plasma"]["2D_prof"] = False
        self["plasma"]["vessl_bd"] = None
        self["plasma"]["Te"] = []
        self["plasma"]["ne"] = []
        self["scaling"] = {}
        self["scaling"]["Bt_vac_scale"] = 1.0
        self["scaling"]["Te_rhop_scale"] = 1.0
        self["scaling"]["ne_rhop_scale"] = 1.0
        self["scaling"]["Te_scale"] = 1.0
        self["scaling"]["ne_scale"] = 1.0
        self["plasma"]["eq_dim"] = 2
        self["plasma"]["dist_obj"] = None
        self["plasma"]["GENE_obj"] = None
        self["diagnostic"] = {}
        self['diagnostic']["f"] = []
        self['diagnostic']["df"] = []
        self['diagnostic']["R"] = []
        self['diagnostic']["phi"] = []
        self['diagnostic']["z"] = []
        self['diagnostic']["theta_pol"] = []
        self['diagnostic']["phi_tor"] = []
        self['diagnostic']["dist_focus"] = []
        self['diagnostic']["width"] = []
        self['diagnostic']["pol_coeff_X"] = []
        self['diagnostic']["diag_name"] = []
        self["used_diags_dict"] = od()
        self["avail_diags_dict"] = DefaultDiagDict
        self["AUG"] = {}
        if(globalsettings.AUG):
            self["AUG"]["IDA_exp"] = "AUGD"
            self["AUG"]["IDA_ed"] = 0
            self["AUG"]["EQ_exp"] = "AUGD"
            self["AUG"]["EQ_diag"] = "EQH"
            self["AUG"]["EQ_ed"] = 0
            self["plasma"]["Bt_vac_scale"] = 1.005
            self.data_source = "aug_database"
            self.default_diag = "ECE"
        else:
            self.data_source = None
        self["plasma"]["Bt_vac_scale"] = 1.0
        self.default_diag = "EXT"
        self["plasma"]["eq_data_2D"] = EQDataExt(self["shot"], \
                                                 Ext_data=True)
        self["plasma"]["eq_data_3D"] = {}
        self["plasma"]["eq_data_3D"]["equilibrium_file"] = ""
        self["plasma"]["eq_data_3D"]["equilibrium_type"] = ''
        self["plasma"]["eq_data_3D"]["use_mesh"] = False
        self["plasma"]["eq_data_3D"]["use_symmetry"] = True
        self["plasma"]["eq_data_3D"]["B_ref"] = 1.0
        self["plasma"]["eq_data_3D"]["s_plus"] = 1.0
        self["plasma"]["eq_data_3D"]["s_max"] = 1.2
        self["plasma"]["eq_data_3D"]["interpolation_acc"] = 1.e-12
        self["plasma"]["eq_data_3D"]["fourier_coeff_trunc"] = 1.e-12
        self["plasma"]["eq_data_3D"]["h_mesh"] = 1.5e-2 # meters
        self["plasma"]["eq_data_3D"]["delta_phi_mesh"] = 2.0 # Degrees
        self["plasma"]["eq_data_3D"]["vessel_filename"] = ""
        self.diags_set = False
        self.plasma_set = False

    def load(self, filename=None, mdict=None, rootgrp=None):
        if(filename is not None):
            ext = os.path.splitext(filename)[1]
            if(ext == ".mat"):
                self.from_mat(path_in=filename)
            elif(ext == ".nc"):
                self.from_netcdf(filename=filename)
            else:
                print("Extension " + ext + " is unknown")
                raise(ValueError)
        elif(mdict is not None):
            self.from_mat(mdict)
        elif(rootgrp is not None):
            self.from_netcdf(rootgrp=rootgrp)
            
    def set_up_dimensions(self):
        self["dimensions"]["N_time"] = len(self["time"])
        if(not self["plasma"]["2D_prof"]):
            self["dimensions"]["N_profiles"] = len(self["plasma"][self["plasma"]["prof_reference"]][0])
        if(self["plasma"]["eq_dim"] == 2):
            self["dimensions"]["N_eq_2D_R"] = self["plasma"]["eq_data_2D"].eq_shape[0]
            self["dimensions"]["N_eq_2D_z"] = self["plasma"]["eq_data_2D"].eq_shape[1]
        self["dimensions"]["N_vessel_bd"] = len(self["plasma"]["vessel_bd"])
        self["dimensions"]["N_vessel_dim"] = 2
        self["dimensions"]["N_ch"] = len(self["diagnostic"]["f"][0])
        self["N_used_diags"] = len(list(self["used_diags_dict"].keys()))
        

    def drop_time_point(self, itime):
        time = self["time"][itime]
        self["time"] = np.delete(self["time"], itime)
        for sub_key in ["rhop_prof", "rhot_prof", "rhop_prof", "Te", "ne"]:
            if(len(self["plasma"][sub_key]) == 0):
                continue
            self["plasma"][sub_key] = np.delete(self["plasma"][sub_key], itime, 0)
        if(self["plasma"]["eq_dim"] == 2):
            self["plasma"]["eq_data_2D"].RemoveSlice(time)
        for sub_key in self["diagnostic"].keys():
            self["diagnostic"][sub_key] = np.delete(self["diagnostic"][sub_key], itime, 0) 
        self["dimensions"]["N_time"] -= 1    

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
        increase_time_dim = False
        if(np.isscalar(mdict["time"])):
            self["time"] = np.array([mdict["time"]])
            increase_time_dim = True
        else:
            self["time"] = mdict["time"]
        try:
            self["plasma"]["2D_prof"] = mdict["profile_dimension"] > 1
        except KeyError:
            if(increase_time_dim):
                self["plasma"]["2D_prof"] = mdict["Te"].ndim  > 1
            else:
                self["plasma"]["2D_prof"] = mdict["Te"][0].ndim  > 1
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["diag", "time", "Diags_exp", "Diags_diag", "Diags_ed", "Extra_arg_1", "Extra_arg_2", "Extra_arg_3", \
                            "used_diags"]
        at_least_2d_keys = ["eq_R", "eq_z", "diag_name", "launch_f", "launch_df", "launch_R", "launch_phi", \
                             "launch_z", "launch_tor_ang" , "launch_pol_ang", "launch_dist_focus", "launch_diag_name", \
                             "launch_width", "launch_pol_coeff_X", "eq_special", "eq_special_complete"  ]
        at_least_3d_keys = ["eq_Psi", "eq_rhop", "eq_Br", "eq_Bt", "eq_Bz"]
        if(self["plasma"]["2D_prof"]  == 2):
            for key in ["Te", "ne"]:
                at_least_3d_keys.append(key)
        else:
            for key in ["rhop_prof", "Te", "ne",  "rhot_prof"]:
                at_least_2d_keys.append(key)
        for key in mdict:
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
        self["shot"] = mdict["shot"]
        if(globalsettings.AUG):
            self["AUG"]["IDA_exp"] = mdict["IDA_exp"]
            self["AUG"]["IDA_ed"] = mdict["IDA_ed"]
            self["AUG"]["EQ_exp"] = mdict["EQ_exp"]
            self["AUG"]["EQ_diag"] = mdict["EQ_diag"]
            self["AUG"]["EQ_ed"] = mdict["EQ_ed"]
            self["used_diags_dict"] = od()
        self.default_diag = mdict["used_diags"][0]
        self["dimensions"]["N_used_diags"] = len(mdict["used_diags"])
        for i in range(len(mdict["used_diags"])):
            diagname = mdict["used_diags"][i]
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI")): #and globalsettings.AUG):
                self["used_diags_dict"].update({diagname: ECI_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                                                   mdict["Extra_arg_1"][i], mdict["Extra_arg_2"][i], int(mdict["Extra_arg_3"][i]))})
            elif("CT" in diagname or "IEC" in diagname):
                try:
                    extra_arg_3 = bool(int(mdict["Extra_arg_3"][i]))
                except ValueError:
                    extra_arg_3 =  mdict["Extra_arg_3"][i] == "True"
                    if(not extra_arg_3):
                        extra_arg_3 =  mdict["Extra_arg_3"][i] == "None"
                self["used_diags_dict"].update({diagname: ECRH_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                              int(mdict["Extra_arg_1"][i]), float(mdict["Extra_arg_2"][i]), extra_arg_3)})
            elif(diagname == "EXT"):
                if("Ext_launch_pol" in mdict):
                    self["used_diags_dict"].update({diagname: EXT_diag(diagname, mdict["Ext_launch_geo"], mdict["Ext_launch_pol"])})
                else:
                    self["used_diags_dict"].update({diagname: EXT_diag(diagname, mdict["Ext_launch_geo"], -1)})
            else:#if(globalsettings.AUG):
                self["used_diags_dict"].update({diagname: \
                        Diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]))})
            for diag_key in self["avail_diags_dict"]:
                if(diag_key in list(self["used_diags_dict"])):
                    self["avail_diags_dict"].update({diag_key: self["used_diags_dict"][diag_key]})
        try:
            self["scaling"]["Bt_vac_scale"] = mdict["bt_vac_correction"]
        except KeyError:
            self["scaling"]["Bt_vac_scale"] = 1.005
        try:
            self["scaling"]["Te_rhop_scale"] = mdict["Te_rhop_scale"]
            self["scaling"]["ne_rhop_scale"] = mdict["ne_rhop_scale"]
        except KeyError:
            self["scaling"]["Te_rhop_scale"] = mdict["Te_rhop_scale"]
            self["scaling"]["ne_rhop_scale"] = mdict["ne_rhop_scale"]
        try:
            self["scaling"]["Te_scale"] = mdict["Te_scale"]
            self["scaling"]["ne_scale"] = mdict["ne_scale"]
        except KeyError:
            self["scaling"]["Te_scale"] = 1.0
            self["scaling"]["ne_scale"] = 1.0
        if("launch_R" in mdict):
            self['dimensions']["N_ch"] = len(mdict["launch_f"][0])
            self['diagnostic']["f"] = mdict["launch_f"]
            self['diagnostic']["df"] = mdict["launch_df"]
            self['diagnostic']["R"] = mdict["launch_R"]
            self['diagnostic']["phi"] = mdict["launch_phi"]
            self['diagnostic']["z"] = mdict["launch_z"]
            self['diagnostic']["theta_pol"] = mdict["launch_pol_ang"]
            self['diagnostic']["phi_tor"] = mdict["launch_tor_ang"]
            self['diagnostic']["dist_focus"] = mdict["launch_dist_focus"]
            self['diagnostic']["width"] = mdict["launch_width"]
            self['diagnostic']["pol_coeff_X"] = mdict["launch_pol_coeff_X"]
            try:
                self['diagnostic']["diag_name"] = mdict["launch_diag_name"]
            except:
                self['diagnostic']["diag_name"] = mdict["diag_name"]
            self.diags_set = True
        else:
            print("No launch information in this save file, you will have to reload the diagnostic info from the AUG database")
        if(not load_plasma_dict):
            return
        try:
            if(bool(mdict["Use_3D_used"])):
                self["plasma"]["eq_dim"] = 3
            else:
                self["plasma"]["eq_dim"] = 2
        except KeyError:
            self["plasma"]["eq_dim"] = 2
        if(self["plasma"]["2D_prof"]):
            self["plasma"]["prof_reference"] = "2D"
        else:            
            try:
                self["plasma"]["rhot_prof"] = mdict["rhot_prof"]
            except KeyError:
                print("Could not find rho_tor profile")
                if(self["plasma"]["eqq_dim"] == 3):
                    print("INFO: 3D Scenario identified")
                    print("INFO: Overriding rhot_prof with rhop_prof")
                    self["plasma"]["rhot_prof"] = mdict["rhop_prof"]
            if(self["plasma"]["eq_dim"] != 3):
                self["plasma"]["rhop_prof"] = mdict["rhop_prof"]
            try:
                self["plasma"]["prof_reference"] = mdict["prof_reference"]
            except KeyError:
                if(self["plasma"]["eq_dim"] == 3):
                    self["plasma"]["prof_reference"] = "rhot_prof"
                else:
                    print("INFO: Could not find profile axis type. Falling  back to rho_pol")
                    self["plasma"]["prof_reference"] = "rhop_prof"
        self["plasma"]["Te"] = mdict["Te"]
        self["plasma"]["ne"] = mdict["ne"]
        if(self["plasma"]["eq_dim"] == 2):
            self["plasma"]["eq_data_2D"] = EQDataExt(self["shot"], \
                                                  Ext_data=True)
            slices = []
            for i in range(len(self["time"])):
                if("eq_special_complete" in mdict):
                    entry = mdict["eq_special_complete"][i]
                    spcl = special_points(entry[0], entry[1], entry[4], entry[2], entry[3], entry[5])
                else:
                    entry = mdict["eq_special"][i]
                    spcl = special_points(0.0, 0.0, entry[0], 0.0, 0.0, entry[1])
                slices.append(EQDataSlice(self["time"][i], \
                                          mdict["eq_R"][i], mdict["eq_z"][i], \
                                          mdict["eq_Psi"][i], mdict["eq_Br"][i], \
                                          mdict["eq_Bt"][i], mdict["eq_Bz"][i], \
                                          spcl, rhop=mdict["eq_rhop"][i]))
                # The old .mat file store the scaled Bt not the original Bt
                # In the new netcdf files the original Bt is stored
                # The scaled Bt is only used directly in ECRad
                EQobj = EQDataExt(self["shot"], Ext_data=True)
                EQobj.set_slices_from_ext([self["time"][i]], [slices[-1]])
                slices[-1].R_ax, slices[-1].z_ax = EQobj.get_axis(self["time"][i])
                slices[-1].Bt = self["plasma"]["eq_data_2D"].adjust_external_Bt_vac(slices[-1].Bt, slices[-1].R, \
                                                                                    slices[-1].R_ax, 1.0/self["scaling"]["Bt_vac_scale"])
            self["plasma"]["eq_data_2D"].set_slices_from_ext(self["time"], slices)
            self["plasma"]["vessel_bd"] = mdict["vessel_bd"].T
            self["dimensions"]["N_eq_2D_R"] = len(mdict["eq_R"][0])
            self["dimensions"]["N_eq_2D_z"] = len(mdict["eq_z"][0])
            self["dimensions"]["N_vessel_bd"] = len(mdict["vessel_bd"].T)
        else:
            self["plasma"]["eq_data_3D"]["equilibrium_files"] = np.array([mdict["Use_3D_" + "equilibrium_file"]])
            self["plasma"]["eq_data_3D"]["equilibrium_type"] = mdict["Use_3D_" + "equilibrium_type"]
            self["plasma"]["eq_data_3D"]["use_mesh"] = bool(mdict["Use_3D_" + "use_mesh"])
            self["plasma"]["eq_data_3D"]["use_symmetry"] = bool(mdict["Use_3D_" + "use_symmetry"])
            self["plasma"]["eq_data_3D"]["B_ref"] = mdict["Use_3D_" + "B_ref"]
            self["plasma"]["eq_data_3D"]["s_plus"] = mdict["Use_3D_" + "s_plus"]
            self["plasma"]["eq_data_3D"]["s_max"] = mdict["Use_3D_" + "s_max"]
            self["plasma"]["eq_data_3D"]["interpolation_acc"] = mdict["Use_3D_" + "interpolation_acc"]
            self["plasma"]["eq_data_3D"]["fourier_coeff_trunc"] = mdict["Use_3D_" + "fourier_coeff_trunc"]
            self["plasma"]["eq_data_3D"]["h_mesh"] = mdict["Use_3D_" + "h_mesh"]
            self["plasma"]["eq_data_3D"]["delta_phi_mesh"] = mdict["Use_3D_" + "delta_phi_mesh"]
            self["plasma"]["eq_data_3D"]["vessel_filename"] = mdict["Use_3D_" + "vessel_filename"]
            if(self["plasma"]["rhot_prof"] is None):
                print("For 3D calculations the rho toroidal is obligatory")
                raise ValueError("Failed to load equilibrium")
            else:
                self["plasma"]["prof_reference"] = "rhot_prof"
        if("data_source" in mdict):
            self.data_source = mdict["data_source"]
        elif(globalsettings.AUG):
            self.data_source = "aug_database"
        else:
            self.data_source = "Unknown"
        self.plasma_set = True
        self["dimensions"]["N_time"] = len(self["time"])
        if(not self["plasma"]["2D_prof"]):
            self["dimensions"]["N_profiles"] = len(self["plasma"]["Te"][0])
        try:        
            self.load_dist_obj(mdict=mdict)
        except Exception:
            print("No distribution data in current Scenario")

    def to_netcdf(self, filename=None, rootgrp=None):
        if(filename is not None):
            rootgrp = Dataset(filename, "w", format="NETCDF4")
        rootgrp.createGroup("Scenario")
        rootgrp["Scenario"].createDimension('str_dim', 1)
        for sub_key in self["dimensions"].keys():
            rootgrp["Scenario"].createDimension(sub_key, self["dimensions"][sub_key])
        var = rootgrp["Scenario"].createVariable("plasma" + "_" + "2D_prof", "b")
        var[...] = self["plasma"]["2D_prof"]
        var = rootgrp["Scenario"].createVariable("time", "f8", ("N_time",))
        var[...] = self["time"]
        var = rootgrp["Scenario"].createVariable("shot", "i8")
        var[...] = self["shot"]
        for sub_key in ["Te", "ne"]:
            if(not self["plasma"]["2D_prof"]):
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + sub_key, \
                                                         "f8", ("N_time", "N_profiles"))
            else:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + sub_key, \
                                                         ("N_time", "N_eq_2D_R", "N_eq_2D_z"))
            var[:] = self['plasma'][sub_key]
        if(not self["plasma"]["2D_prof"]):
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + "prof_reference", str, ('str_dim',))
            var[0] = self["plasma"]["prof_reference"]
            sub_key = self["plasma"]["prof_reference"]
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + sub_key, "f8", \
                                                     ("N_time", "N_profiles"))
            var[:] = self["plasma"][sub_key]
        for sub_key in self["scaling"].keys():
            var = rootgrp["Scenario"].createVariable("scaling"+ "_" + sub_key, "f8")
            var[...] = self["scaling"][sub_key]
        var = rootgrp["Scenario"].createVariable("plasma" + "_" + "eq_dim", "i8")
        var[...] = self["plasma"]["eq_dim"]
        if(self["plasma"]["eq_dim"] == 3):
            for sub_key in ["B_ref", "s_plus", "s_max", \
                           "interpolation_acc", "fourier_coeff_trunc", \
                           "h_mesh", "delta_phi_mesh"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_3D" + "_" +  sub_key, "f8")
                var[...] = self["plasma"]["eq_data_3D"][sub_key]
            for sub_key in ["use_mesh", "use_symmetry"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_3D" + "_" +  sub_key, "b")
                var[...] = self["plasma"]["eq_data_3D"][sub_key]
            for sub_key in ["equilibrium_type", "vessel_filename"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_3D" + "_" +  sub_key, str, ('str_dim',))
                var[0] = self["plasma"]["eq_data_3D"][sub_key]
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                     "eq_data_3D" + "_" + \
                                                     "equilibrium_files", str, ('N_time',))
            var[:] = self["plasma"]["eq_data_3D"]["equilibrium_files"]
        else:
            for sub_key in ["R", "z"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key, "f8", \
                                                         ("N_time", "N_eq_2D_" + sub_key))
                var[:] = self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key)
            for sub_key in ["Psi", "rhop", "Br", "Bt", "Bz"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key, "f8", \
                                                         ("N_time", "N_eq_2D_R", "N_eq_2D_z"))
                var[:] = self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key)
            for sub_key in ["R_ax", "z_ax", "R_sep", "z_sep", "Psi_ax", "Psi_sep"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key, "f8", \
                                                         ("N_time",))
                var[:] = self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key)
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                     "vessel_bd", "f8", \
                                                     ("N_vessel_bd", "N_vessel_dim"))
            var[...,0] = self["plasma"]['vessel_bd'][...,0]
            var[...,1] = self["plasma"]['vessel_bd'][...,1]
        for sub_key in self["diagnostic"].keys():
            if(sub_key == "diag_name"):
                var = rootgrp["Scenario"].createVariable("diagnostic_" +  sub_key, str, \
                                                             ("N_time","N_ch"))
                var[:] = self["diagnostic"][sub_key]
            else:
                var = rootgrp["Scenario"].createVariable("diagnostic_" +  sub_key, "f8", \
                                                             ("N_time","N_ch"))
                var[:] = self["diagnostic"][sub_key]
        if(globalsettings.AUG):
            for sub_key in self["AUG"].keys():
                var = rootgrp["Scenario"].createVariable("AUG_" +  sub_key, str, \
                                                         ("str_dim",))
                var[0] = self["AUG"][sub_key]
        used_diag_dict_sub_keys = ["diags_exp", "diags_diag", \
                                   "diags_ed", "diags_Extra_arg_1",\
                                   "diags_Extra_arg_2","diags_Extra_arg_3"]
        used_diag_dict_formatted = {}
        for sub_key in used_diag_dict_sub_keys:
            used_diag_dict_formatted[sub_key] = []
        for diagname in self["used_diags_dict"]:
            cur_diag = self["used_diags_dict"][diagname]
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI")):
                used_diag_dict_formatted["diags_Extra_arg_1"].append(cur_diag.Rz_exp)
                used_diag_dict_formatted["diags_Extra_arg_2"].append(cur_diag.Rz_diag)
                used_diag_dict_formatted["diags_Extra_arg_3"].append(str(cur_diag.Rz_ed))
            elif("CT" in diagname or "IEC" in diagname):
                used_diag_dict_formatted["diags_Extra_arg_1"].append(str(cur_diag.beamline))
                used_diag_dict_formatted["diags_Extra_arg_2"].append("{0:1.8f}".format(cur_diag.pol_coeff_X))
                if(cur_diag.base_freq_140):
                    used_diag_dict_formatted["diags_Extra_arg_3"].append("True")
                else:
                    used_diag_dict_formatted["diags_Extra_arg_3"].append("False")
            else:
                used_diag_dict_formatted["diags_Extra_arg_1"].append("None")
                used_diag_dict_formatted["diags_Extra_arg_2"].append("None")
                used_diag_dict_formatted["diags_Extra_arg_3"].append("None")
            cur_diag = self["used_diags_dict"][diagname]
            if("EXT" == diagname.upper()):
                used_diag_dict_formatted["diags_exp"].append("None")
                used_diag_dict_formatted["diags_diag"].append("None")
                used_diag_dict_formatted["diags_ed"].append("-1")
            else:
                used_diag_dict_formatted["diags_exp"].append(cur_diag.exp)
                used_diag_dict_formatted["diags_diag"].append(cur_diag.diag)
                used_diag_dict_formatted["diags_ed"].append(str(cur_diag.ed))
        rootgrp["Scenario"].createVariable("used_diags_dict_" +  "diags", str, \
                                           ("N_used_diags",))
        for sub_key in used_diag_dict_sub_keys:
            rootgrp["Scenario"].createVariable("used_diags_dict_" +  sub_key, str, \
                                               ("N_used_diags",))
        for idiag, diagname in enumerate(self["used_diags_dict"].keys()):
            rootgrp["Scenario"]["used_diags_dict_" +  "diags"][idiag] = diagname
            for sub_key in used_diag_dict_sub_keys:
                rootgrp["Scenario"]["used_diags_dict_" + sub_key][idiag] = \
                    used_diag_dict_formatted[sub_key][idiag]
        if(filename is not None):
            rootgrp.close()

    def from_netcdf(self, filename=None, rootgrp=None):
        if(filename is not None):
            rootgrp = Dataset(filename, "r", format="NETCDF4")
        for sub_key in rootgrp["Scenario"].dimensions.keys():
            if(sub_key == "str_dim"):
                continue
            self["dimensions"][sub_key] = rootgrp["Scenario"].dimensions[sub_key].size
        self['shot'] = rootgrp["Scenario"]["shot"][...].item()
        self['time'] = np.array(rootgrp["Scenario"]["time"])
        self["plasma"]["2D_prof"] = bool(rootgrp["Scenario"]["plasma_" + "2D_prof"][...].item())
        self["plasma"]["eq_dim"] = rootgrp["Scenario"]["plasma_" + "eq_dim"][...].item()
        for sub_key in ["Te", "ne"]:
            self['plasma'][sub_key] = np.array(rootgrp["Scenario"]["plasma_" + sub_key])
        if(not self["plasma"]["2D_prof"]):
            self["plasma"]["prof_reference"] = rootgrp["Scenario"]["plasma_prof_reference"][0]
        self["plasma"][self["plasma"]["prof_reference"]] = np.array(rootgrp["Scenario"]["plasma_" + self["plasma"]["prof_reference"]])
        for sub_key in self["scaling"].keys():
            self['scaling'][sub_key] = float(rootgrp["Scenario"]["scaling_" + sub_key][...].item())
        if(self["plasma"]["eq_dim"] == 3):
            for sub_key in ["B_ref", "s_plus", "s_max", \
                           "interpolation_acc", "fourier_coeff_trunc", \
                           "h_mesh", "delta_phi_mesh"]:
                self['plasma']["eq_data_3D"][sub_key] = float(rootgrp["Scenario"]["plasma" + "_" + \
                                                                                  "eq_data_3D" + "_"  + sub_key][...].item())
            for sub_key in ["use_mesh", "use_symmetry"]:
                self['plasma']["eq_data_3D"][sub_key] = bool(rootgrp["Scenario"]["plasma" + "_" + \
                                                                                  "eq_data_3D" + "_"  + sub_key][...].item())
            for sub_key in ["equilibrium_type", "vessel_filename"]:
                self['plasma']["eq_data_3D"][sub_key] = rootgrp["Scenario"]["plasma" + "_" + \
                                                                            "eq_data_3D" + "_"  + sub_key][0]
            self["plasma"]["eq_data_3D"]["equilibrium_files"] = np.array(rootgrp["Scenario"]["plasma" + "_" + \
                                                                                     "eq_data_3D" + "_" + \
                                                                                     "equilibrium_files"])
        else:
            self["plasma"]['eq_data_2D'] = EQDataExt(self["shot"], Ext_data=True)
            eq_slice_data = {}
            for sub_key in ["R", "z", "Psi", "rhop", "Br", "Bt", "Bz",\
                            "R_ax", "z_ax", "R_sep", "z_sep", "Psi_ax", "Psi_sep"]:
                eq_slice_data[sub_key] = np.array(rootgrp["Scenario"]["plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key])
            self["plasma"]['eq_data_2D'].fill_with_slices_from_dict(self["time"], eq_slice_data)
            self["plasma"]['vessel_bd'] = np.array(rootgrp["Scenario"]["plasma" + "_" + \
                                                         "vessel_bd"])
        for sub_key in self["diagnostic"].keys():
            self["diagnostic"][sub_key] = np.array(rootgrp["Scenario"]["diagnostic_" + sub_key])
        if(globalsettings.AUG):
            for sub_key in self["AUG"].keys():
                self["AUG"][sub_key] = rootgrp["Scenario"]["AUG_" +  sub_key][0]
        self["dimensions"]["N_used_diags"] = len(rootgrp["Scenario"]["used_diags_dict_diags"])
        self["used_diags_dict"] = od()
        for idiag, diagname in enumerate(rootgrp["Scenario"]["used_diags_dict_diags"]):
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI")):
                self["used_diags_dict"].update({diagname: ECI_diag(diagname, \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]), \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_1"][idiag], \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_2"][idiag], \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_3"][idiag]))})
            elif("CT" in diagname or "IEC" in diagname):
                self["used_diags_dict"].update({diagname: ECRH_diag(diagname, \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]), \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_1"][idiag]), \
                                                                   float(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_2"][idiag]), \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_3"][idiag] == "True")})
            elif("EXT" == diagname.upper()):
                self["used_diags_dict"].update({diagname: EXT_diag(diagname)})
                self["used_diags_dict"][diagname].set_from_scenario_diagnostic(self["diagnostic"],0)
            else:
                self["used_diags_dict"].update({diagname: \
                        Diag(diagname, \
                             rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], \
                             rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], \
                             int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]))})
            for diag_key in self["avail_diags_dict"]:
                if(diag_key in list(self["used_diags_dict"])):
                    self["avail_diags_dict"].update({diag_key: self["used_diags_dict"][diag_key]})
        self.default_diag = list(self["used_diags_dict"].keys())[0]
        if(filename is not None):
            rootgrp.close()
        

    def autosave(self):
        try:
            os.remove(self.scenario_file)
        except FileNotFoundError:
            pass
        self.to_netcdf(filename=self.scenario_file)


    def load_dist_obj(self, filename=None, mdict=None):
        self["plasma"]["dist_obj"] = load_f_from_mat(filename, use_dist_prefix=None)
        
    def load_GENE_obj(self, filename, dstf):
        it = 0 # Only single time point supported
        try:
            if(dstf == "Ge"):
                self["plasma"]["GENE_obj"] = Gene(filename, self["plasma"]["time"][it], self["plasma"]["eq_data_2D"][it])
            else:
                self["plasma"]["GENE_obj"] = GeneBiMax(filename, self["plasma"]["time"][it], self["plasma"]["eq_data_2D"][it], it)
            return True
        except Exception as e:
            self["plasma"]["GENE_obj"] = None
            print("Failed to load the GENE data")
            print(e)
            return False
        
    def duplicate_time_point(self, it, new_times):
        # Copies the time index it len(new_times) times using the times in new_times.
        first_time = True
        plasma_dict_0 = self["plasma"].copy()
        new_plasma_dict = {}
        new_ray_launch = []
        ray_launch_0 = self.ray_launch[0]
        new_plasma_dict["vessel_bd"] = plasma_dict_0["vessel_bd"]
        new_plasma_dict["prof_reference"] = plasma_dict_0["prof_reference"]
        for time in new_times:
            for key in self["plasma"]:
                if(plasma_dict_0[key] is None or np.isscalar(plasma_dict_0[key])):
                    continue
                elif(key == "vessel_bd" or key == "prof_reference" or len(plasma_dict_0[key]) == 0):
                    continue
                if(first_time):
                    new_plasma_dict[key] = []
                if(key != "time"):
                    new_plasma_dict[key].append(plasma_dict_0[key][it])
                elif(key == "time" ):
                    new_plasma_dict[key].append(time)
            first_time = False
            new_ray_launch.append({})
            for key in ray_launch_0:
                new_ray_launch[-1][key] = ray_launch_0[key]
        self["time"] = np.array(new_plasma_dict["time"])
        self["plasma"] = new_plasma_dict
        self["diagnostic"] = new_ray_launch
        
    def integrate_GeneData(self, used_times):
        # We need this to hack in extra time points for the GENE computation
        if(self.GENE_obj == None):
            print("Gene object not initialized")
            return False
        new_times = []
        for used_time in used_times:
            it_gene = np.argmin(np.abs(self.GENE_obj.time - (used_time - self["time"][0])))
            new_times.append(self["time"][0] + self.GENE_obj.time[it_gene])
        new_times = np.array(new_times)
        self.duplicate_time_point(0, new_times)
        return True

if(__name__ == "__main__"):
    newScen = ECRadScenario(noLoad=True)
    newScen.from_mat( path_in="/mnt/c/Users/Severin/ECRad/ECRad_35662_ECECTCCTA_ed10.mat")
    newScen.to_netcdf("/mnt/c/Users/Severin/ECRad/ECRad_35662_ECECTCCTA_Scenario.nc")
    newScen = ECRadScenario(noLoad=True)
    newScen.from_netcdf("/mnt/c/Users/Severin/ECRad/ECRad_35662_ECECTCCTA_Scenario.nc")
    newScen.from_mat( path_in="/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_ed19.mat")
    newScen.to_netcdf("/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_Scenario.nc")
    newScen = ECRadScenario(noLoad=True)
    newScen.from_netcdf("/mnt/c/Users/Severin/ECRad_regression/W7X/ECRad_20180823016002_EXT_Scenario.nc")
     
    
    
    
    