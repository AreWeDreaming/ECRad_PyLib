# -*- coding: utf-8 -*-
import os
from scipy.io import loadmat
from netCDF4 import Dataset

class ECRadConfig(dict):
    def __init__(self, noLoad = False):
        self.default_config_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_Default.nc")
        if(noLoad):
            self.reset()
        else:
            try:
                self.reset()
                self.load(filename=self.default_config_file, default=True)
            except IOError:
                self.reset()

    def reset(self):
        self.main_keys = ["Physics", "Execution", "Numerics"]
        self.sub_keys = {}
        for key in self.main_keys:
            self[key] = {}
        self.sub_keys["Physics"] = ["dstf", "raytracing", "ripple", \
                                    "weak_rel", "considered_modes", \
                                    "N_freq", "N_ray", \
                                    "ratio_for_3rd_harm", "N_max", "tau_ignore", "mode_conv" ,\
                                    "reflec_X", "reflec_O", \
                                    "R_shift", "z_shift", \
                                    "use_ext_rays"]
        self.sub_keys["Execution"] = ["working_dir", "scratch_dir", "extra_output", \
                                      "debug", "batch", "parallel" , \
                                      "parallel_cores", "wall_time", \
                                      "vmem"]
        self.sub_keys["Numerics"] = ["large_ds", "small_ds", "max_points_svec" , \
                                     "N_BPD"]
        self["Physics"]["dstf"] = "Th"
        self["Physics"]["raytracing"] = True
        self["Physics"]["ripple"] = False
        self["Physics"]["weak_rel"] = True
        self["Physics"]["N_freq"] = 1
        self["Physics"]["N_ray"] = 1
        self["Physics"]["ratio_for_3rd_harm"] = 0.4
        self["Physics"]["N_max"] = 3
        self["Physics"]["tau_ignore"] = 1.e-8
        self["Physics"]["considered_modes"] = 1
        # 1 -> Only X
        # 2 -> Only O
        # 3 -> Both
        self["Physics"]["mode_conv"] = 0.0
        self["Physics"]["reflec_X"] = 0.9
        self["Physics"]["reflec_O"] = 0.9
        self["Physics"]["R_shift"] = 0.0
        self["Physics"]["z_shift"] = 0.0
        self["Physics"]["use_ext_rays"] = False
        self["Execution"]["working_dir"] = ""
        self["Execution"]["scratch_dir"] = ""
        self["Execution"]["extra_output"] = True
        self["Execution"]["debug"] = False
        self["Execution"]["batch"] = True
        self["Execution"]["parallel"] = True
        self["Execution"]["parallel_cores"] = 32
        self["Execution"]["wall_time"] = 2.0
        self["Execution"]["vmem"] = 32000
        self["Numerics"]["large_ds"] = 2.5e-3
        self["Numerics"]["small_ds"] = 2.5e-4
        self["Numerics"]["max_points_svec"] = 20000
        self["Numerics"]["N_BPD"] = 2000
        self.types = {"working_dir":str, "scratch_dir":str, "dstf":str, "extra_output": "b", 
                      "debug": "b", "batch": "b", "parallel": "b",  
                      "parallel_cores": "i8", "wall_time": "f8", 
                      "vmem": "i8", "raytracing": "b", "ripple": "b", 
                      "weak_rel": "b", "N_freq" : "i8",  "N_ray": "i8", 
                      "ratio_for_3rd_harm": "f8", "N_max" : "i8","tau_ignore" :"f8", "considered_modes" : "i8", 
                      "mode_conv" : "f8", "reflec_X" : "f8","reflec_O" : "f8", 
                      "R_shift" : "f8","z_shift" : "f8","large_ds" : "f8",
                      "small_ds" : "f8","max_points_svec" : "i8","use_ext_rays" : "b", 
                      "N_BPD" : "i8"}
        self.nice_labels = {"working_dir":"Working dir.", "scratch_dir":"Scratch dir.","dstf":"Distribution type", "extra_output": "Extra output", 
                            "debug": "Debug", "batch": "Batch", "parallel": "Parallel",  
                            "parallel_cores": "# cores", "wall_time": "wall time [h]", 
                            "vmem": "virtual memory [MB]", "raytracing": "Raytracing", "ripple": "Magn. field Ripple", 
                            "weak_rel": "Relativistic cor. for rt.", "N_freq" : "# frequencies",  "N_ray": "# rays", 
                            "ratio_for_3rd_harm": "omega_c/omega for higher harmonics", "N_max": "Highest harmonic to include","tau_ignore": "tau threshhold for computation of alpha/j",
                            "considered_modes" : "Modes to consider", 
                            "mode_conv" : "mode conv. ratio", "reflec_X" : "Wall refl. coeff. X-mode", 
                            "reflec_O" : "Wall refl. coeff. O-mode", "use_ext_rays" : "Use ext rays", 
                            "R_shift" : "R shift [m]","z_shift" : "z shift [m]", "large_ds" :  "Large step size [m]",
                            "small_ds" : "Small step size [m]","max_points_svec" : "Max points on LOS", 
                            "N_BPD" : "Points for BPD"}
        
    def load(self, filename=None, mdict=None, rootgrp=None, default=False):
        if(filename is not None):
            ext = os.path.splitext(filename)[1]
            if(ext == ".mat"):
                self.from_mat(path_in=filename)
            elif(ext == ".nc"):
                self.from_netcdf(filename=filename,default=default)
            else:
                print("Extension " + ext + " is unknown")
                raise(ValueError)
        elif(mdict is not None):
            self.from_mat(mdict)
        elif(rootgrp is not None):
            self.from_netcdf(rootgrp=rootgrp)
    
    def from_mat(self, mdict=None, path_in=None):
        ext_mdict = False
        temp_config = None
        if(mdict is not None or path_in is not None):
            if(path_in is not None):
                mdict = loadmat(path_in, chars_as_strings=True, squeeze_me=True)
            else:
                ext_mdict = True
        else:
            raise ValueError("Either filename or mdict must be present")
        self.reset()
        key = "Execution"
        for sub_key in ["working_dir", "scratch_dir"]:
            if(os.path.isdir(mdict[sub_key])):
                self[key][sub_key] = mdict[sub_key]
            elif(not ext_mdict):
                self[key][sub_key] = mdict[sub_key]
            else:
                print("Warning working dir not imported, since it is not a valid directory")
                print("Falling back to last used working directory")
                temp_config = ECRadConfig()
                self[key][sub_key] = temp_config[key][sub_key]
        for key in self.main_keys:
            for sub_key in self.sub_keys[key]:
                if(sub_key in ["working_dir", "scratch_dir"]):
                    continue
                else:
                    if(sub_key in mdict.keys()):
                        self[key][sub_key] = mdict[sub_key]
                    else:
                        print("Could not find " + sub_key + " in config file.")
        if(path_in is None and not ext_mdict):
            print("Successfully loaded last used configuration")
        return
    
    def to_netcdf(self, filename=None, rootgrp=None):
        if(filename is not None):
            rootgrp = Dataset(filename, "w", format="NETCDF4")
        rootgrp.createGroup("Config")
        rootgrp["Config"].createDimension('str_dim', 1)
        for key in self.main_keys:
            for sub_key in self.sub_keys[key]:
                if(self.types[sub_key] == str):
                    var = rootgrp["Config"].createVariable(key + "_" + sub_key, self.types[sub_key], 'str_dim')
                    var[0] = self[key][sub_key]
                else:
                    var = rootgrp["Config"].createVariable(key + "_" + sub_key, self.types[sub_key])
                    var[...] = self[key][sub_key]
        if(filename is not None):
            rootgrp.close()
        
    def from_netcdf(self, filename=None, rootgrp=None, default=False):
        self.reset()
        if(filename is not None):
            rootgrp = Dataset(filename, "r", format="NETCDF4")
        key = "Execution"
        for sub_key in ["working_dir", "scratch_dir"]:
            if(os.path.isdir(rootgrp["Config"][key + "_" + sub_key][0])):
                self[key][sub_key] = rootgrp["Config"][key + "_" + sub_key][0]
            elif(not default):
                print("Warning " + sub_key + " not imported, since it is not a valid directory")
                print("Falling back to last used  " + sub_key)
                temp_config = ECRadConfig()
                self[key][sub_key] = temp_config["Execution"][sub_key]
            else:
                continue
        for key in self.main_keys:
            for sub_key in self.sub_keys[key]:
                try:
                    if(sub_key in ["working_dir", "scratch_dir"]):
                        continue
                    if(self.types[sub_key] == "b"):
                        self[key][sub_key] = bool(rootgrp["Config"][key + "_" + sub_key][...])
                    elif(self.types[sub_key] == str):
                        self[key][sub_key] = rootgrp["Config"][key + "_" + sub_key][0]
                    else:
                        self[key][sub_key] = rootgrp["Config"][key + "_" + sub_key][...].item()
                except IndexError:
                    print("WARNING: Cannot find {0:s} in Config file.".format(key + "/" + sub_key))
                    print("INFO: Using default value.")
        if(filename is not None):
            rootgrp.close()

    def autosave(self):
        config_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_Default.nc")
        self.to_netcdf(filename=config_file)


if(__name__ == "__main__"):
    newConf = ECRadConfig(noLoad=True)
    newConf.load( filename="/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc")
    newConf.to_netcdf("/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_Config.nc")
