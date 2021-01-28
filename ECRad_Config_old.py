# -*- coding: utf-8 -*-
import os
from scipy.io import loadmat, savemat

class ECRadConfig:
    def __init__(self, noLoad =False):
        self.default_config_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_Default.mat")
        if(noLoad):
            self.from_mat_file()
        else:
            try:
                self.from_mat_file(path=self.default_config_file, default=True)
            except IOError:
                self.from_mat_file()

    def from_mat_file(self, mdict=None, path=None, default=False):
        ext_mdict = False
        temp_config = None
        if(mdict is not None or path is not None):
            if(path is not None):
                mdict = loadmat(path, chars_as_strings=True, squeeze_me=True)
            else:
                ext_mdict = True
        else:
            mdict = provide_default_mdict()
        if(os.path.isdir(mdict["working_dir"])):
            self.working_dir = mdict["working_dir"]
        elif(not ext_mdict):
            self.working_dir = mdict["working_dir"]
        else:
            print("Warning working dir not imported, since it is not a valid directory")
            print("Falling back to last used working directory")
            temp_config = ECRadConfig()
            self.working_dir = temp_config.working_dir
        try:
            if(os.path.isdir(mdict["scratch_dir"])):
                self.scratch_dir = mdict["scratch_dir"]
            elif(not ext_mdict):
                self.scratch_dir = mdict["scratch_dir"]
            else:
                print("Warning working dir not imported, since it is not a valid directory")
                print("Falling back to last used working directory")
                if(temp_config is None):
                    temp_config = ECRadConfig()
                self.scratch_dir = temp_config.scratch_dir
        except KeyError:
            print("Scratch dir not set in ECRad config -> failling back to working directory")
            self.scratch_dir = self.working_dir
        self.dstf = mdict["dstf"]
        self.extra_output = mdict["extra_output"]
        try:
            self.debug = mdict["debug"]
        except KeyError:
            self.debug = False
        try:
            self.batch = mdict["batch"]
        except KeyError:
            self.batch = False
        try:
            self.parallel = mdict["parallel"]
        except KeyError:
            self.parallel = False
        try:
            self.parallel_cores = mdict["parallel_cores"]
        except KeyError:
            self.parallel_cores = 16
        if(not default):
            try:
                self.use_ext_rays = mdict["use_ext_rays"]
            except KeyError:
                self.use_ext_rays = False
        else:
            self.use_ext_rays = False
        try:
            self.wall_time = mdict["wall_time"]
        except KeyError:
            self.wall_time = 2
        try:
            self.vmem = mdict["vmem"]
        except KeyError:
            self.vmem = 32000
        self.raytracing = mdict["raytracing"]
        self.ripple = mdict["ripple"]
        self.weak_rel = mdict["weak_rel"]
        self.N_freq = mdict["N_freq"]
        self.N_ray = mdict["N_ray"]
        try:
            self.ratio_for_3rd_harm = mdict["ratio_for_3rd_harm"]
        except KeyError:
            self.ratio_for_3rd_harm = 0.4
        self.considered_modes = mdict["considered_modes"]  # :
            # 1 -> Only X
            # 2 -> Only O
            # 3 -> Both
        self.mode_conv = mdict["mode_conv"]
        try:
            self.reflec_model = mdict["reflec_model"]
        except KeyError:
            self.reflec_model = 0
        try:
            self.reflec_X = mdict["reflec_X"]
        except KeyError:
            try:
                self.reflec_X = mdict["reflec"]
            except KeyError:
                self.reflec_X = 0.92
        try:
            self.reflec_O = mdict["reflec_O"]
        except KeyError:
            self.reflec_O = 0.95
        self.gene_obj = None
        self.Te_filename = "Te_file.dat"
        self.ne_filename = "ne_file.dat"
        try:
            self.R_shift = mdict["R_shift"]
        except KeyError:
            self.R_shift = 0.0
        try:
            self.z_shift = mdict["z_shift"]
        except KeyError:
            self.z_shift = 0.0
        try:
            self.large_ds = mdict["large_ds"]
        except KeyError:
            self.large_ds = 25.e-4
        try:
            self.small_ds = mdict["small_ds"]
        except KeyError:
            self.small_ds = 25.e-5
        try:
            self.max_points_svec = mdict["max_points_svec"]
        except KeyError:
            self.max_points_svec = 20000
        if(path is None and not ext_mdict):
            print("Successfully loaded last used configuration")
        return

    def autosave(self):
        config_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_Default.mat")
        self.saveconfig(path=config_file)

    def saveconfig(self, mdict=None, path=None):
        write_mat = False
        if(mdict is None):
            if(path is None):
                print("Either mdict or path need to be provided")
                raise ValueError
            mdict = {}
            write_mat = True
        mdict["working_dir"] = self.working_dir
        mdict["scratch_dir"] = self.scratch_dir
        mdict["dstf"] = self.dstf
        mdict["extra_output"] = self.extra_output
        mdict["debug"] = self.debug
        mdict["batch"] = self.batch
        mdict["parallel"] = self.parallel
        mdict["parallel_cores"] = self.parallel_cores
        mdict["wall_time"] = self.wall_time
        mdict["vmem"] = self.vmem
        mdict["raytracing"] = self.raytracing
        mdict["ripple"] = self.ripple
        mdict["weak_rel"] = self.weak_rel
        mdict["N_freq"] = self.N_freq
        mdict["N_ray"] = self.N_ray
        mdict["ratio_for_3rd_harm"] = self.ratio_for_3rd_harm
        mdict["considered_modes"] = self.considered_modes
        mdict["mode_conv"] = self.mode_conv
        mdict["reflec_model"] = self.reflec_model
        mdict["reflec_X"] = self.reflec_X
        mdict["reflec_O"] = self.reflec_O
        mdict["R_shift"] = self.R_shift
        mdict["z_shift"] = self.z_shift
        mdict["large_ds"] = self.large_ds
        mdict["small_ds"] = self.small_ds
        mdict["max_points_svec"] = self.max_points_svec
        mdict["use_ext_rays"] = self.use_ext_rays
        if(write_mat):
            try:
                savemat(path, mdict, appendmat=False)
                print("Successfully created: ", path)
            except TypeError as e:
                print("Failed to save to .mat")
                print(e)
                print(mdict)
        else:
            return mdict



def provide_default_mdict():
    mdict = {}
    mdict["shot"] = 0
    mdict["working_dir"] = ""
    mdict["scratch_dir"] = ""
    mdict["dstf"] = "Th"
    mdict["extra_output"] = True
    mdict["debug"] = False
    mdict["batch"] = True
    mdict["parallel"] = True
    mdict["parallel_cores"] = 32
    mdict["wall_time"] = 2
    mdict["vmem"] = 32000
    mdict["raytracing"] = True
    mdict["ripple"] = True
    mdict["weak_rel"] = True
    mdict["N_freq"] = 1
    mdict["N_ray"] = 1
    mdict["ratio_for_3rd_harm"] = 0.4
    mdict["considered_modes"] = 1
    mdict["mode_conv"] = 0.0
    mdict["reflec_model"] = 0
    mdict["reflec_X"] = 0.9
    mdict["reflec_O"] = 0.9
    mdict["IDA_exp"] = "AUGD"
    mdict["IDA_ed"] = 0
    mdict["EQ_exp"] = "AUGD"
    mdict["EQ_diag"] = "EQH"
    mdict["EQ_ed"] = 0
    mdict["default_diag"] = "ECE"
    mdict["R_shift"] = 0.0
    mdict["z_shift"] = 0.0
    mdict["large_ds"] = 2.5e-3
    mdict["small_ds"] = 2.5e-4
    mdict["max_points_svec"] = 20000
    mdict["use_ext_rays"] = False
    return mdict

# InvokeECRad = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/Ecfm_Model_new/ecfm_model"
# test_config = ECRadConfig()
