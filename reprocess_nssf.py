'''
Created on Jun 12, 2018

@author: sdenk
'''
shot_in_folder = '/tokp/work/sdenk/nssf/'
shot_out_folder = '/tokp/work/sdenk/nssf/'
import os
from get_ECRH_config import get_ECRH_viewing_angles
from shotfile_handling_AUG import get_ECI_launch
from ECRad_Interface import prepare_input_file
import numpy as np
import subprocess
import shlex
from ECRad_Config import ECRadConfig
from ECRad_DIAG_AUG import DefaultDiagDict
from shotfile_handling_AUG import load_IDA_data
from shutil import copytree, rmtree, copyfile
from time import clock

def reprocess(shot, time, ed_list, Config, ssh=True, Overwrite=False):
    Config.IDA_ed, time, Config.plasma_dict = load_IDA_data(Config.shot, \
                                                     time, Config.IDA_exp, Config.IDA_ed)
    Config.gy_dict = {}
    start = clock()
    in_dir = os.path.join(shot_in_folder, str(shot), "{0:1.2f}".format(time[0]), "OERT")
    if(not os.path.isdir(in_dir)):
        print("In folder does not exist")
    if(Overwrite):
        out_dir = in_dir
    else:
        out_dir = os.path.join(shot_out_folder)
    if(not os.path.isdir(out_dir)):
            os.mkdir(out_dir)
    for add_path in [str(shot), "{0:1.2f}".format(time[0]), "OERT"]:
        out_dir = os.path.join(out_dir, add_path)
        if(not os.path.isdir(out_dir)):
            os.mkdir(out_dir)
    for diag_key in Config.used_diags_dict.keys():
        if("CT" in diag_key or "IEC" == diag_key):
            new_gy = get_ECRH_viewing_angles(shot, \
                                             Config.used_diags_dict[diag_key].beamline, \
                                             Config.used_diags_dict[diag_key].base_freq_140)
            if(new_gy.error == 0):
                Config.gy_dict[str(Config.used_diags_dict[diag_key].beamline)] = new_gy
            else:
                print("Error when reading viewing angles")
                print("Launch aborted")
                return
        if(diag_key in ["ECN", "ECO", "ECI"]):
            Config.ECI_dict = get_ECI_launch(Config.used_diags_dict[diag_key], shot)
    for ed in ed_list:
        if(not Overwrite):
            ed_out = 0
            while(os.path.isdir(os.path.join(out_dir, "ed_" + str(ed_out)) + os.path.sep)):
                ed_out += 1
            Config.working_dir = os.path.join(out_dir, "ed_" + str(ed_out)) + os.path.sep
            ed_in_dir = os.path.join(in_dir, "ed_" + str(ed)) + os.path.sep
            ECRad_in_dir = os.path.join(ed_in_dir, "ECRad_data")
            if(not (os.path.isfile(os.path.join(ECRad_in_dir, "comment")) and os.path.isfile(os.path.join(ed_in_dir, "ida.log")))):
                print("Skipping empty edition")
                continue
            if(not os.path.isdir(Config.working_dir)):
                os.mkdir((Config.working_dir))
            copyfile(os.path.join(ed_in_dir, "ida.log"), os.path.join(Config.working_dir, "ida.log"))
            ECRad_out_dir = os.path.join(Config.working_dir, "ECRad_data")
            if(not os.path.isdir(ECRad_out_dir)):
                os.mkdir(ECRad_out_dir)
            copyfile(os.path.join(ECRad_in_dir, "comment"), os.path.join(ECRad_out_dir, "comment"))
            fRe_in_dir = os.path.join(ECRad_in_dir, "fRe")
            fRe_out_dir = os.path.join(ECRad_out_dir, "fRe")
            if(os.path.isdir(fRe_out_dir)):
                rmtree(fRe_out_dir)
            copytree(fRe_in_dir, fRe_out_dir)
        else:
            Config.working_dir = os.path.join(in_dir, "ed_" + str(ed)) + os.path.sep
        if(not prepare_input_file(time, 0, Config, False, copy_dist=False)):
            print("Error!! Launch aborted")
        if("ECN" in Config.used_diags_dict.keys()  or "ECO"  in Config.used_diags_dict.keys()  or "ECI" in Config.used_diags_dict.keys()):
            stacksize = int(np.ceil(Config.max_points_svec * 3.125) * 3)
        else:
            stacksize = int(np.ceil(Config.max_points_svec * 3.125))
        invoke = []
        if(ssh):
            invoke.append("ssh")
            invoke.append("tokp01.itm")
            invoke.append("\"cd")
            invoke.append(Config.working_dir)
            invoke.append("&&")
            invoke.append("setenv")
            invoke.append("ECRad_working_dir_1")
            invoke.append(Config.working_dir)
            invoke.append("&&")
            invoke.append("setenv OMP_STACKSIZE")
            invoke.append("{0:d}k".format(stacksize))
            invoke.append("&&")
            invoke.append("setenv ECRad")
            invoke.append("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model_parallel_dev_light/ECRad_model")
            invoke.append("&&")
            invoke.append("setenv")
            invoke.append("OMP_NUM_THREADS")
            invoke.append("{0:d}".format(Config.parallel_cores))
            invoke.append("&&")
        invoke.append("qsub")
        invoke.append("-N")
        invoke.append("E{0:5d}{2:d}{1:1.1f}".format(Config.shot, time[0], ed))
        invoke.append("-l")
        invoke.append("h_rt={0:02d}:00:00".format(Config.wall_time))
        if(ssh):
            invoke.append("-pe")
            invoke.append("'impi_hydra.*\'")
            invoke.append("{0:d}".format(Config.parallel_cores))
        else:
            invoke.append("-pe")
            invoke.append("openmp")
            invoke.append("{0:d}".format(Config.parallel_cores))
        if(ssh):
            invoke.append("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model_parallel_dev_light/batch_submit_ECRad_parallel_nosync.sge\"")
        else:
            invoke.append("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model_parallel_dev_light/batch_submit_ECRad_parallel_toks.sge")
        os.environ['ECRad'] = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model_parallel_dev_light/ECRad_model"
        os.environ['ECRad_working_dir_1'] = Config.working_dir
        os.environ['WALLTIME'] = r"{0:2d}:00:00".format(Config.wall_time)
        os.environ['OMP_STACKSIZE'] = r"{0:d}k".format(stacksize)
        os.environ['OMP_NUM_THREADS'] = "{0:d}".format(Config.parallel_cores)
        print("setenv ECRad /afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model_parallel_dev_light/ECRad_model")
        print("setenv ECRad_working_dir_1 " + Config.working_dir)
        print("setenv WALLTIME "  r"{0:2d}:00:00".format(Config.wall_time))
        print("setenv OMP_STACKSIZE " + r"{0:d}k".format(stacksize))
        print("setenv OMP_NUM_THREADS " + r"{0:d}".format(Config.parallel_cores))
        call = invoke[0]
        for arg in invoke[1:len(invoke)]:
            call += " " + arg
        print(call)
        print(shlex.split(call))
        ECRad_process = subprocess.Popen(shlex.split(call))
        ECRad_process.wait()
        if(ssh):
            print("Submitted job to tokp queue.")
        else:
            print("Submitted job to toks queue.")
#        print(elapsed - start)
#        return

def reprocess_34663():
    shot = 34663
    time = np.array([3.60])
    Config = ECRad_Config()
    ECE_diag = DefaultDiagDict["ECE"]
    Config.used_diags_dict.update({"ECE" : ECE_diag})
    CTA_diag = DefaultDiagDict["CTA"]
    CTA_diag.beamline = 7
    Config.used_diags_dict.update({"CTA" : CTA_diag})
    Config.shot = shot
    Config.time = np.copy(time)
    Config.IDA_exp = "SDENK"
    Config.IDA_ed = 8
    Config.EQ_diag = "IDE"
    Config.wall_time = 12
    Config.N_ray = 82
    Config.debug = True
    Config.dstf = "Re"
    Config.parallel_cores = 16
    Config.considered_modes = 3
    Config.large_ds = 30.e-4
    Config.small_ds = 30.e-5
#    ed_list = np.arange(96, 122, 1, dtype=np.int)
    ed_list = [8]
    reprocess(shot, time, ed_list, Config, Overwrite=False)

def reprocess_33697():
    shot = 33697
    time = np.array([4.80])
    Config = ECRad_Config()
    ECE_diag = DefaultDiagDict["ECE"]
    Config.used_diags_dict.update({"ECE" : ECE_diag})
    CTA_diag = DefaultDiagDict["CTA"]
    CTA_diag.beamline = 6
    Config.used_diags_dict.update({"CTA" : CTA_diag})
    Config.shot = shot
    Config.time = np.copy(time)
    Config.IDA_exp = "SDENK"
    Config.IDA_ed = 27
    Config.EQ_diag = "IDE"
    Config.wall_time = 12
    Config.N_ray = 82
    Config.debug = True
    Config.dstf = "Re"
    Config.parallel_cores = 16
    Config.considered_modes = 3
    Config.large_ds = 30.e-4
    Config.small_ds = 30.e-5
#    ed_list = np.arange(147, 172, 1, dtype=np.int)  # 172
    ed_list = [8]
    reprocess(shot, time, ed_list, Config, Overwrite=False)

def reprocess_33705():
    shot = 33705
    time = np.array([4.90])
    Config = ECRad_Config()
    ECE_diag = DefaultDiagDict["ECE"]
    Config.used_diags_dict.update({"ECE" : ECE_diag})
    CTA_diag = DefaultDiagDict["CTA"]
    CTA_diag.beamline = 6
    Config.used_diags_dict.update({"CTA" : CTA_diag})
    Config.shot = shot
    Config.time = np.copy(time)
    Config.IDA_exp = "SDENK"
    Config.IDA_ed = 6
    Config.EQ_diag = "IDE"
    Config.wall_time = 12
    Config.N_ray = 82
    Config.debug = True
    Config.dstf = "Re"
    Config.parallel_cores = 16
    Config.considered_modes = 3
    Config.large_ds = 30.e-4
    Config.small_ds = 30.e-5
#    ed_list = np.arange(82, 104, 1, dtype=np.int)
    ed_list = [8]
    reprocess(shot, time, ed_list, Config, Overwrite=False)

if(__name__ == "__main__"):
    reprocess_33697()
    reprocess_33705()
    reprocess_34663()
#
