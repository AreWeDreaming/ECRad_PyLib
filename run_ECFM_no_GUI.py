'''
Created on Mar 30, 2017

@author: sdenk
'''

import sys
from shotfile_handling_AUG import get_ECI_launch
from ECRad_Config import ECRadConfig
from Diags import  Diag, ECRH_diag, ECI_diag, EXT_diag, TCV_diag, TCV_CCE_diag
from ECRad_Interface import load_plasma_from_mat, prepare_input_file
from equilibrium_utils_AUG import EQData
from shotfile_handling_AUG import load_IDA_data, get_diag_data_no_calib, get_freqs, get_divertor_currents, filter_CTA, get_data_calib, get_ECI_launch
from ECRad_DIAG_AUG import DefaultDiagDict
from get_ECRH_config import get_ECRH_viewing_angles
import numpy as np
from ECRad_Results import ECRadResults
import os
import shutil
import subprocess

def run_ECRad():
    args = sys.argv
    working_dir = args[1]
    shot = int(args[2])
    diag_id = args[3]
    shot_data_file = os.path.join(working_dir, args[4])
    times_to_analyze = None
    try:
        if(len(args) > 5):
            try:
                times_to_analyze = np.loadtxt(os.path.join(working_dir, args[5]))
                if(times_to_analyze.ndim == 0):
                    times_to_analyze = np.array([times_to_analyze])
            except:
                print("Failed to load " + os.path.join(working_dir, args[5]))
                print("Analyzing everything")
    except Exception as e:
        print("Initialization failed.")
        print("Error:", e)
        print("Usage: python run_ECRad_no_GUI.py <shotno> <diag id>")
        if(TCV):
            print("Possible diag ids: UCE, LCE, VCE, CCE")
        elif(AUG):
            print("Possible diag ids: ECE, CTA, CTC, IEC, ECN, ECO")
        else:
            print("Only AUG or TCV supported at this time")
        print("Got the following args", args)
        return -1
    try:
        Config = ECRadConfig()
        Config.from_mat_file(path=os.path.join(working_dir, "UserConfig.mat"))
    except IOError:
        print("Failed to load user config at : ")
        print(os.path.join(working_dir, "UserConfig.mat"))
    Config.working_dir = working_dir
    if(TCV):
        Config.time, Config.plasma_dict = load_plasma_from_mat(Config, shot_data_file)
        if(diag_id in ["UCE", "LCE", "VCE"]):
            Config.used_diags_dict.update({diag_id: Diag(diag_id, "TCV", diag_id, 0)})
        elif(diag_id == "CCE"):
            launch_geo = make_CCE_diag_launch(shot, shot_data_file)
            Config.used_diags_dict.update({diag_id: TCV_CCE_diag(diag_id, "TCV", diag_id, 0, launch_geo)})
        else:
            print("Selected diag_id {0:s} is not supported for TCV".format(diag_id))
            return -1
        Config.Ext_plasma = True
    elif(AUG):
        Config.time, Config.plasma_dict = load_IDA_data(shot, timepoints=None, exp="AUGD", ed=0)
        if(diag_id == "ECE"):
            Config.used_diags_dict.update({diag_id: Diag(diag_id, "AUGD", 'RMD', 0)})
        elif(diag_id in ["CTA", "CTC", "IEC"]):
            if(shot > 33724):
                if(diag_id == "CTA"):
                    beamline = 7
                elif(diag_id == "CTC"):
                    beamline = 8
                else:
                    beamline = 5
            else:
                if(diag_id == "CTA"):
                    beamline = 6
                else:
                    beamline = 5
            Config.used_diags_dict.update({diag_id: ECRH_diag(diag_id, "AUGD", diag_id, 0, beamline, 1.0, True)})
        elif(diag_id in ["ECN", "ECO"]):
            if(diag_id == "ECN"):
                Config.used_diags_dict.update({diag_id: ECI_diag(diag_id, "AUGD", "TDI", 0, "ECEI", "RZN", 0)})
            else:
                Config.used_diags_dict.update({diag_id: ECI_diag(diag_id, "AUGD", "TDI", 0, "ECEI", "RZO", 0)})
            Config.ECI_dict = get_ECI_launch(Config.used_diags_dict[diag_id], Config.shot)
        else:
            print("Selected diag_id {0:s} is not supported for TCV".format(diag_id))
            return -1
    else:
        print("Only AUG or TCV supported at this time")
    if(Config.time == None):
        print("Failed to initialize")
        return -1
    Config.shot = shot
    Config.working_dir = working_dir
    for diag_key in Config.used_diags_dict:
        if("CT" in diag_key or "IEC" == diag_key):
            if(str(Config.used_diags_dict[diag_key].beamline) not in Config.gy_dict):
                new_gy = get_ECRH_viewing_angles(Config.shot, \
                                                Config.used_diags_dict[diag_key].beamline)
                if(new_gy.error == 0):
                    Config.gy_dict[str(Config.used_diags_dict[diag_key].beamline)] = new_gy
                else:
                    print("Error when reading viewing angles")
                    print("Launch aborted")
                    return
    Results = ECRadResults()
    Results.parse_config(Config, Config.time)
    if(not times_to_analyze is None):
        it_ext = 0
        actual_times = []
        for i in range(len(times_to_analyze)):
            actual_times.append(np.argmin(np.abs(Config.time - times_to_analyze[it_ext])))
        actual_times = np.array(actual_times)
        Config.time = Config.time[actual_times]
        Config.plasma_dict["Te"] = Config.plasma_dict["Te"][actual_times]
        Config.plasma_dict["ne"] = Config.plasma_dict["ne"][actual_times]
        Config.plasma_dict["rhop"] = Config.plasma_dict["rhop"][actual_times]
        Config.plasma_dict["ne_rhop_scale"] = Config.plasma_dict["ne_rhop_scale"][actual_times]
        if(Config.plasma_dict["eq_data"] is not None):
            Config.plasma_dict["eq_data"] = Config.plasma_dict["eq_data"][actual_times]
    if(Config.debug and AUG and not itm):
        InvokeECRad = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model_new/ECRad_model"
    elif(not Config.debug and AUG and not itm):
        InvokeECRad = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model/ECRad_model"
    elif(AUG and itm):
        InvokeECRad = "/marconi_work/eufus_gw/work/g2sdenk/ECRad_Model_parallel/ECRad_model"
    elif(TCV and Config.debug):
        InvokeECRad = "../ECRad_Model_TCV/ECRad_model"
    elif(TCV):
        InvokeECRad = "../ECRad_Model_TCV_no_debug/ECRad_model"
    else:
        print('Neither AUG nor TCV selected - no Machine!!')
        raise IOError
    Results = run_ECRad_from_script(working_dir, shot, Config, InvokeECRad, working_dir)
    Results.to_mat_file()
    print("Finished successfully")

def run_ECRad_from_script(working_dir, shot, Config, InvokeECRad, args):
    Results = ECRadResults()
    Results.parse_config(Config, Config.time)
    next_time_index_to_analyze = 0
    print("Analyzing {0:d} time points".format(len(Config.time)))
    while True:
        if(next_time_index_to_analyze >= len(Config.time)):
            break
        print("Working on t = {0:1.4f}".format(Config.time[next_time_index_to_analyze]))
        if(os.path.isdir(os.path.join(Config.working_dir, "ECRad_data"))):
            shutil.rmtree(os.path.join(Config.working_dir, "ECRad_data"))
        for diag_key in Config.used_diags_dict:
            if("CT" in diag_key or "IEC" == diag_key):
                if(str(Config.used_diags_dict[diag_key].beamline) not in Config.gy_dict):
                    new_gy = get_ECRH_viewing_angles(Config.shot, \
                                                     Config.used_diags_dict[diag_key].beamline)
                    if(new_gy.error == 0):
                        Config.gy_dict[str(Config.used_diags_dict[diag_key].beamline)] = new_gy
                    else:
                        print("Error when reading viewing angles")
                        print("Launch aborted")
                        return
        if(not prepare_input_file(Config.time, next_time_index_to_analyze, Config)):
            print("Error!! Launch aborted")
            return None
        print("-------- Launching ECRad -----------\n")
        print("-------- INVOKE COMMAND------------\n")
        print(InvokeECRad + " " + Config.working_dir)
        ECRad_process = subprocess.Popen([InvokeECRad, args])
        print("-----------------------------------\n")
        ECRad_process = subprocess.Popen([InvokeECRad, Config.working_dir])
        ECRad_process.wait()
        next_time_index_to_analyze += 1
        try:
            Results.append_new_results()
        except (IOError, IndexError) as e:
            print("Error: Results of ECRad cannot be found")
            print("Most likely cause is an error that occurred within the ECRad")
            print("Please run the ECRad with current input parameters in a separate shell.")
            print("The command to launch the ECRad can be found above.")
            print("Afterwards please send any error messages that appear at sdenk|at|ipp.mpg.de")
            print("If no errors occur make sure that you don't have another instance of ECRad GUI working in the same working directory")
            print(e)
            return None

    Results.tidy_up()
    print("Finished!")
    return Results

if(__name__ == "__main__"):
    val = run_ECRad()
    exit(val)





