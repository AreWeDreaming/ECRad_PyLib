'''
Created on Jan 27, 2021

@author: denk
'''
from Global_Settings import globalsettings
import os 
import numpy as np

def SetupECRadBatch(Config, Scenario):
    if(not Config["Execution"]["batch"]):
        raise ValueError("SetupECRadBatch should only be used with batch == true")
    # Determine OMP stacksize
    parallel = Config["Execution"]["parallel"]
    parallel_cores = Config["Execution"]["parallel_cores"]
    if(parallel and parallel_cores > globalsettings.max_cores):
        print("The maximum amount of cores for the current machine is: ", globalsettings.max_cores )
        print("Settings amount of cores to maximum for best performance")
        parallel_cores = globalsettings.max_cores
    if(parallel):
        stacksize = 0
        factor = 1
        for diag in Scenario["used_diags_dict"]:
            if(diag == "ECN" or diag == "ECO"  or diag == "ECI"):
                factor = 3
        if(Config["Physics"]["dstf"] in ["Ge", "GB", "Re", "Lu"]):
            factor *= 3
        stacksize += int(np.ceil(Config["Numerics"]["max_points_svec"] * 3.125) * factor)
        os.environ['OMP_STACKSIZE'] = "{0:d}k".format(stacksize)
    else:
        parallel_cores = 1 # serial
    os.environ['ECRad_WORKING_DIR'] = Config["Execution"]["scratch_dir"]
    os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
    print("Scratch dir set to: " + os.environ['ECRad_WORKING_DIR'])
    launch_options_dict = {}
    launch_options_dict["jobname"] = "-J " + "E{0:5d}".format(Scenario["shot"])
    launch_options_dict["stdout"] = "-o {0:s}".format(os.path.join(Config["Execution"]["scratch_dir"], "ECRad.stdout"))
    launch_options_dict["stderr"] = "-e {0:s}".format(os.path.join(Config["Execution"]["scratch_dir"], "ECRad.stderr"))
    launch_options_dict["partition"] = globalsettings.partition_function(parallel_cores, Config["Execution"]["wall_time"])
    launch_options_dict["qos"] = globalsettings.qos_function(parallel_cores, Config["Execution"]["wall_time"])
    launch_options_dict["account"] = globalsettings.account_fuction()
    launch_options_dict["memory"] = "--mem-per-cpu={0:d}M".format(int(Config["Execution"]["vmem"] / parallel_cores))
    launch_options_dict["cpus"] = "--cpus-per-task={0:d}".format(parallel_cores)
    launch_options_dict["chdir"] = "--chdir=" + globalsettings.ECRadPylibRoot
    InvokeECRad = ["sbatch"]
    for key in launch_options_dict:
        if(len(launch_options_dict[key]) > 0):
            InvokeECRad += [launch_options_dict[key]]
    InvokeECRad += [globalsettings.ECRadPathBSUB]
    return InvokeECRad

