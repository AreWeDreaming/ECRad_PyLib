'''
Created on Jan 29, 2017

@author: sdenk
'''
import os
import multiprocessing
import sys
from glob import glob
import socket
import getpass

def qos_function_tok(cores, wall_time):
    if(cores == 1):
        if(wall_time <= 4):
            return "--qos=s.tok.short"
        elif(wall_time <= 36):
            return "--qos=s.tok.standard"
        else:
            return "--qos=s.tok.long"
    else:
        if(wall_time <= 2):
            return "--qos=p.tok.openmp.2h"
        elif(wall_time <= 4):
            return "--qos=p.tok.openmp.4h"
        elif(wall_time <= 24):
            return "--qos=p.tok.openmp.24h"
        else:
            return "--qos=p.tok.openmp.48h"
    
def qos_function_itm(cores, wall_time):
    if(wall_time <= 48):
        return ""
    else:
        return "--qos=skl_qos_fuagwlong"
    
def qos_function_iris(cores, wall_time):
    return ""

def partition_function_tok(cores, wall_time):
    if(cores == 1):
        #serial
        return "--partition=s.tok"
    else:
        return "--partition=p.tok.openmp"
    
def partition_function_itm(cores, wall_time):
    return "--partition=skl_fua_gw"

def partition_function_iris(cores, wall_time):
    if(wall_time <= 0.5):
        return "--partition=short"
    elif(wall_time > 0.5 and wall_time <= 24):
        return "--partition=medium"
    else:
        return "--partition=long"

def partition_function_engaging(cores, wall_time):
    return "--partition=sched_mit_psfc"

def qos_function_engaging(cores, wall_time):
    return "--qos=psfc_24h"

def account_function_current_user():
    return "--account=" + getpass.getuser()

class GlobalSettingsITM:
    def __init__(self):
        self.AUG = True  # True  -> Start with True, set it to false if we run into problems
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = "/afs/eufus.eu/g2itmdev/user/g2sdenk/git/augd_ecrad"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,os.environ['SYS'])
        self.ECRadPylibRoot = "../ECRad_PyLib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.qos_function = qos_function_itm
        self.partition_function = partition_function_itm
        self.account_fuction = account_function_current_user
        self.max_cores = 48
        self.pylib_folder = "../ECRad_Pylib"
        self.GUI_folder = "../ECRad_GUI"
        self.plot_mode = "Presentation"
                
class GlobalSettingsAUG:
    def __init__(self):
        self.AUG = True  # True  -> Start with True, set it to false if we run into problems
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = "../ECRad_core/"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,os.environ['SYS'])
#         self.ECRadRoot =      # "/afs/ipp/home/s/sdenk/ECRad_testing/augd_ecrad/"# "/afs/ipp/home/r/rrf/F90/IDA/augd_ecrad/"
        self.ECRadPylibRoot = "../ECRad_PyLib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadPylibRoot,"ECRad_Driver_submit.bsub")
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.account_fuction = account_function_current_user
        self.max_cores = 32
        self.plot_mode = "Presentation"

class GlobalSettingsMIT:
    def __init__(self):
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = "../ECRad_core/"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,"bin")
#         self.ECRadRoot =      # "/afs/ipp/home/s/sdenk/ECRad_testing/augd_ecrad/"# "/afs/ipp/home/r/rrf/F90/IDA/augd_ecrad/"
        self.ECRadPylibRoot = "../ECRad_PyLib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,"bin","ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadPylibRoot,"ECRad_Driver_submit.bsub")
        self.qos_function = qos_function_engaging
        self.partition_function = partition_function_engaging
        self.account_fuction = account_function_current_user
        self.max_cores = 32
        self.plot_mode = "Presentation"

class GlobalSettingsITER:
    def __init__(self):
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = "../ECRad_core/"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,"bin")
#         self.ECRadRoot =      # "/afs/ipp/home/s/sdenk/ECRad_testing/augd_ecrad/"# "/afs/ipp/home/r/rrf/F90/IDA/augd_ecrad/"
        self.ECRadPylibRoot = "../ECRad_PyLib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,"bin","ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadPylibRoot,"ECRad_Driver_submit.bsub")
        self.qos_function = qos_function_engaging
        self.partition_function = partition_function_engaging
        self.account_fuction = account_function_current_user
        self.max_cores = 16
        self.plot_mode = "Presentation"
        
class GlobalSettingsIRIS:
    def __init__(self):
        self.AUG = False
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = "../ECRad_core/"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,os.environ['SYS'])
#         self.ECRadRoot =      # "/afs/ipp/home/s/sdenk/ECRad_testing/augd_ecrad/"# "/afs/ipp/home/r/rrf/F90/IDA/augd_ecrad/"
        self.ECRadPylibRoot = "../ECRad_PyLib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadPylibRoot,"ECRad_Driver_submit.bsub")
        self.qos_function = qos_function_iris
        self.partition_function = partition_function_iris
        self.account_fuction = account_function_current_user
        self.max_cores = 16
        self.plot_mode = "Presentation"
        
        
class GlobalSettingsEXT:
    def __init__(self):
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = os.path.abspath("../ECRad_core")
        self.ECRadPylibRoot = os.path.abspath("../ECRad_PyLib/")
        self.ECRadGUIRoot = os.path.abspath("../ECRad_GUI/")
        if "SYS" in os.environ:
            self.ECRadLibDir = os.path.join(self.ECRadRoot, os.environ["SYS"])
        else:
            self.ECRadLibDir = os.path.join(self.ECRadRoot, "bin")
        self.ECRadPathBSUB = os.path.join(self.ECRadPylibRoot,"ECRad_Driver_submit.bsub")
        self.TB_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/"
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.max_cores = multiprocessing.cpu_count()
        self.pylib_folder = "../ECRad_PyLib"
        self.GUI_folder = "../ECRad_GUI"
        self.plot_mode = "Software"
try:        
    if("mpg.de" in socket.getfqdn()):
        globalsettings = GlobalSettingsAUG()
    elif("eufus" in socket.getfqdn()):
        globalsettings = GlobalSettingsITM()
    elif("iris" in socket.getfqdn()):
        globalsettings = GlobalSettingsIRIS()
    elif("cm.cluster" in socket.getfqdn()):
        globalsettings = GlobalSettingsMIT()
    elif("iter" in socket.getfqdn()):
        globalsettings = GlobalSettingsITER()
    else:
        globalsettings = GlobalSettingsEXT()
except KeyError:
    globalsettings = GlobalSettingsEXT()
library_list = glob("../*pylib") + glob("../*PyLib")
found_lib = False

for folder in library_list:
    if("ECRad" in folder or "ecrad" in folder ):
        sys.path.append(os.path.abspath(folder))
        found_lib = True
        ECRadPylibFolder = folder
        break
if(not found_lib):
    print("Could not find pylib")
    print("Important: ECRad_GUI must be launched with its home directory as the current working directory")
    print("Additionally, the ECRad_Pylib must be in the parent directory of the GUI and must contain one of ECRad, ecrad and Pylib or pylib")
    exit(-1)
#
#globalsettings = GlobalSettingsITM()
