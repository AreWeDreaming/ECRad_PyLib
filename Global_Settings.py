'''
Created on Jan 29, 2017

@author: sdenk
'''
import os
import wx
import multiprocessing
import sys
from glob import glob

def qos_function_tok(cores, wall_time):
    if(cores == 1):
        if(wall_time <= 4):
            return "--qos s.tok.short"
        elif(wall_time <= 36):
            return "--qos s.tok.standard"
        else:
            return "--qos s.tok.long"
    else:
        if(wall_time <= 2):
            return "--qos p.tok.openmp.2h"
        elif(wall_time <= 4):
            return "--qos p.tok.openmp.4h"
        elif(wall_time <= 24):
            return "--qos p.tok.openmp.24h"
        else:
            return "--qos p.tok.openmp.48h"
    
def qos_function_itm(cores, wall_time):
    if(wall_time <= 48):
        return ""
    else:
        return "--qos skl_qos_fuagwlong"
    
def partition_function_tok(cores, wall_time):
    if(cores == 1):
        #serial
        return "--partition=s.tok"
    else:
        return "--partition=p.tok.openmp"
    
def partition_function_itm(cores, wall_time):
    return "--partition=skl_fua_gw"

class GlobalSettingsITM:
    def __init__(self):
        self.AUG = True  # True  -> Start with True, set it to false if we run into problems
        self.TCV = False # Not fully supported -> needs some work
        self.root = os.path.expanduser("~/")
        self.Phoenix = "phoenix" in wx.PlatformInfo
        self.ECRadRoot = "/afs/eufus.eu/g2itmdev/user/g2sdenk/git/augd_ecrad"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,os.environ['SYS'])
        self.ECRadPylibRoot = "../ECRad_Pylib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.TB_path = "/gss_efgw_work/work/g2sdenk/torbeam/lib-OUT/"
        self.qos_function = qos_function_itm
        self.partition_function = partition_function_itm
        self.max_cores = 48
        self.pylib_folder = "../ECRad_Pylib"
        self.GUI_folder = "../ECRad_GUI"
                
class GlobalSettingsAUG:
    def __init__(self):
        self.AUG = True  # True  -> Start with True, set it to false if we run into problems
        self.TCV = False # Not fully supported -> needs some work
        self.root = os.path.expanduser("~/")
        self.Phoenix = "phoenix" in wx.PlatformInfo
        if(os.path.isdir("/afs/ipp-garching.mpg.de/home/s/sdenk/ECRad_testing/augd_ecrad")):
            self.ECRadRoot = "/afs/ipp-garching.mpg.de/home/s/sdenk/ECRad_testing/augd_ecrad"
        else:
            self.ECRadRoot = "../ECRad_core/"
        self.ECRadLibDir = os.path.join(self.ECRadRoot, self.ECRadRoot,os.environ['SYS'])
#         self.ECRadRoot =      # "/afs/ipp/home/s/sdenk/ECRad_testing/augd_ecrad/"# "/afs/ipp/home/r/rrf/F90/IDA/augd_ecrad/"
        self.ECRadPylibRoot = "../ECRad_Pylib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.TB_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/torbeam/lib-OUT"
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.max_cores = 32
        
class GlobalSettingsEXT:
    def __init__(self):
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.TCV = False # Not fully supported -> needs some work
        self.root = os.path.expanduser("~/")
        self.Phoenix = "phoenix" in wx.PlatformInfo
        self.ECRadRoot = os.path.abspath("../ECRad_core")
        self.ECRadPylibRoot = os.path.abspath("../augd_ecrad_Pylib/")
        self.ECRadGUIRoot = os.path.abspath("../augd_ecrad_GUI/")
        self.ECRadLibDir = os.path.join(self.ECRadRoot, os.environ["SYS"])
        try:
            self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        except KeyError:
            print("WARNING COULD NOT FIND ECRAD")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.TB_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/"
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.max_cores = multiprocessing.cpu_count()
        self.pylib_folder = "../ECRad_Pylib"
        self.GUI_folder = "../ECRad_GUI"
try:        
    if("sles" in os.environ["SYS"]):
        globalsettings = GlobalSettingsAUG()
    elif("rhel" in os.environ["SYS"]):
        globalsettings = GlobalSettingsITM()
    else:
        globalsettings = GlobalSettingsEXT()
except KeyError:
    globalsettings = GlobalSettingsEXT()
library_list = glob("../*pylib") + glob("../*Pylib")
found_lib = False

for folder in library_list:
    if("ECRad" in folder or "ecrad"in folder ):
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