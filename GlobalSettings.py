'''
Created on Jan 29, 2017

@author: sdenk
'''
import os
import sys
import wx

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
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.TCV = False # Not fully supported -> needs some work
        self.root = os.path.expanduser("~/")
        self.SLES12 = False
        if(os.getenv("SYS") == 'amd64_sles12'  or os.getenv("SYS") == 'amd64_sles15'):
            self.SLES12 = True
        self.Phoenix = "phoenix" in wx.PlatformInfo
        self.ECRadRoot = "/gss_efgw_work/work/g2sdenk/ECRad/"
        self.ECRadPylibRoot = "../ECRad_Pylib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadDevPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRaddb")
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.TB_path = "/gss_efgw_work/work/g2sdenk/torbeam/lib-OUT/"
        self.qos_function = qos_function_itm
        self.partition_function = partition_function_itm
        self.max_cores = 36
        self.pylib_folder = "../ECRad_Pylib"
        self.GUI_folder = "../ECRad_GUI"
                
class GlobalSettingsAUG:
    def __init__(self):
        self.AUG = True  # True  -> Start with True, set it to false if we run into problems
        self.TCV = False # Not fully supported -> needs some work
        self.root = os.path.expanduser("~/")
        self.SLES12 = False
        if(os.getenv("SYS") == 'amd64_sles12'  or os.getenv("SYS") == 'amd64_sles15'):
            self.SLES12 = True
        self.Phoenix = "phoenix" in wx.PlatformInfo
        self.ECRadRoot ="/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/"
        self.ECRadPylibRoot = "../ECRad_Pylib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadDevPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRaddb")
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.TB_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/"
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.max_cores = 32
        
class GlobalSettingsAUGEXT:
    def __init__(self):
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.TCV = False # Not fully supported -> needs some work
        self.root = os.path.expanduser("~/")
        self.SLES12 = False
        if(os.getenv("SYS") == 'amd64_sles12'  or os.getenv("SYS") == 'amd64_sles15'):
            self.SLES12 = True
        self.Phoenix = "phoenix" in wx.PlatformInfo
        self.ECRadRoot ="/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/"
        self.ECRadPylibRoot = "../ECRad_Pylib/"
        self.ECRadGUIRoot = "../ECRad_GUI/"
        self.ECRadDevPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRaddb")
        self.ECRadPath = os.path.join(self.ECRadRoot,os.environ['SYS'],"ECRad")
        self.ECRadPathBSUB = os.path.join(self.ECRadRoot,"ECRad_submit.bsub")
        self.TB_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam_repo/TORBEAM/branches/lib-OUT/"
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.max_cores = 32
        self.pylib_folder = "../ECRad_Pylib"
        self.GUI_folder = "../ECRad_GUI"
        
globalsettings = GlobalSettingsITM()
