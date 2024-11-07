'''
Created on Jan 29, 2017
@author: Severin Denk
'''
import os
import multiprocessing

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
    return ""

class GlobalSettings:
    def __init__(self):
        self.AUG = False  # True  -> Start with True, set it to false if we run into problems
        self.root = os.path.expanduser("~/")
        self.ECRadRoot = os.path.join(os.path.dirname(__file__), os.pardir)
        self.ECRadCoreRoot = os.path.join(self.ECRadRoot, "ECRad_core")
        self.ECRadPylibRoot = os.path.join(self.ECRadRoot, "ECRad_PyLib")
        self.ECRadGUIRoot = os.path.join(self.ECRadRoot, "ECRad_GUI")
        if "SYS" in os.environ:
            self.ECRadLibDir = os.path.join(self.ECRadRoot, os.environ["SYS"])
        else:
            self.ECRadLibDir = os.path.join(self.ECRadRoot, "bin")
        self.ECRadPathBSUB = os.path.join(self.ECRadPylibRoot,"ECRad_Driver_submit.bsub")
        self.TB_path = os.path.abspath("../libtorbeam")
        self.batch_submission_cmd = "bash"
        self.qos_function = qos_function_tok
        self.partition_function = partition_function_tok
        self.account_fuction = account_function_current_user
        self.max_cores = multiprocessing.cpu_count() / 2
        self.plot_mode = "Presentation"
        self.mathrm = r"\mathrm"
        self.omas = True
        try:
            import omas
        except ImportError:
            self.omas = False

globalsettings = GlobalSettings()

