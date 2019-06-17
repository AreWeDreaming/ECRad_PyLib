'''
Created on Jan 29, 2017

@author: sdenk
'''
import os
import sys
import wx
AUG = True  # True  #
TCV = False  # False  #
root = os.path.expanduser("~/")
itm = False
SLES12 = False
if(os.getenv("SYS") == 'amd64_sles12'  or os.getenv("SYS") == 'amd64_sles15'):
    SLES12 = True
Phoenix = "phoenix" in wx.PlatformInfo
ECRadRoot= os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/")
ECRadDevPath = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/@sys/ECRaddb")
ECRadPath = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/@sys/ECRad")
ECRadPathBSUB = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/ECRad_submit.bsub")
os.environ["SYS"] = 'amd64_sles11'

