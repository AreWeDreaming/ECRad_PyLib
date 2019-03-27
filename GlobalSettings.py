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
import platform
SLES12 = False
if("SuSE-12" in platform.platform() or "SuSE-15" in platform.platform()):
    SLES12 = True
    # This is necessary to ensure that local packages from e.g. pip are not considered
    # Since they were installed on sles11 they are not compatible with sles12
    i = 0
    while True:
        if(".local" in sys.path[i]):
            sys.path.pop(i)
        i += 1
        if(i >= len(sys.path)):
            break
elif(not "SuSE-11" in platform.platform()):
    itm=True # Make this better?
Phoenix = "phoenix" in wx.PlatformInfo
ECRadDevPath = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/@sys/ECRaddb")
ECRadPath = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/@sys/ECRad")
ECRadPathBSUB = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/ECRad_submit.bsub")

