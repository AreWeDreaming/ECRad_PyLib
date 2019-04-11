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
    print("SLES12  or  SLES15 detected -> Removing .local from path")
    # This is necessary to ensure that local packages from e.g. pip are not considered
    # Since they were installed on sles11 they are not compatible with sles12
    i = 0
    while True:
        if(".local" in sys.path[i]):
            sys.path.pop(i)
        else:
            i += 1
        if(i >= len(sys.path)):
            break
elif(os.getenv("SYS") == 'amd64_sles11'):
    itm = True  # Make this better?
Phoenix = "phoenix" in wx.PlatformInfo
ECRadDevPath = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/@sys/ECRaddb")
ECRadPath = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/@sys/ECRad")
ECRadPathBSUB = os.path.join("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad/ECRad_submit.bsub")
os.environ["SYS"] = 'amd64_sles11'

