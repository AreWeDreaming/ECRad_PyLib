'''
Created on Dec 16, 2015

@author: Severin Denk
'''
from collections import OrderedDict as od
import numpy as np
from Diag_Types import EXT_diag, TCV_diag

DefaultDiagDict = od()
for diag_name in ["UCE", "LCE", "VCE", "EXT"]:
    if(diag_name in ["UCE", "LCE", "VCE"]):
        DefaultDiagDict.update({diag_name: TCV_diag(diag_name, "TCV", diag_name, 0, 1.0, 1.0)})
    elif(diag_name == "EXT"):
        launch_geo = np.zeros((13, 1))
        launch_geo[0, 0] = 140.e9
        launch_geo[1, 0] = 0.2e9
        launch_geo[2, 0] = 1
        launch_geo[3, 0] = 1
        launch_geo[4, 0] = 1.0
        launch_geo[5, 0] = 0.0
        launch_geo[6, 0] = 2.3
        launch_geo[7, 0] = 104.0
        launch_geo[8, 0] = 0.33
        launch_geo[9, 0] = -0.824
        launch_geo[10, 0] = -8.24
        launch_geo[11, 0] = 1.1851
        launch_geo[12, 0] = 0.0865
        DefaultDiagDict.update({diag_name:  EXT_diag(diag_name, "AUGD", diag_name, 0, launch_geo)})
