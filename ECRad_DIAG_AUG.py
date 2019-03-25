'''
Created on Dec 16, 2015

@author: sdenk
'''
from collections import OrderedDict as od
import numpy as np
from GlobalSettings import *
from Diags import Diag, ECI_diag, ECRH_diag, EXT_diag, TCV_diag
import getpass

DefaultDiagDict = od()
for diag_name in ["ECE", "CTC", "CTA", "IEC", "ECN", "ECO", "EXT"]:
    if(diag_name == "ECE"):
        DefaultDiagDict.update({diag_name: Diag(diag_name, "AUGD", "RMD", 0)})
    elif(diag_name == "ECN" or diag_name == "ECO"):
        Rz_diag = "RZO"
        if("N" in diag_name):
            Rz_diag = "RZN"
        DefaultDiagDict.update({diag_name: ECI_diag(diag_name, "AUGD", "TDI", 0, \
                                      "ECEI", Rz_diag, 0)})
    elif("CT" in diag_name or "IEC" in diag_name):
        beam_line = "8"
        if("A" in diag_name):
            beam_line = "7"
        if(getpass.getuser() == "skni"):
            DefaultDiagDict.update({diag_name:  ECRH_diag(diag_name, "AUGD", diag_name, 0, beam_line, 0.0, False)})
        else:
            DefaultDiagDict.update({diag_name:  ECRH_diag(diag_name, "AUGD", diag_name, 0, beam_line, 1.0, False)})
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
        launch_geo[11, 0] = 1.1850
        launch_geo[12, 0] = 0.0865
        DefaultDiagDict.update({diag_name:  EXT_diag(diag_name, "AUGD", diag_name, 0, launch_geo)})
