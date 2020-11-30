'''
Created on Oct 21, 2020

@author: denk
'''

import numpy as np
from ECRad_Results import ECRadResults
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

class ECRadRayManipulator(object):
    '''
    classdocs
    '''


    def __init__(self, ECRad_result_file):
        '''
        Constructor
        '''
        self.org_results_file = ECRad_result_file
        self.ECRad_results = ECRadResults()
        self.ECRad_results.from_mat_file(ECRad_result_file)
        self.N_ch = self.ECRad_results.Scenario.ray_launch[0]["f"]
        self.N_ray = self.ECRad_results.Config.N_ray
        if(self.ECRad_results.Config.considered_modes == 1):
            self.modes = ["X"]
        elif(self.ECRad_results.Config.considered_modes == 2):
            self.modes = ["O"]
        elif(self.ECRad_results.Config.considered_modes == 3):
            self.modes = ["X", "O"]
        else:
            raise ValueError("Invalid value in considered modes")
        
    def get_Rz_single_ray(self, it, ich, mode="X", iray=0):
        if(self.N_ray == 1):
            R = np.sqrt(self.ECRad_results.ray["x"+mode][it][ich]**2 + \
                        self.ECRad_results.ray["y"+mode][it][ich]**2)
            return R, self.ECRad_results.ray["z"+mode][it][ich]
        else:
            R = np.sqrt(self.ECRad_results.ray["x"+mode][it][ich][iray]**2 + \
                        self.ECRad_results.ray["y"+mode][it][ich][iray]**2)
            return R, self.ECRad_results.ray["z"+mode][it][ich][iray]
    
    def get_field_single_ray(self, field, it, ich, mode="X", iray=0):
        if(self.N_ray == 1):
            return self.ECRad_results.ray[field+mode][it][ich]
        else:
            return self.ECRad_results.ray[field+mode][it][ich][iray]
       
    def set_field_single_ray(self, field, values, it, ich, mode="X", iray=0):
        if(self.N_ray == 1):
            if(self.ECRad_results.ray[field+mode][it][ich].shape != values.shape):
                raise ValueError("The new values must have the same shape as the original values")
            self.ECRad_results.ray[field+mode][it][ich] = values
        else:
            if(self.ECRad_results.ray[field+mode][it][ich][iray].shape != values.shape):
                raise ValueError("The new values must have the same shape as the original values")
            self.ECRad_results.ray[field+mode][it][ich][iray] = values
            
    def save(self, new_filename, comment=""):
        if(new_filename == self.org_results_file):
            print("This routine does not allow overwriting of the old file!")
            return
        self.ECRad_results.to_mat_file(new_filename, comment)
            
if(__name__=="__main__"):
    from Plotting_Configuration import plt
    ECRad_manip = ECRadRayManipulator("/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed10.mat")
    R, z = ECRad_manip.get_Rz_single_ray(0, 0)
    Te_spl = RectBivariateSpline(ECRad_manip.ECRad_results.Scenario.plasma_dict["eq_data"][0].R, \
                                 ECRad_manip.ECRad_results.Scenario.plasma_dict["eq_data"][0].z, \
                                 ECRad_manip.ECRad_results.Scenario.plasma_dict["Te"][0])
    rhop = ECRad_manip.get_field_single_ray("rhop", 0, 0)
    plt.plot(rhop[rhop >=0], \
             ECRad_manip.get_field_single_ray("Te", 0, 0)[rhop >= 0], "+",  label="before")
    ECRad_manip.set_field_single_ray("Te", Te_spl(R,z, grid=False) * 1.5, 0, 0)
    ECRad_manip.save("/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed10_1.5Te.mat", \
                     "increased Te by a factpr pf1.5")
    ECRad_manip = ECRadRayManipulator("/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed10_1.5Te.mat")
    plt.plot(rhop[rhop >=0], \
             ECRad_manip.get_field_single_ray("Te", 0, 0)[rhop >= 0], "^", label="after")
    plt.legend()
    plt.show()
    
    
    
    
    
            