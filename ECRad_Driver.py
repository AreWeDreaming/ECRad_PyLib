'''
Created on Jan 27, 2021

@author: denk
'''
from Global_Settings import globalsettings
from ECRad_F2PY_Interface import ECRadF2PYInterface
from ECRad_Results import ECRadResults
from ECRad_Config import ECRadConfig
from ECRad_Scenario import ECRadScenario
import sys
import os
import re
class ECRadDriver():
    '''
    Driver for ECRad. Passes information to ECRad, runs ECRad and extracts results
    '''


    def __init__(self, Result=None, Scenario_file=None, Config_file=None):
        '''
        Constructor
        '''
        if(Result is None):
            if(Scenario_file is None or Config_file is None):
                raise ValueError("Either result or a Scenario and Config file must be present")
            else:
                self.Result = ECRadResults()
                self.Result.Scenario = ECRadScenario(noLoad=True)
                self.Result.Scenario.load(Scenario_file)
                self.Result.Config = ECRadConfig(noLoad=True)
                self.Result.Config.load(Config_file)
        else:
            self.Result = Result
        self.ECRad_F2PY_interface = ECRadF2PYInterface()
    
    def run(self, id=None):
        itime = 0
        self.Result.set_dimensions()
        while itime < self.Result.Scenario["dimensions"]["N_time"]:
            try:
                self.process_time_point(itime)
                itime += 1
            except Exception as e:
                print("Error when processing t = {0:1.4f}".format(self.Result.Scenario["time"][itime]))
                print("Removing this time point and continuing")
                raise(e)
                self.Result.Scenario.drop_time_point(itime)
                self.Result.set_dimensions()
        self.Result.tidy_up(autosave=False)
        self.Result.to_netcdf(scratch=True, ed=id)
        
    def process_time_point(self, itime):
        self.Result = self.ECRad_F2PY_interface.process_single_timepoint(self.Result, itime)
        
if(__name__=="__main__"):
    try:
        Config_file = sys.argv[1]
        Scenario_file = sys.argv[2]
        ids = re.findall("_(\d{7})", os.path.basename(Scenario_file))
        if(len(ids) == 1):
            id = int(ids[0])
        else:
            id = None
    except:
        Config_file=os.path.join(os.path.expanduser("~"), ".ECRad_GUI_Default.nc") 
        Scenario_file=os.path.join(os.path.expanduser("~"), ".ECRad_GUI_last_scenario.nc")
    driver = ECRadDriver(Scenario_file=Scenario_file, Config_file=Config_file)

#     driver = ECRadDriver(Scenario_file="/mnt/c/Users/Severin/ECRad/ECRad_33585_EXT_ed1.nc", \
#                          Config_file="/mnt/c/Users/Severin/ECRad/ECRad_33585_EXT_ed1.nc")
#     driver = ECRadDriver(Scenario_file="/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc", \
#                          Config_file="/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc")
    # driver = ECRadDriver(Scenario_file="/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed11.nc", \
    #                      Config_file="/mnt/c/Users/Severin/ECRad/Yu/ECRad_179328_EXT_ed11.nc")
    driver.run(id=id)
    
    