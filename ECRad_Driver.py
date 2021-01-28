'''
Created on Jan 27, 2021

@author: denk
'''
from ECRad_F2PY_Interface import ECRadF2PYInterface
from ECRad_Results import ECRadResults
from ECRad_Config import ECRadConfig
from ECRad_Scenario import ECRadScenario
import numpy as np

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
        self.ECRad_F2PY_interface = ECRadF2PYInterface(self.Result.Config, self.Result.Scenario)
    
    def run(self):
        itime = 0
        while itime < self.Result.Scenario["dimensions"]["N_time"]:
            try:
                self.process_time_point(itime)
                itime += 1
            except Exception as e:
                print("Error when processing t = {0:1.4f}".format(self.Result.Scenario["time"][itime]))
                print("Removing this time point and continuing")
                raise(e)
                self.Result.Scenario.drop_time_point(itime)
        self.Result.tidy_up(True)
        self.Result["git"] = np.genfromtxt()
        
    def process_time_point(self, itime):
        self.ECRad_F2PY_interface.reset()
        self.ECRad_F2PY_interface.set_config_and_diag(self.Result.Config, self.Result.Scenario, itime)
        self.ECRad_F2PY_interface.set_equilibrium(self.Result.Scenario, itime)
        self.ECRad_F2PY_interface.make_rays(self.Result.Scenario, itime)
        self.ECRad_F2PY_interface.run_and_get_output(self.Result, itime)
        
if(__name__=="__main__"):
#     Scenario_file = "Scenario.nc"
#     Config_file = "Config.nc"
#     driver = ECRadDriver(Scenario_file=Scenario_file, Config_file=Config_file)
    driver = ECRadDriver(Scenario_file="/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc", \
                         Config_file="/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc")
    driver.run()
    
    