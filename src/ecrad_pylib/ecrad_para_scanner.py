from ecrad_pylib.ECRad_F2PY_Interface import ECRadF2PYInterface
from ecrad_pylib.ECRad_Scenario import ECRadScenario
from ecrad_pylib.ECRad_Config import ECRadConfig
from ecrad_pylib.ECRad_Results import ECRadResults
import numpy as np

def scan_Te_scale(scenario_path, scaling_range, result_destination, source_time_index=0):
    scenario = ECRadScenario(noLoad=True)
    scenario.load(scenario_path)
    times = np.linspace(0, 1, len(scaling_range))
    Te = np.copy(scenario["plasma"]["Te"][source_time_index])
    scenario.duplicate_time_point(source_time_index, times)
    config = ECRadConfig(noLoad=True)
    config.load(scenario_path)
    ECRad_interface = ECRadF2PYInterface()
    results = ECRadResults()
    config["Execution"]["extra_output"] = False
    config["Execution"]["parallel_cores"] = 16
    config["Physics"]["N_max"] = 5
    results.Config = config
    results.Scenario = scenario
    results.set_dimensions()
    for time_index, scale in enumerate(scaling_range):
        scenario["plasma"]["Te"][time_index] = Te * scale
        ECRad_interface.process_single_timepoint(results, time_index)
    results.tidy_up()
    results.to_netcdf(result_destination)

if __name__ == "__main__":
    scan_Te_scale("/mnt/c/Users/denk/Old/ECRad/ECRad_runs/ECRad_104103_EXT_ed4.nc", np.linspace(0.2, 1.5, 50), "/home/denks/ECRad/ECRad_104103_Te_scan.nc")
    

