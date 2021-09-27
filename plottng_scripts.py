# Collections of plots that are too specific to be part of the GUI
import numpy as np
from ECRad_Results import ECRadResults
from Plotting_Configuration import plt
from Plotting_Core import PlottingCore

def plot_harmonics_and_frequencies(res_file):
    Results = ECRadResults()
    Results.load(res_file)
    fig = plt.figure()
    pc_obj = PlottingCore(fig=fig)
    ich_list = np.arange(0, len(Results.Scenario["diagnostic"]["f"][0]), 3, dtype = np.int)
    imode_list = np.zeros(ich_list.shape, dtype = np.int)
    ir_list = np.zeros(ich_list.shape, dtype = np.int)
    pc_obj.B_plot(Results, 0,ich_list, imode_list, ir_list)
    plt.show()


if __name__ == "__main__":
    plot_harmonics_and_frequencies("/mnt/c/Users/Severin/ECRad/ITER/ECRad_104103_EXT_ed37.nc")