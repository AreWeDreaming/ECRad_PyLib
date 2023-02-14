from matplotlib.pyplot import figure
import numpy as np
from ecrad_pylib.ECRad_Results import ECRadResults
from ecrad_pylib.ece_optics_2019 import plot1DECE
from ecrad_pylib.Plotting_Configuration import plt

def verify_CECE_LOS(ECRad_results_file):
    res = ECRadResults()
    res.load(ECRad_results_file)
    x,y = plot1DECE(project="toroidal" , freq=res.Scenario["diagnostic"]["f"][0][0]/1.e9,doPlot=False, verb=False)
    R,z = plot1DECE(project="poloidal" , freq=res.Scenario["diagnostic"]["f"][0][0]/1.e9,doPlot=False, verb=False)
    fig_top = plt.figure()
    ax_top = fig_top.add_subplot(111)
    fig_side = plt.figure()
    ax_side = fig_side.add_subplot(111)
    label_Pedro = "Pedro"
    for i in [0, 2, 4]:
        ax_top.plot(x, y.T[i], "--", label=label_Pedro)
        ax_side.plot(R, z.T[i], "--", label=label_Pedro)
        label_Pedro =None
    label_ECRad = "ECRad"
    for ir in range(res.Config["Physics"]["N_ray"]):
        phi = np.arctan2(np.array(res["ray"]["y"][0][0][0][ir],dtype=np.float), 
                         np.array(res["ray"]["x"][0][0][0][ir],dtype=np.float))
        phi_rot = 22.5 * 8.5
        x_ECRad = res["ray"]["R"][0][0][0][ir] * np.cos(phi - np.deg2rad(phi_rot))
        y_ECRad = res["ray"]["R"][0][0][0][ir] * np.sin(phi - np.deg2rad(phi_rot))
        ax_top.plot(x_ECRad, y_ECRad, "-", label=label_ECRad)
        ax_side.plot(res["ray"]["R"][0][0][0][ir], res["ray"]["z"][0][0][0][ir], "-", label=label_ECRad)
        label_ECRad = None
    ax_top.set_xlabel(r"$x$ [m]")
    ax_top.set_xlim(1,3)
    ax_top.set_ylim(-0.1,0.2)
    ax_top.set_ylabel(r"$y$ [m]")
    ax_side.set_xlabel(r"$R$ [m]")
    ax_side.set_ylabel(r"$z$ [m]")
    ax_side.set_xlim(1,3)
    ax_side.set_ylim(-0.1,0.2)
    ax_top.legend()
    ax_side.legend()
    plt.show()

if(__name__ == "__main__"):
    verify_CECE_LOS("/mnt/c/Users/Severin/ECRad/AUG_CECE/ECRad_36974_CEC_ed1.nc")