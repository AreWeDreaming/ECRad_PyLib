
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline, RectBivariateSpline
from Equilibrium_Utils_AUG import EQData
from Shotfile_Handling_AUG import load_IDA_data
from ece_optics_2019 import get_ECE_launch_v2
from TB_Communication import launch, make_TORBEAM_no_data_load
from ECRad_Scenario import ECRadScenario
import numpy as np
import os
from scipy import constants as cnst

def run_ECE_TORBEAM_ECRad_Scenario(working_dir, ECRad_Scenario_file, time):
    Scenario = ECRadScenario(noLoad=True)
    Scenario.load(ECRad_Scenario_file)
    itime = np.argmin(np.abs(Scenario["time"] - time))
    plasma_dict = Scenario["plasma"]
    eq_slice = Scenario["plasma"]["eq_data_2D"].GetSlice(time)
    ece_launch = Scenario["diagnostic"]
    TB_results = torbeam_interface(working_dir, Scenario["shot"], time, itime, plasma_dict, eq_slice, ece_launch)


def run_ECE_TORBEAM_AUG(working_dir, shot, time, frequencies, launch_override=None, EQ_exp="AUGD", 
                        EQ_diag="EQH", EQ_ed=0, bt_vac_correction = 1.005, IDA_exp="AUGD", IDA_ed=0):

    IDA_dict = load_IDA_data(shot, [time], IDA_exp, IDA_ed)
    itime = 0
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    eq_slice = EQ_obj.GetSlice(time, bt_vac_correction=bt_vac_correction)
    ece_launch = get_ECE_launch_v2(8, "CECE", 0.055, frequencies, np.zeros(len(frequencies)))
    TB_results = torbeam_interface(working_dir, shot, time, itime, IDA_dict, eq_slice, ece_launch)

def torbeam_interface(working_dir, shot, time, itime, plasma_dict, eq_slice, ece_launch):
    cece_launches = []
    for ich in range(len(ece_launch["f"][itime])):
        cece_launches.append(launch())
        cece_launches[-1].parse_custom(ece_launch["f"][itime][ich], ece_launch["R"][itime][ich] * np.cos(np.deg2rad(ece_launch["phi"][itime][ich])), 
                                ece_launch["R"][itime][ich] * np.sin(np.deg2rad(ece_launch["phi"][itime][ich])), ece_launch["z"][itime][ich], 
                                ece_launch["phi_tor"][itime][ich], ece_launch["theta_pol"][itime][ich], ece_launch["width"][itime][ich], ece_launch["dist_focus"][itime][ich])
    make_TORBEAM_no_data_load(working_dir, shot, time, plasma_dict["rhop_prof"][itime], plasma_dict["Te"][itime], plasma_dict["ne"][itime], 
                              eq_slice.R, eq_slice.z, eq_slice.Psi, eq_slice.Br, eq_slice.Bt, eq_slice.Bz,
                              eq_slice.Psi_ax, eq_slice.Psi_sep, cece_launches, ITM=False, 
                              ITER=False, mode = -1)
    for ich in range(len(ece_launch["f"][itime])):
        Rz_data = np.loadtxt(os.path.join(working_dir, "{0:d}_{1:1.3f}_rays".format(shot, time), 
                             "Rz_beam_{0:1d}.dat".format(ich + 1).replace(",", "")))
        B_tot_spl = RectBivariateSpline(eq_slice.R, eq_slice.z, 
            np.sqrt(eq_slice.Br**2 + eq_slice.Bt**2 + eq_slice.Bz**2))
        s = np.zeros(Rz_data.T[0].shape)
        s[1:] = np.sqrt((Rz_data.T[0][1:] - Rz_data.T[0][:-1])**2 + 
                            (Rz_data.T[1][1:] - Rz_data.T[1][:-1])**2)
        s = np.cumsum(s)
        R_center = Rz_data.T[0]
        z_center = Rz_data.T[1]
        root_spl = InterpolatedUnivariateSpline(s, 
                2.0 * B_tot_spl(R_center/1.e2, z_center/1.e2, grid=False) * cnst.e / cnst.m_e
              - cece_launches[0].f * 2.0 * np.pi )
        s_res = np.max(root_spl.roots())
        R_res = []
        z_res = []
        for column in [2,4]:
            R_spl = InterpolatedUnivariateSpline(s, Rz_data.T[column])
            z_spl = InterpolatedUnivariateSpline(s, Rz_data.T[column+1])
            R_res.append(R_spl(s_res))
            z_res.append(z_spl(s_res))
        width = np.sqrt((R_res[1] - R_res[0])**2 + (z_res[1] - z_res[0])**2)
        print(width)



if __name__ == "__main__":
    run_ECE_TORBEAM_ECRad_Scenario("/mnt/c/Users/Severin/ECRad/AUG_CECE/", "/mnt/c/Users/Severin/ECRad/AUG_CECE/ECRad_36974_EXT_ed5.mat", 3.0)