
from multiprocessing import Value
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline, RectBivariateSpline
from Equilibrium_Utils_AUG import EQData
from Shotfile_Handling_AUG import load_IDA_data
from ece_optics_2019 import get_ECE_launch_v2
from TB_Communication import launch, make_TORBEAM_no_data_load
from ECRad_Scenario import ECRadScenario
from ECRad_Results import ECRadResults
import numpy as np
import os
from scipy import constants as cnst
from Basic_Methods.Data_Fitting import make_fit, gauss_fit_func
from Plotting_Configuration import plt
from ece_optics_2019 import plot1DECE

def CECE_workflow(working_dir, Results_filename, time):
    Results = ECRadResults()
    Results.load(Results_filename)
    logfile = open(os.path.join(working_dir, f"CECE_params_{Results.Scenario['shot']}"), "w")
    run_ECE_TORBEAM_ECRad_Scenario(working_dir, Results, time, logfile=logfile)
    get_BPD_width(Results, 0, logfile=logfile)
    logfile.close()
    plt.show()

def get_BPD_width(Results, itime=0, logfile=None):
    # Gaussian fit of all BPD
    plt.figure()
    for ich in range(len(Results.Scenario["diagnostic"]["f"][itime])):
        params = np.zeros(3)
        params[0] = np.max(Results["BPD"]["BPD"][itime][ich][0])
        params[1] = Results["resonance"]["rhop_warm"][0][itime][ich]
        params[2] = 0.05
        fitted_params, unc = make_fit("gauss", Results["BPD"]["rhop"][itime][ich][0], Results["BPD"]["BPD"][itime][ich][0], 
                p_est = params)
        plt.plot(Results["BPD"]["rhop"][itime][ich][0], Results["BPD"]["BPD"][itime][ich][0], label=f"original ch. {ich+1}")
        plt.plot(Results["BPD"]["rhop"][itime][ich][0], 
                gauss_fit_func(Results["BPD"]["rhop"][itime][ich][0], fitted_params[0], 
                               fitted_params[1], fitted_params[2]), linestyle="--", label=f"fit ch. {ich+1}")
        if(logfile is not None):
            logfile.write("Channel {0:d} rho_pol position: {1:1.3f}\n".format(ich + 1, fitted_params[1]))
            logfile.write("Channel {0:d} rho_pol width: {1:1.3f}\n".format(ich + 1, fitted_params[2]))
        else:
            print("Channel {0:d} rho_pol width: {1:1.3f}".format(ich + 1, fitted_params[2]))
        mask = Results["ray"]["rhop"][itime][ich][0][0] > 0.0
        R_center_spl = InterpolatedUnivariateSpline(Results["ray"]["s"][itime][ich][0][0][mask], 
                                                    Results["ray"]["rhop"][itime][ich][0][0][mask]
                                                    - fitted_params[1])
        R_in_spl = InterpolatedUnivariateSpline(Results["ray"]["s"][itime][ich][0][0][mask], 
                                                Results["ray"]["rhop"][itime][ich][0][0][mask]
                                                 - fitted_params[1] + fitted_params[2]/2.0)
        R_out_spl = InterpolatedUnivariateSpline(Results["ray"]["s"][itime][ich][0][0][mask], 
                                                Results["ray"]["rhop"][itime][ich][0][0][mask]
                                                 - fitted_params[1] - fitted_params[2]/2.0)
        R_spl =  InterpolatedUnivariateSpline(Results["ray"]["s"][itime][ich][0][0], 
                                              Results["ray"]["R"][itime][ich][0][0]) 
        s_center_roots = R_center_spl.roots()
        s_center = s_center_roots[np.argmin(np.abs(s_center_roots - Results["resonance"]["s_cold"][0][itime][ich]))]                        
        s_in_roots = R_in_spl.roots()
        s_in = s_in_roots[np.argmin(np.abs(s_in_roots - Results["resonance"]["s_cold"][0][itime][ich]))]
        s_out_roots = R_out_spl.roots()
        s_out = s_out_roots[np.argmin(np.abs(s_out_roots - Results["resonance"]["s_cold"][0][itime][ich]))]
        width = np.abs(R_spl(s_in)-R_spl(s_out))
        if(logfile is not None):
            logfile.write("Channel {0:d} R position: {1:1.3f} cm\n".format(ich + 1, R_spl(s_center)))
            logfile.write("Channel {0:d} R width: {1:1.3f} cm\n".format(ich + 1, width*1.e2))
        else:
            print("Channel {0:d} R width: {1:1.3f} cm\n".format(ich + 1, width*1.e2))
    plt.legend()
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$")
    plt.gca().set_ylabel(r"$D_\omega$")
    plt.suptitle(f"\# {Results.Scenario['shot']}")


def run_ECE_TORBEAM_ECRad_Scenario(working_dir, Results, time, logfile=None, imode=0):
    itime = np.argmin(np.abs(Results.Scenario["time"] - time))
    plasma_dict = Results.Scenario["plasma"]
    eq_slice = Results.Scenario["plasma"]["eq_data_2D"].GetSlice(time)
    diag = Results.Scenario["used_diags_dict"]["CEC"]
    f = Results.Scenario["diagnostic"]["f"][itime]
    df = Results.Scenario["diagnostic"]["df"][itime]
    ece_launch = get_ECE_launch_v2(wgIn=diag.wg, antenna="CECE",dtoECESI=diag.dtoECESI, freqsSI=f, dfreqsSI=df, R_start=2.5)
    TB_results = torbeam_interface(working_dir, Results.Scenario["shot"], time, itime, 
            plasma_dict, eq_slice, ece_launch, 
            R_res = Results["resonance"]["R_warm"][itime][imode], 
            z_res = Results["resonance"]["z_warm"][itime][imode], 
            logfile=logfile, wg=diag.wg, dtoECESI=diag.dtoECESI)


def run_ECE_TORBEAM_AUG(working_dir, shot, time, frequencies, launch_override=None, EQ_exp="AUGD", 
                        EQ_diag="EQH", EQ_ed=0, bt_vac_correction = 1.005, IDA_exp="AUGD", IDA_ed=0):
    # NOT TESTED!
    IDA_dict = load_IDA_data(shot, [time], IDA_exp, IDA_ed)
    itime = 0
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    eq_slice = EQ_obj.GetSlice(time, bt_vac_correction=bt_vac_correction)
    ece_launch = get_ECE_launch_v2(8, "CECE", 0.055, frequencies, np.zeros(len(frequencies)))
    TB_results = torbeam_interface(working_dir, shot, time, itime, IDA_dict, eq_slice, ece_launch)

def torbeam_interface(working_dir, shot, time, itime, plasma_dict, eq_slice, ece_launch, R_res, z_res,
                      wg=8, dtoECESI=0.055, logfile=None, TB_plot=True):
    cece_launches = []
    for ich in range(len(ece_launch["f"])):
        cece_launches.append(launch())
        cece_launches[-1].parse_custom(ece_launch["f"][ich], ece_launch["R"][ich] * np.cos(np.deg2rad(ece_launch["phi"][ich])), 
                                ece_launch["R"][ich] * np.sin(np.deg2rad(ece_launch["phi"][ich])), ece_launch["z"][ich], 
                                ece_launch["phi_tor"][ich], ece_launch["theta_pol"][ich], ece_launch["width"][ich], ece_launch["dist_focus"][ich])
    make_TORBEAM_no_data_load(working_dir, shot, time, plasma_dict["rhop_prof"][itime], plasma_dict["Te"][itime], plasma_dict["ne"][itime], 
                              eq_slice.R, eq_slice.z, eq_slice.Psi, eq_slice.Br, eq_slice.Bt, eq_slice.Bz,
                              eq_slice.Psi_ax, eq_slice.Psi_sep, cece_launches, ITM=False, 
                              ITER=False, mode = -1)
    for ich in range(len(ece_launch["f"])):
        Rz_data = np.loadtxt(os.path.join(working_dir, "{0:d}_{1:1.3f}_rays".format(shot, time), 
                             "Rz_beam_{0:1d}.dat".format(ich + 1).replace(",", "")))
        R_center = Rz_data.T[0] * 1.e-2
        mask = np.logical_and(R_center > np.min(eq_slice.R), R_center < np.max(eq_slice.R))
        z_center = Rz_data.T[1] * 1.e-2
        mask[np.logical_and(z_center < np.min(eq_slice.z), z_center > np.max(eq_slice.z))] = False
        try:
            if(not np.any(mask)):
                raise ValueError("No points inside flux Matrix!")
            if(TB_plot):
                plt.figure()
                plt.plot(R_center, z_center)
                plt.plot(Rz_data.T[2]*1.e-2, Rz_data.T[3]*1.e-2, "--")
                plt.plot(Rz_data.T[4]*1.e-2, Rz_data.T[5]*1.e-2, "--")
                x, z = plot1DECE(wgIn=wg, freq=ece_launch["f"][ich]/1.e9, dtoECE=dtoECESI*1.e3, project='poloidal', doPlot=False)
                plt.plot(R_center, z_center)
                plt.plot(x, z.T[0], ":")
                plt.plot(x, z.T[2], "-.")
                plt.plot(x, z.T[4], ":")
                plt.gca().set_xlabel(r"$R$ [m]")
                plt.gca().set_ylabel(r"$z$ [m]")
            i_min = np.argmin(np.abs((R_center[mask]-R_res[ich])**2 + (z_center[mask]-z_res[ich])**2))
            width = np.sqrt((Rz_data.T[2][i_min]*1.e-2 - R_center[i_min])**2 
                            + (Rz_data.T[3][i_min]*1.e-2 - z_center[i_min])**2)
            if(logfile is not None):
                logfile.write("TORBEAM width channel {0:d}: {1:1.3f} cm\n".format(ich+1, width*1.e2))
            else:
                print("TORBEAM width channel {0:d}: {1:1.3f} cm\n".format(ich+1, width*1.e2))
            return
        except ValueError:
            pass
        print("Failed to run TORBEAM. Using vacuum values!")
        x, z = plot1DECE(wgIn=wg, freq=ece_launch["f"][ich]/1.e9, dtoECE=dtoECESI*1.e3, project='poloidal', doPlot=False)
        waist = np.zeros(len(x))
        waist = np.sqrt( ((z[:,2]-z[:,0])**2)+((z[:,2]-z[:, 1])**2) ) 
        waist += np.sqrt( ((z[:,2]-z[:,3])**2)+((z[:,2]-z[:, 4])**2) )
        waist /= 2
        R_center = x
        mask = np.logical_and(R_center > np.min(eq_slice.R), R_center < np.max(eq_slice.R))
        z_center = z.T[2]
        mask[np.logical_or(z_center < np.min(eq_slice.z), z_center > np.max(eq_slice.z))] = False
        if(not np.any(mask)):
            raise ValueError("No resonance found!")
        i_min = np.argmin(np.abs((R_center-R_res[ich])**2 + (z_center-z_res[ich])**2))
        if(logfile is not None):
            logfile.write("Vacuum width channel {0:d}: {1:1.3f} cm\n".format(ich + 1, waist[i_min] * 100.0))
        else:
            print("Vacuum width channel {0:d}: {1:1.3f} cm\n".format(ich + 1, waist[i_min]  * 100.0))



if __name__ == "__main__":
    CECE_workflow("/mnt/c/Users/Severin/ECRad/AUG_CECE/", 
                  "/mnt/c/Users/Severin/ECRad/AUG_CECE/ECRad_38420_CEC_ed1.nc", 3.0)