# Collections of plots that are too specific to be part of the GUI
import numpy as np
from ECRad_Results import ECRadResults
from Plotting_Configuration import plt
from Plotting_Core import PlottingCore
from scipy.interpolate import InterpolatedUnivariateSpline


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

def plot_mixed_n2_n3_spectrum(res_file, i_time, i_mode):
    Results = ECRadResults()
    Results.load(res_file)
    time = Results.Scenario["time"][i_time]
    if Results.Config["Physics"]["considered_modes"] == 3:
        if i_mode == 0:
            raise ValueError("This routine only works for pure X/O mode")
        else:
            i_mode_ray = i_mode - 1
    else:
        i_mode_ray = i_mode    
    if(i_mode == 0):
        raise ValueError("This routine only handles ")
    res = np.zeros(Results["Trad"]["Trad"][i_time][i_mode].shape)
    X_3 = np.zeros(Results["Trad"]["Trad"][i_time][i_mode].shape, dtype=bool)
    if Results.Config["Physics"]["N_ray"] != 1 :
        raise ValueError("This routine can only handle single ray runs at present")
    if Results.Config["Physics"]["N_freq"] != 1 :
        raise ValueError("This routine can only handle single ray runs at present")
    i_ray = 0
    i_freq = 0
    R_axis = Results.Scenario["plasma"]["eq_data_2D"].GetSlice(time).R_ax
    for i_ch, ch in enumerate(Results["ray"]["s"][i_time]):
        Y = Results["ray"]["Y"][i_time][i_ch][i_mode_ray][i_ray]
        mask = Y > 0.0
        Y = Y[mask]
        try:
            if np.any(Y < 0.34):
                Y_root_spl = InterpolatedUnivariateSpline(Results["ray"]["s"][i_time][i_ch][i_mode_ray][i_ray][mask], Y - 1.0/3.0)
                s_res = np.min(Y_root_spl.roots())
                rhop = Results["ray"]["rhop"][i_time][i_ch][i_mode_ray][i_ray]
                rhop_mask = np.logical_and(rhop < 1.0, rhop > -1.0)
                rhop_spl = InterpolatedUnivariateSpline(Results["ray"]["s"][i_time][i_ch][i_mode_ray][i_ray][rhop_mask],
                        Results["ray"]["rhop"][i_time][i_ch][i_mode_ray][i_ray][rhop_mask])
                R_spl = InterpolatedUnivariateSpline(Results["ray"]["s"][i_time][i_ch][i_mode_ray][i_ray][rhop_mask],
                        Results["ray"]["R"][i_time][i_ch][i_mode_ray][i_ray][rhop_mask])
                if R_spl(s_res) < R_axis :
                    res[i_ch] = rhop_spl(s_res) * -1.0
                else:
                    res[i_ch] = rhop_spl(s_res)
                X_3[i_ch] = True
            else:
                raise ValueError("No third harmonic resonance")
        except ValueError:
            if Results["resonance"]["R_cold"][i_time][i_mode][i_ch] < R_axis :
                res[i_ch] = Results["resonance"]["rhop_cold"][i_time][i_mode][i_ch] * -1.0
            else:
                res[i_ch] = Results["resonance"]["rhop_cold"][i_time][i_mode][i_ch]
            X_3[i_ch] = False
    fig = plt.figure(figsize = (12, 8.5))
    ax = fig.add_subplot(111)
    ax_ne = ax.twinx()
    rhop = Results.Scenario["plasma"]["rhop_prof"][i_time]
    Te_spl = InterpolatedUnivariateSpline(rhop,
                                          np.log(Results.Scenario["plasma"]["Te"][i_time]))
    ne_spl = InterpolatedUnivariateSpline(rhop,
                                          np.log(Results.Scenario["plasma"]["ne"][i_time]))
    rhop_signed = np.linspace(-np.max(rhop), np.max(rhop), 400)
    ax.plot(rhop_signed, np.exp(Te_spl(np.abs(rhop_signed))) /1.e3, "-", label = r"Actual profile")
    ax.plot(res, np.exp(Te_spl(np.abs(res)))/1.e3, "^", label = r"Expectation from classical ECE")
    # ax.plot(res[np.logical_not(X_3)], Results["Trad"]["Trad"][i_time][i_mode][np.logical_not(X_3)]/1.e3, "o", label=r"2nd harmonic ECE")
    # ax.plot(res[X_3], Results["Trad"]["Trad"][i_time][i_mode][X_3]/1.e3, "+", label=r"3rd harmonic ECE")
    ax_ne.plot(rhop_signed, np.exp(ne_spl(np.abs(rhop_signed)))/1.e19, "--", label = r"$n_\mathrm{e}$ profile", color=(0.5,0.0,0.5))
    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax.set_ylabel(r"$T_\mathrm{e,\,rad}$ [keV]")
    ax_ne.set_ylabel(r"$n_\mathrm{e}$ [$\times 10^{19}$ m$^-3$]")
    lns = ax.get_lines() + ax_ne.get_lines()
    labs = [l.get_label() for l in lns]
    leg = ax.legend(lns, labs)
    plt.show()


if __name__ == "__main__":
    # verify_CECE_LOS("/mnt/c/Users/Severin/ECRad/AUG_CECE/ECRad_36974_CEC_ed1.nc")
    plot_mixed_n2_n3_spectrum("/mnt/c/Users/Severin/ECRad/ITER/ECRad_104103_EXT_ed9.nc", 0, 1)