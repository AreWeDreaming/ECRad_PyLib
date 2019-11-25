'''
Created on Oct 28, 2019

@author: sdenk
'''
from shotfile_handling_AUG import load_IDA_data, get_total_current
from equilibrium_utils_AUG import EQData
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import datetime
from plotting_configuration import *
from ECRad_Results import ECRadResults
#import mconf

def make_Travis_input_from_AUG(working_dir, shot, time, IDA_exp, IDA_ed, EQ_exp, EQ_diag, EQ_ed):
    N = 200
    EQObj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed, bt_vac_correction=1.005)
    try:
        IDA_dict = load_IDA_data(shot, [time], IDA_exp, IDA_ed)
        prof_file = open(os.path.join(working_dir, "{0:d}_{1:d}_plasma_profiles.txt".format(shot, int(time*1.e3))), 'w')
        prof_file.write("CC  Dummy line\n")
        prof_file.write("CC rho_tor     Ne[1/m^3]     Te[keV]     Zeff\n")
        prof_file.write("Number_of_points {0: 5d}\n".format(N))
        rhop = np.linspace(0.0, 0.99999, N)
        rhot = EQObj.rhop_to_rhot(time, rhop)
        Te_spl = InterpolatedUnivariateSpline(IDA_dict["rhop_prof"][0], IDA_dict["Te"][0], k=1)
        ne_spl = InterpolatedUnivariateSpline(IDA_dict["rhop_prof"][0], IDA_dict["ne"][0], k=1)
        for i in range(N):
            prof_file.write("{0: 1.4e}{1: 1.4e}{2: 1.4e}{3: 1.4e}\n".format(rhot[i], ne_spl(rhop[i]), Te_spl(rhop[i]) * 1.e-3, 1.0))
    except Exception as e:
        print("Failed to load IDA -> only writing EFIT")
    make_E_Fit_EQDSK(working_dir, shot, time, EQObj)
    
    
def make_E_Fit_EQDSK(working_dir, shot, time, EQObj):
    EQDSK_file = open(os.path.join(working_dir, "g{0:d}_{1:05d}".format(shot, int(time*1.e3))), "w")
    dummy = 0.0
    EQ_t = EQObj.GetSlice(time)
    I_p = get_total_current(shot, time)
    today = datetime.date.today()
    # First line
    EQDSK_file.write("{0:50s}".format("EFIT " + str(today.month) + "/" + str(today.day) + "/" + str(today.year) + " #" + str(shot) + \
                                           " " + str(int(time*1.e3)) + "ms"))
    # Second line
    EQDSK_file.write("{0: 2d}{1: 4d}{2: 4d}\n".format(EQObj.EQ_ed, len(EQ_t.R), len(EQ_t.z)))
    # Third line
    EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(EQ_t.R[-1] - EQ_t.R[0], EQ_t.z[-1] - EQ_t.z[0], \
                                                                                  1.65, EQ_t.R[0], np.mean(EQ_t.z)))
    B_axis = np.double(EQObj.get_B_on_axis(time)) * np.sign(np.mean(EQ_t.Bt.flatten()))
    # Fourth line
    EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(EQ_t.R_ax, EQ_t.z_ax, \
                                                                                  EQ_t.Psi_ax, EQ_t.Psi_sep, np.double(B_axis)))
    
    # Fifth line
    EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(I_p, EQ_t.Psi_ax, 0.0, EQ_t.R_ax, 0.0))
    # Sixth line
    EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(EQ_t.z_ax, 0.0, EQ_t.Psi_sep, 0.0, 0.0))
    N = len(EQ_t.R)
    Psi = np.linspace(EQ_t.Psi_ax, EQ_t.Psi_sep, N)
    rhop = np.sqrt((Psi - EQ_t.Psi_ax)/(EQ_t.Psi_sep - EQ_t.Psi_ax))
    quant_dict = {}
    quant_dict["q"] = EQObj.getQuantity(rhop, "Qpsi", time)
    quant_dict["pres"] = EQObj.getQuantity(rhop, "Pres", time)
    quant_dict["pprime"] = EQObj.getQuantity(rhop, "dPres", time)
    quant_dict["ffprime"] = EQObj.getQuantity(rhop, "FFP", time)
    ffp_spl = InterpolatedUnivariateSpline(rhop, quant_dict["ffprime"] )
    f_sq_spl = ffp_spl.antiderivative(1)
    f_spl = InterpolatedUnivariateSpline(rhop, np.sign(B_axis) * \
                                         (np.sqrt(2.0 * f_sq_spl(rhop) +  \
                                                  (EQ_t.R_ax * B_axis)**2)))
    # Get the correct sign back since it is not included in ffprime
    quant_dict["fdia"] = f_spl(rhop)
    N_max = 5
    format_str = " {0: 1.9e}"
    for key in ["fdia", "pres", "ffprime", "pprime"]:
        i = 0
        while i < N:
            EQDSK_file.write(format_str.format(quant_dict[key][i]))
            i += 1
            if(i %  N_max == 0):
                EQDSK_file.write("\n")
    if(i % N_max != 0):
        EQDSK_file.write("\n")
    N_new = 0
    for i in range(len(EQ_t.R)):
        for j in range(len(EQ_t.z)):
            EQDSK_file.write(format_str.format(EQ_t.Psi[i,j]))
            N_new += 1
            if(N_new == N_max):
                EQDSK_file.write("\n")
                N_new = 0
    if(N_new != 0):
        EQDSK_file.write("\n")
    for key in ["q"]:
        i = 0
        while i < N:
            EQDSK_file.write(format_str.format(quant_dict[key][i]))
            i += 1
            if(i %  N_max == 0):
                EQDSK_file.write("\n")
    EQDSK_file.flush()
    EQDSK_file.close()

def fix_wall_file(filename, filename_out):
    old_file = open(filename, "r")
    new_file = open(filename_out, "w")
    for i, line in enumerate(old_file.readlines()):
        if(i <= 1 or len(line) < 20):
            new_file.write(line)
            continue
        a = np.fromstring(line, count=3, dtype=np.double, sep=" ")
        new_file.write("{0: 7.7f} {1: 7.7f} {2: 4d}\n".format(a[0] * 1.e-2, a[1] * 1.e-2, int(a[2])))
    old_file.close()
    new_file.flush()
    new_file.close()        
    
def make_TB_angles_from_launch_vector(x1, x2):
    N_vec = x2 - x1
    N_vec /= np.linalg.norm(N_vec)
    N_vec_sphere = np.zeros(3)
    N_vec_sphere[0] = 1.0
    N_vec_sphere[1] = np.arctan2(N_vec[1], N_vec[0])
    N_vec_sphere[2] = np.arccos(N_vec[2])
    x1_norm = np.copy(x1)
    x1_norm /= np.linalg.norm(x1_norm)
    Radial_N = np.zeros(3)
    Radial_N[0] = 1.0
    Radial_N[1] = np.arctan2(-x1_norm[1], -x1_norm[0])
    Radial_N[2] = np.pi / 2.0
    phi_tor = np.rad2deg(N_vec_sphere[1] - Radial_N[1])
    theta_pol = np.rad2deg(N_vec_sphere[2] - Radial_N[2])  
    return theta_pol,phi_tor

def compare(ECRad_data, Travis_folder, quantity, channelno = None):
    fig = plt.figure(figsize=(9,12))
    ax = fig.add_subplot(111)
    results = ECRadResults()
    results.from_mat_file(ECRad_data)
    twinx = None
    if(not os.path.isfile(os.path.join(Travis_folder, "ECE_spectrum_parsed"))):
        Travis_file = open(os.path.join(Travis_folder, "ECE_spectrum"))
        lines = []
        for line in Travis_file.readlines():
            lines.append(line.replace("X","-1").replace("O","+1"))
        Travis_file.close()
        parsed_Travis_file = open(os.path.join(Travis_folder, "ECE_spectrum_parsed"), "w")
        for line in lines:
            parsed_Travis_file.write(line)
        parsed_Travis_file.flush()
        parsed_Travis_file.close()
    if("Trad" in quantity or "tau" in quantity):
        Travis_Trad = np.loadtxt(os.path.join(Travis_folder, "ECE_spectrum_parsed"),skiprows=1)
        rho = results.Scenario.plasma_dict["rhot_prof"][0]
        Te = results.Scenario.plasma_dict["Te"][0] / 1.e3
        ne = results.Scenario.plasma_dict["ne"][0] / 1.e19
        if("warm" not in quantity and "_f"  not in quantity):
            rho_cold = results.resonance["rhop_cold"][0]
            if("Trad" in quantity):
                try:
                    ax.plot(rho_cold, results.XTrad[0], "rs", label="ECRad Primary")
                    ax.plot(rho_cold, results.XTrad_comp[0], "b*", label="ECRad Secondary")
                except IndexError:
                    ax.plot(rho_cold, results.Trad[0], "rs", label="ECRad Primary")
                    ax.plot(rho_cold, results.Trad_comp[0], "b*", label="ECRad Secondary")
                ax.plot(np.abs(Travis_Trad.T[10]), Travis_Trad.T[3], "c+", label="Travis")
                ax.plot(rho, Te, "k-", label="ECRad $T_\mathrm{e}$")
                ax.set_xlabel(r"$\rho_\mathrm{tor,\,cold}$")
                ax.set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\kilo\electronvolt}]$")
                twinx = ax.twinx()
                twinx.plot(rho, ne, "m:", label="ECRad $n_\mathrm{e}$")
                twinx.set_ylabel(r"$n_\mathrm{e}\,[10^{19}\times\si{\per\cubic\metre}]$")
            elif("tau" in quantity):
                try:
                    ax.plot(rho_cold, results.Xtau[0], "r+", label="ECRad Primary")
                    ax.plot(rho_cold, results.Xtau_comp[0], "b*", label="ECRad Secondary")
                except IndexError:
                    ax.plot(rho_cold, results.tau[0], "r+", label="ECRad Primary")
                    ax.plot(rho_cold, results.tau_comp[0], "b*", label="ECRad Secondary")
                ax.plot(np.abs(Travis_Trad.T[10]), Travis_Trad.T[8], "cs", label="Travis")
                ax.set_xlabel(r"$\rho_\mathrm{tor,\,cold}$")
                ax.set_ylabel(r"$T_\mathrm{rad}\,[\si{\kilo\electronvolt}]$")
            ax.set_xlim(0.0,1.0)
        elif("warm" in quantity):
            rho_warm = results.resonance["rhop_warm"][0]
            if("Trad" in quantity):
                ax.plot(rho_warm, results.XTrad[0], "r+", label="ECRad Primary")
                ax.plot(rho_warm, results.XTrad_comp[0], "b*", label="ECRad Secondary")
                ax.plot(np.abs(Travis_Trad.T[14]), Travis_Trad.T[3], "cs", label="Travis")
                ax.plot(rho, Te, "k-", label="ECRad $T_\mathrm{e}$")
                ax.set_xlabel(r"$\rho_\mathrm{tor,\,warm}$")
                ax.set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\kilo\electronvolt}]$")
                twinx = ax.twinx()
                twinx.plot(rho, ne, "m:", label="ECRad $n_\mathrm{e}$")
                twinx.set_ylabel(r"$n_\mathrm{e}\,[10^{19}\times\si{\per\cubic\metre}]$")
            elif("tau" in quantity):
                ax.plot(rho_warm, results.Xtau[0], "r+", label="ECRad Primary")
                ax.plot(rho_warm, results.Xtau_comp[0], "b*", label="ECRad Secondary")
                ax.plot(np.abs(Travis_Trad.T[14]), Travis_Trad.T[8], "cs", label="Travis")
                ax.set_xlabel(r"$\rho_\mathrm{tor,\,warm}$")
                ax.set_ylabel(r"$\tau_\omega$")
                ax.set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\kilo\electronvolt}]$")
                twinx = ax.twinx()
                twinx.plot(rho, Te, "m:", label="ECRad $T_\mathrm{e}$")
                twinx.set_ylabel(r"$T_\mathrm{e}\,[\si{\kilo\electronvolt}]$")
            ax.set_xlim(0.0,1.0)
        else:
            f = results.Scenario.ray_launch[0]["f"] / 1.e9
            if("Trad" in quantity):
                ax.plot(f, results.XTrad[0], "r+", label="ECRad Primary")
                ax.plot(f, results.XTrad_comp[0], "b*", label="ECRad Secondary")
                ax.plot(f, Travis_Trad.T[3], "cs", label="Travis")
                ax.set_xlabel(r"$f\,[\si{\giga\hertz}]$")
                ax.set_ylabel(r"$T_\mathrm{rad/e}\,[\si{\kilo\electronvolt}]$")
            elif("tau" in quantity):
                ax.plot(f, results.Xtau[0], "r+", label="ECRad Primary")
                ax.plot(f, results.Xtau_comp[0], "b*", label="ECRad Secondary")
                ax.plot(f, Travis_Trad.T[8], "cs", label="Travis")
                ax.set_xlabel(r"$f\,[\si{\giga\hertz}]$")
                ax.set_ylabel(r"$\tau_\omega$")
        
    elif(quantity == "Rres"):
        Travis_Trad = np.loadtxt(os.path.join(Travis_folder, "ECE_spectrum_parsed"),skiprows=1)
        ECRad_chno = np.linspace(1, len(results.resonance["R_cold"][0]),  \
                                 len(results.resonance["R_cold"][0]), dtype = np.int)
        Travis_chno = np.linspace(1, len(Travis_Trad.T[14]), len(Travis_Trad.T[14]),  dtype = np.int)
        ax.plot(ECRad_chno, results.resonance["R_cold"][0], "r+", label="ECRad")
        ax.plot(Travis_chno, Travis_Trad.T[9].T, "b*", label="Travis")
        ax.set_ylabel(r"$R_\mathrm{cold,\,cold}\,[\si{\metre}]$")
        ax.set_xlabel(r"chno.")
    else:
        if(channelno is None):
            print("Channel number is not optional for ray plots")
            return
        Travis_ray = np.loadtxt(os.path.join(Travis_folder, "beamtrace_{0:d}".format(channelno)), skiprows=1)
        if(quantity == "Rz"):
            x = results.ray["xX"][0][channelno - 1][::-1]
            y = results.ray["yX"][0][channelno - 1][::-1]
            z = results.ray["zX"][0][channelno - 1][::-1]
            R = np.sqrt(x**2 + y**2)
            ax.plot(R, z,  "r*", label="ECRad",linestyle="-")
            ax.plot(np.sqrt(Travis_ray.T[2]**2 + Travis_ray.T[3]**2), Travis_ray.T[4], "b+", label="Travis",linestyle="--")
            dR = R[1] - R[0]
            dz = z[1] - z[0]
            x1 = np.array([R[0], z[0]])
            x2 = x1 + np.array([dR, dz]) / np.sqrt(dR**2 + dz**2) # 1 m straight line
            ax.plot(np.array([x1[0], x2[0]]), np.array([x1[1], x2[1]]), "k:", label="straight")
            ax.set_xlabel(r"$R\,[\si{\metre}]$")
            ax.set_ylabel(r"$z\,[\si{\metre}]$")
        elif(quantity == "xy"):
            x = results.ray["xX"][0][channelno - 1][::-1]
            y = results.ray["yX"][0][channelno - 1][::-1]
            ax.plot(x, y, "r*", label="ECRad",linestyle="-")
            ax.plot(Travis_ray.T[2], Travis_ray.T[3], "b+", label="Travis",linestyle="--")
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            x1 = np.array([x[0], y[0]])
            x2 = x1 + np.array([dx, dy]) / np.sqrt(dx**2 + dy**2) # 1 m straight line
            x2 = x1 + np.array([dx, dy]) / np.sqrt(dx**2 + dy**2) # 1 m straight line
            ax.plot(np.array([x1[0], x2[0]]), np.array([x1[1], x2[1]]), "k:", label="straight")
            ax.set_xlabel(r"$x\,[\si{\metre}]$")
            ax.set_ylabel(r"$y\,[\si{\metre}]$")
        elif(quantity == "Nabs"):
            s = results.ray["sX"][0][channelno - 1]
            ax.plot(s, results.ray["NX"][0][channelno - 1], "r-", label="ECRad",linestyle="-")
            ax.plot(s, results.ray["cX"][0][channelno - 1], "b--", label="ECRad cold plasma",linestyle="--")
            ax.plot(Travis_ray.T[1], np.sqrt(Travis_ray.T[5]**2 + Travis_ray.T[6]**2+ Travis_ray.T[7]**2), "g+", label="Travis")
#             ax.plot(Travis_ray.T[1], np.sqrt(Travis_ray.T[14]**2 + Travis_ray.T[15]**2), "k:", label="Travis cold plasma")
            ax.set_xlabel(r"$s\,[\si{\metre}]$")
            ax.set_ylabel(r"$N_\omega$")
        elif(quantity == "Te"):
            s = results.ray["sX"][0][channelno - 1][::-1]
            s -= s[0]
            ax.plot(s, results.ray["TeX"][0][channelno - 1], "r*", label="ECRad",linestyle="-")
            ax.plot(Travis_ray.T[1], Travis_ray.T[10], "g+", label="Travis",linestyle="--")
            ax.set_xlabel(r"$s\,[\si{\metre}]$")
            ax.set_ylabel(r"$T_\mathrm{e} \,[\si{\electronvolt}]$")
#         elif(quantity == "ne"):
#             ax.plot(ECRad_chdata.T[0][-1] - ECRad_chdata.T[0][::-1], ECRad_chdata.T[4][::-1], "r*", label="ECRad",linestyle="-")
#             ax.plot(Travis_ray.T[1], Travis_ray.T[9], "g+", label="Travis",linestyle="--")
#             ax.set_xlabel(r"$s\,[\si{\metre}]$")
#             ax.set_ylabel(r"$n_\mathrm{e}\,[\si{\per\cubic\metre}]$")
#         elif(quantity == "rho"):
#             ax.plot(ECRad_chdata.T[0][-1] - ECRad_chdata.T[0][::-1], ECRad_chdata.T[3][::-1], "r*", label="ECRad",linestyle="-")
#             ax.plot(Travis_ray.T[1], Travis_ray.T[8], "g+", label="Travis",linestyle="--")
#             ax.set_xlabel(r"$s\,[\si{\metre}]$")
#             ax.set_ylabel(r"$\rho_\mathrm{tor}$")
#         elif(quantity == "Btot"):
#             ax.plot(ECRad_BPDray.T[0][-1] - ECRad_BPDray.T[0][::-1], np.sqrt(ECRad_BPDray.T[-1][::-1]**2 + ECRad_BPDray.T[-2][::-1]**2 + ECRad_BPDray.T[-3][::-1]**2), "r*", label="ECRad",linestyle="-")
#             ax.plot(Travis_ray.T[1], np.sqrt(Travis_ray.T[-4]**2 + Travis_ray.T[-5]**2 + Travis_ray.T[-6]**2), "g+", label="Travis",linestyle="--")
#             ax.set_xlabel(r"$s\,[\si{\metre}]$")
#             ax.set_ylabel(r"$\vert B\vert\,[\si{\tesla}]$")
#         elif(quantity == "Bcomp"):
#             ax.plot(ECRad_BPDray.T[0][-1] - ECRad_BPDray.T[0][::-1], ECRad_BPDray.T[-3][::-1], "r*", label="ECRad $B_x$",linestyle="-")
#             ax.plot(ECRad_BPDray.T[0][-1] - ECRad_BPDray.T[0][::-1], ECRad_BPDray.T[-2][::-1], "r*", label="ECRad $B_y$",linestyle="-")
#             ax.plot(ECRad_BPDray.T[0][-1] - ECRad_BPDray.T[0][::-1], ECRad_BPDray.T[-1][::-1], "r*", label="ECRad $B_z$",linestyle="-")
#             ax.plot(Travis_ray.T[1], Travis_ray.T[-6], "g+", label="Travis $B_x$",linestyle="--")
#             ax.plot(Travis_ray.T[1], Travis_ray.T[-5], "g+", label="Travis $B_y$",linestyle="--")
#             ax.plot(Travis_ray.T[1], Travis_ray.T[-4], "g+", label="Travis $B_z$",linestyle="--")
#             ax.set_xlabel(r"$s\,[\si{\metre}]$")
#             ax.set_ylabel(r"$\vert B\vert\,[\si{\tesla}]$")
        elif(quantity == "alpha"):
            s = np.max(results.ray["sX"][0][channelno - 1]) - results.ray["sX"][0][channelno - 1][::-1]
            ax.plot(s, results.ray["abX"][0][channelno - 1][::-1], "r*", label="ECRad primary abs. coeff.", linestyle="-")
            ax.plot(s, results.ray["ab_secondX"][0][channelno - 1][::-1], "mo", label="ECRad secondary abs. coeff.", linestyle=":")
            ax.plot(Travis_ray.T[1], Travis_ray.T[16], "g+", label="Travis abs. coeff.", linestyle="--")
            ax.set_xlabel(r"$R\,[\si{\metre}]$")
            ax.set_ylabel(r"$\alpha_\omega\,[\si{\per\metre}]$")
        elif(quantity == "BPD"):
            R = np.sqrt(results.ray["xX"][0][channelno - 1][::-1]**2 + results.ray["yX"][0][channelno - 1][::-1]**2)
            ax.plot(R, results.ray["BPDX"][0][channelno - 1][::-1] / np.max(results.ray["BPDX"][0][channelno - 1]), "r*", label="ECRad primary BPD",linestyle="-")
            ax.plot(R, results.ray["BPD_secondX"][0][channelno - 1][::-1] / np.max(results.ray["BPD_secondX"][0][channelno - 1]), "mo", label="ECRad secondary BPD",linestyle=":")
            ax.plot(np.sqrt(Travis_ray.T[2]**2 + Travis_ray.T[3]**2), Travis_ray.T[21]* Travis_ray.T[23] / np.max(Travis_ray.T[21]* Travis_ray.T[23]), "g+", label="Travis BPD",linestyle="--")
            ax.set_xlabel(r"$R\,[\si{\metre}]$")
            ax.set_ylabel(r"$D_\omega$")
    if(twinx is None ):
        plt.legend()
    else:
        lns = ax.get_lines() + twinx.get_lines()
        labs = [l.get_label() for l in lns]
        leg = ax.legend(lns, labs)
        leg.get_frame().set_alpha(0.5)
        leg.draggable()
    plt.tight_layout()
    #plt.show()
                
 
def test_mconf(filename):
    Config = {'B0_angle': 0.0, \
              'extraLCMS': 1.2, \
              'accuracy': 1e-10,\
              'truncation': 1e-10}
    EQU = mconf.Mconf_equilibrium(filename, Config)
    x = np.array([-4.40760208218341,-4.12896886742981,0.209955400463033 ])
    s, B = EQU.get_s_and_B(x[0], x[1], x[2])
    print(np.sqrt(s))
    s, B = EQU.get_s_B(x[0], x[1], x[2])
    print(np.sqrt(s))
    s2, B, grad_B, grad_s = EQU.grad_B_grad_s(x)
    print(np.sqrt(s2))
    print(grad_s / (2.0 * np.sqrt(s2)))
                
                
if(__name__ == "__main__"):
#     test_mconf("/tokp/scratch/sdenk/ECRad/ECRad_data/W7X-EIM-stand5ard-beta=0.wout")
#     compare('/tokp/work/sdenk/ECRad/ECRad_20181009043_EXT_ed16.mat', "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/example/travisinput/results/", "Trad_warm", 2)
#compare('/tokp/work/sdenk/ECRad/ECRad_20180823016002_EXT_ed19.mat', "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/Travis_ECRad_Benchmark/20180823.016.002_4_45s/results_0/", "Trad", 7)
    compare('/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/Travis_ECRad_Benchmark/20180823.016.002_4_45s/ECRad_20180823016002_EXT_ed19.mat', \
             "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/Travis_ECRad_Benchmark/20180823.016.002_4_45s/results_0/", "tau", 7)
#      compare('/tokp/work/sdenk/ECRad/ECRad_20181009043_EXT_ed18.mat', "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/low_ne_case/results_0/", "Rz", 2)
#     compare('/tokp/work/sdenk/ECRad/ECRad_20181009043_EXT_ed18.mat', "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/W7X_stuff/example2/results/", "Rz", 2)
    plt.show()
#     make_Travis_input_from_AUG("/tokp/work/sdenk/ECRad/", 32934, 3.298, "SDENK", 0, "AUGD", "EQH", 0)
#     print("Travis output", make_TB_angles_from_launch_vector(np.array([-4.731100E+000, -4.571900E+000,  2.723000E-001]), \
#                                             np.array([-4.459112E+000, -4.199626E+000,  2.198411E-001])))
#     fix_wall_file("/tokp/work/sdenk/ECRad/W7X_wall.dat", "/tokp/work/sdenk/ECRad/W7X_wall_SI.dat")
    