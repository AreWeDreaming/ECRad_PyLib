'''
Created on Oct 7, 2016

@author: sdenk
'''
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.integrate import simps
import os
import scipy.constants as cnst
from GlobalSettings import AUG, TCV
import ECRad_Results
try:
    import h5py
except ImportError:
    print("h5py not available")
from electron_distribution_utils import read_svec_dict_from_file, load_f_from_ASCII, \
                                        read_LUKE_data, read_Fe, Gauss_norm, \
                                        Gauss_not_norm, Juettner2D, Juettner2D_bidrift, multi_slope, \
                                        RunAway2D, make_dist_from_Gene_input, get_dist_moments, \
                                        get_dist_moments_non_rel, Gene_BiMax, Juettner1D, \
                                        get_B_min_from_file, read_waves_mat_to_beam, read_ray_dict_from_file, \
                                        find_cold_res, SynchrotonDistribution, get_0th_and_2nd_moment, remap_f_Maj, \
                                        g0_approx, g2_approx
from scipy.optimize import fsolve
# if(AUG):
#    from equilibrium_utils_AUG import EQData
#    from shotfile_handling_AUG import get_freqs, get_RMC_data_calib, get_data_calib
# elif(TCV):
#    from equilibrium_utils_TCV import EQData
# else:
#    print('Neither AUG nor TCV selected')
#    raise(ValueError('No system selected!'))
from Diags import Diag
from em_Albajar import em_abs_Alb, distribution_interpolator, \
                       gene_distribution_interpolator, s_vec, \
                       rotate_vec_around_axis, N_with_pol_vec
from plotting_configuration import *
import glob
from scipy.io import loadmat
from ECRad_Results import ECRadResults
from Fitting import make_fit
from scipy.stats import linregress
from scipy import odr
from equilibrium_utils import EQDataExt
from GlobalSettings import AUG, TCV
if(AUG):
    from equilibrium_utils_AUG import EQData
elif(TCV):
    from equilibrium_utils_TCV import EQData

wave_folder = "/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/DistData"



def repair_ECRad_results(folder_in, folder_out=None):
    # Allows to make bulk modification of result files using glob
    # If folder_out is True it overwrites!
    filelist = glob.glob(os.path.join(folder_in, "*.mat"))
#     filelist = ['/tokp/work/sdenk/DRELAX_Results/ECRad_35662_ECECTACTC_run0208.mat']
    cur_result = ECRadResults()
    for filename in filelist:
        cur_result.reset()
        cur_result.from_mat_file(filename)
        # Enter bulk modifcations here
        cur_result.Scenario.used_diags_dict["CTC"].diag = "CTC"
        if(folder_out is None):
            cur_result.to_mat_file(filename)
        else:
            cur_result.to_mat_file(os.path.join(folder_out, os.path.basename(filename)))
            

def get_files_and_labels(dstf, simpl=None):
        third_model = False
        data_third_name = None
        backup_name = None
        backup_dist = None
        tertiary_dist = None
        if(dstf == "Mx"):
            simplfile = "TRadM_Maxwl.dat"
        elif(dstf == "TB" or dstf == "BA"):
            simplfile = "TRadM_TBeam.dat"
        elif(dstf == "TO"):
            simplfile = "TRadM_TBold.dat"
        elif(dstf == "OM"):
            simplfile = "TRadM_olMax.dat"
        elif(dstf == "OB" or dstf == "ON"):
            simplfile = "TRadM_noBes.dat"
            data_third_name = "TRadM_Maxwl.dat"
        elif(dstf == "O1"):
            simplfile = "TRadM_O1mod.dat"
        elif(dstf == "GB"):
            simplfile = "TRadM_GComp.dat"
        elif(dstf == "Al"):
            simplfile = "TRadM_olBes.dat"
        else:
            simplfile = "TRadM_therm.dat"
            backup_name = "TRadM_TBeam.dat"
        if(dstf == "Th" or dstf == "Mx" or dstf == "TB" or dstf == "O1"  or dstf == "ON"  or dstf == "BA"):
            data_name = "TRadM_therm.dat"
        elif(dstf == "OM"):
            data_name = "TRadM_noBes.dat"
        elif(dstf == "OB"):
            data_name = "TRadM_olBes.dat"
        elif(dstf == "Re"):
            data_name = "TRadM_RELAX.dat"
            third_model = True
            data_third_name = "TRadM_LUKE.dat"
        elif(dstf == "Al"):
            data_name = "TRadM_Maxwl.dat"
            third_model = True
            data_third_name = "TRadM_oNBes.dat"
        elif(dstf == "Ge"):
            data_name = "TRadM_GENE.dat"
        elif(dstf == "GB"):
            data_name = "TRadM_GENE.dat"
        elif(dstf == "Lu"):
            data_name = "TRadM_LUKE.dat"
            third_model = True
            data_third_name = "TRadM_RELAX.dat"
        elif(dstf == "SH"):
            data_name = "TRadM_SpitH.dat"
        elif(dstf == "Pd"):
            data_name = "TRadM_Pdstl.dat"
        elif(dstf == "BJ"):
            data_name = "TRadM_BiMnJ.dat"
        elif(dstf == "BM"):
            data_name = "TRadM_BiMax.dat"
        elif(dstf == "DM"):
            data_name = "TRadM_Drift.dat"
        elif(dstf == "MS"):
            data_name = "TRadM_MultS.dat"
            third_model = True
            data_third_name = "TRadM_Maxwl.dat"
        elif(dstf == "RA"):
            data_name = "TRadM_RunAw.dat"
        elif(dstf == "Ta"):
            data_name = "TRadM_analy.dat"
        elif(dstf == "TO"):
            data_name = "TRadM_thrms.dat"
        else:
            print("Unknown dstf", dstf)
            raise ValueError
        dist = r"f_\mathrm{MJ}"
        if(dstf == "Re"):
            dist = r"f_\mathrm{RELAX}"
            tertiary_dist = r"f_\mathrm{LUKE}"
        elif(dstf == "Lu"):
            dist = r"f_\mathrm{LUKE}"
            tertiary_dist = r"f_\mathrm{RELAX}"
        elif(dstf == "Ge"):
            dist = r"f_\mathrm{GENE}"
        elif(dstf == "GB"):
            dist = r"f_\mathrm{BiMax,GENE}"
        elif(dstf == "SH"):
            dist = r"f_\mathrm{Spitzer\,Haerm}"
        elif(dstf == "Pd"):
            dist = "ECRH-Pedestal distribution"
        elif(dstf == "BM"):
            dist = r"f_\mathrm{BiM}"
        elif(dstf == "BJ"):
            dist = r"f_\mathrm{BiMJ}"
        elif(dstf == "DM"):
            dist = "Drifting Maxwellian distribution"
        elif(dstf == "RA"):
            dist = "Runaway distribution"
        elif(dstf == "MS"):
            dist = r"f_\mathrm{TwoSlope}"  # ,\, R = 90"
            tertiary_dist = r"f_\mathrm{Maxwell}"
        elif(dstf == "Ta"):
            dist = "Analytical Integration"
        elif(dstf == "Mx"):
            dist = r"ECRad"
        dist = r"\left[" + dist + r"\right]"
        if(dstf == "Mx"):
            dist_simpl = r"\mathrm{Maxwellian}"  # "f_\mathrm{M}"
            dist = r"\mathrm{Maxwell\mbox{-}J\ddot{u}ttner}"
            dist = r"\left[" + dist + r"\right]"
        elif(dstf == "OM"):
            dist_simpl = r"\mathrm{Maxwellian}"  # "f_\mathrm{M}"
            dist = r"\mathrm{Maxwell\mbox{-}J\ddot{u}ttner}"
            dist = r"\left[" + dist + r"\right]"
        elif(dstf == "OB"):
            dist_simpl = r"\mathrm{Expanded\,bessel}"  # "f_\mathrm{M}"
            dist = r"\mathrm{Full\,bessel}"
            dist = r"\left[" + dist + r"\right]"
        elif(dstf == "TB" or dstf == "BA"):
            dist_simpl = r"\mathrm{warm\,dispersion}"  # "Maxwell dist integrated w. fwd. Euler"#!$\alpha$
            dist = r"\mathrm{tenuous\,plasma}"
            dist = r"\left[" + dist + r"\right]"
        elif(dstf == "TO"):
            dist_simpl = r"\mathrm{warm\,dispersion}"  # "Maxwell dist integrated w. fwd. Euler"#!$\alpha$
            dist = r"\mathrm{Imp.\,vacuum\,model}"
            dist = r"\left[" + dist + r"\right]"
        elif(dstf == "Al"):
            dist_simpl = r"\mathrm{\,vacuum\,model}"
            dist = r"\mathrm{tenuous\,plasma}"
            dist = r"\left[" + dist + r"\right]"
            tertiary_dist = r"\mathrm{imp.\,vacuum\,model}"
        elif(dstf == "ON"):
            dist_simpl = r"{\mathrm{Model } A}"
            dist = r"{\mathrm{Model } B}"
            dist = r"\left[" + dist + r"\right]"
            tertiary_dist = r"{\mathrm{Model } A'}"
        elif(dstf == "O1"):
            dist_simpl = "O1"
        elif(dstf == "Th"):
            dist_simpl = r"\mathrm{Impr.\,Hutchinson}"  # \,\mathrm{no \, O1}#,\, R = 0.85"#"\right.$fwd. Euler$\left."#non-thermal dist. R = 0.95"#r"without O1-mode"#$R = 0.95$"#" fwd. Euler"
            # dist = r"\right.$Rk-4$\left."
        elif(dstf == "Ge"):
            dist_simpl = r"f_\mathrm{0,GENE}"
        elif(dstf == "GB"):
            dist_simpl = r"f_\mathrm{BiMax, Analytical}"
        else:
            if(simplfile != "TRadM_TBeam.dat"):
                dist_simpl = r"f_\mathrm{MJ}"  # "Internal distribution"#"r
                backup_dist = "\mathrm{Farina\,model}"
                backup_dist = r"\left[" + backup_dist + r"\right]"
            else:
                dist_simpl = r"f_\mathrm{MJ}"  # $ fully rel. disp$ "Internal distribution"#"r
        if(len(dist_simpl) > 0):
            dist_simpl = r"\left[" + dist_simpl + r"\right]"
        if(tertiary_dist is not None):
            tertiary_dist = r"\left[" + tertiary_dist + r"\right]"
        return data_name, simplfile, third_model, data_third_name, dist, dist_simpl, backup_name, backup_dist, tertiary_dist

def solve_Te_perp_Te_par(Te_vec, args):
    Te_perp_grad = args[0]
    Te_par_grad = args[1]
    u_perp = args[2]
    u_par = args[3]
    T0 = Te_vec[1] ** (1.0e0 / 3.0e0) * Te_vec[1] ** (2.0e0 / 3.0e0)
    r = T0 / Te_vec[1]
    s = T0 / Te_vec[0]
    gamma = np.sqrt(1.0 + r * u_par ** 2 + s * u_perp ** 2)
    mu_perp = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * Te_vec[0])
    mu_par = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * Te_vec[1])
    return np.array([Te_perp_grad + mu_perp * u_perp / gamma, \
                    Te_par_grad + mu_par * u_par / gamma])

class resonance:
    def __init__(self, path, shot, time, channelno, dstf, mode, rhop_in, HFS, beta, \
                 EQ_slice, B_ax):
        ecfm_data = os.path.join(path, "ecfm_data")
        flag_use_ASCII = True
        flag_show_ECRH = False
        flag_show_ECE = True
        flag_show_j = False
        gy_freq = 140.e9
        gy_mode = +1
        c_lower_lim = 1.e-20
        Te_ext = -1
        if(np.abs(mode) != 1):
            print("Mode has to be either +1 (X) or -1 (O). mode:", mode)
            return
        elif(mode == -1):
            print("O mode selected")
            mode_str = "O"
        elif(mode == 1):
            print("X mode selected")
            mode_str = "X"
        svec_dict, ece_freq = read_svec_dict_from_file(ecfm_data, channelno, mode=mode_str)
        self.omega = ece_freq * 2.0 * np.pi
        n_min = 2
        n_max = 3  # note that the harmonic number set here is included
        Te_filename = os.path.join(ecfm_data, "Te_file.dat")
        Te_data = np.loadtxt(Te_filename, skiprows=1)
        rhop_vec_Te = Te_data.T[0]
        Te_vec = Te_data.T[1]
        ne_filename = os.path.join(ecfm_data, "ne_file.dat")
        ne_data = np.loadtxt(ne_filename, skiprows=1)
        rhop_vec_ne = ne_data.T[0]
        ne_vec = ne_data.T[1]
        Te_spl = InterpolatedUnivariateSpline(rhop_vec_Te, Te_vec)
        ne_spl = InterpolatedUnivariateSpline(rhop_vec_ne, ne_vec)
        Alb = em_abs_Alb()
        self.rhop = rhop_in
        self.success = False
        Gray_beam = None
        try:
            mdict = loadmat(os.path.join(wave_folder, "GRAY_rays_{0:d}_{1:1.2f}.mat".format(shot, time)), squeeze_me=True)
            Gray_beam = read_waves_mat_to_beam(mdict, EQ_slice)
        except IOError as e:
            print("Failed to load waves for this shot, because of:")
            print(e)
        freq_2X_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, np.sqrt(EQ_slice.Br ** 2 + EQ_slice.Bt ** 2 + EQ_slice.Bz ** 2) * cnst.e / cnst.m_e / np.pi)
        if(dstf == "Re" or dstf == "Lu"):
            if(flag_use_ASCII):
                f_folder = os.path.join(ecfm_data, "f" + dstf)
                full_dist = load_f_from_ASCII(f_folder, rhop_in)
                x = full_dist.u
                y = full_dist.pitch
                Fe = np.log(full_dist.f)
                self.rhop = full_dist.rhop[0]
                Fe = Fe[0]
                self.B_min = full_dist.B_min(self.rhop).item()
                # Fe = np.exp(Fe)
                print("Distribution shape", x.shape, y.shape, Fe.shape)
            elif(dstf == "Lu"):
                full_dist = read_LUKE_data(ecfm_data)
                self.rhop = full_dist.rhop
                x = full_dist.u
                y = full_dist.rhop
                Fe = full_dist.f
            else:
                ipsi, psi, x, y, Fe = read_Fe(ecfm_data)
                rhop = np.sqrt(psi)
                irhop = np.argmin(np.abs(rhop - rhop_in))
                self.rhop = rhop[irhop]
                Fe = Fe[irhop]
            # bt_vac correction not necessary here since constant factor that cancels out for B/B_min
        elif(dstf != "Ge"):
            self.B_min = 100  # We want pitch := u_par / u here
            if(dstf == "DM"):
                beta[0] = Gauss_norm(self.rhop, [beta[0], beta[1], beta[2]])
            elif(dstf == "BJ"):
                beta[0] = Gauss_norm(self.rhop, [beta[0], beta[1], beta[2]])
                beta[5] = 0.0
                beta[6] = 0.0
            elif(dstf == "BM"):
                beta[0] = Gauss_norm(self.rhop, [beta[0], beta[1], beta[2]])
                beta[5] = 0.0
                beta[6] = 0.0
            elif(dstf == "SH"):
                print("dstf = SH is not supported!")
                return
            elif(dstf == "RA"):
                beta[0] = Gauss_norm(self.rhop, [beta[0], beta[1], beta[2]])
            elif(dstf == "GB"):
                beta[0] = 1.0
                beta[5] = 0.0
                beta[6] = 0.0
            elif(dstf != "MS"):
                beta[0] = 0.0
            x = np.linspace(0.0, 3.0, 200)
            y = np.linspace(0.0, np.pi, 200)
            Fe = np.zeros((len(x), len(y)))
            self.rhop = rhop_in
            if((dstf != "SH" and dstf != "MS") and beta[0] == 0.0):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = Juettner2D(u_par, u_perp, Te_spl(rhop_in))
            elif(dstf != "SH" and dstf != "MS" and dstf != "RA" and dstf != "GB" and beta[0] != 0.0):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = (1.e0 - beta[0]) * (Juettner2D(u_par, u_perp, Te_spl(rhop_in)) + \
                                    beta[0] * Juettner2D_bidrift(u_par, u_perp, beta[3], beta[4], beta[5], beta[6]))
            elif(dstf == "GB"):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = Juettner2D_bidrift(u_par, u_perp, beta[3], beta[4], beta[5], beta[6])
            elif(dstf == "MS"):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = multi_slope(u_par, u_perp, Te_spl(rhop_in), \
                                               beta[1], (1.0 - beta[0] * Te_spl(rhop_in) ** 2 / np.max(Te_vec) ** 2) * Te_spl(rhop_in))
            elif(dstf == "RA"):
                for i in range(len(x)):
                    u_perp = np.cos(y) * x[i]
                    u_par = np.sin(y) * x[i]
                    Fe[i] = (RunAway2D(u_par, u_perp, Te_spl(rhop_in), ne_spl(rhop_in), \
                                               beta[0], beta[3], beta[4]))
            else:
                print("The chosen dstf is not supported", dstf)
            Fe[Fe < 1.e-30] = 1.e-30
            Fe = np.log(Fe)
        else:
            f_folder = os.path.join(ecfm_data, "f" + dstf)
            self.rhop, beta_par, mu_norm, Fe, B0 = load_f_from_ASCII(f_folder, rhop_in, Gene=True)
            # np.abs(g)
            Alb.B_min = B0
        if(dstf != "Ge"):
            dist_obj = distribution_interpolator(x, y, RectBivariateSpline(x, y, Fe))
            Alb.dist_mode = "ext"
            Alb.ext_dist = dist_obj
            Alb.B_min = self.B_min
            pmax = max(x)
        else:
            dist_obj = gene_distribution_interpolator(beta_par, mu_norm, RectBivariateSpline(beta_par, mu_norm, Fe))
            Alb.dist_mode = "gene"
            Alb.ext_dist = dist_obj
            pmax = np.max(2 * beta_par)
        if(np.min(svec_dict["rhop"]) > self.rhop):
            print("LOS does not intersect requested flux surface")
            print("Smallest rhop on LOS", np.min(svec_dict["rhop"]))
            return
        npts = 400
        intersect_spl = InterpolatedUnivariateSpline(svec_dict["s"], svec_dict["rhop"] - self.rhop)
        s_res_list = intersect_spl.roots()
        print("possible resonances", s_res_list)
        omega_c_spl = InterpolatedUnivariateSpline(svec_dict["s"], svec_dict["freq_2X"] * np.pi)
        omega_c_ax = B_ax * cnst.e / cnst.m_e
        omega_c_max = np.max(svec_dict["freq_2X"]) * np.pi
        if(HFS and omega_c_ax > omega_c_max):
            print("LOS does not cross HFS")
            return
        elif(len(s_res_list > 2) and omega_c_ax > omega_c_max):
            print("More than one resonance found on LFS - the second one will be ignored")
        elif(len(s_res_list > 2)):
            if(HFS):
                FS_str = "HFS"
            else:
                FS_str = "LFS"
            print("More than two resonances found. Picking the one closest to the antenna on the " + FS_str + ".")
        omega_c_res = omega_c_spl(s_res_list)
        if(HFS):
            if(len(s_res_list[omega_c_ax < omega_c_res]) == 0):
                print("All found intersection with the chosen flux surface lie on the LFS")
                return
            s_res = s_res_list[omega_c_ax < omega_c_res][np.argmax(s_res_list[omega_c_ax < omega_c_res])]
        else:
            if(len(s_res_list[omega_c_ax >= omega_c_res]) == 0):
                print("All found intersection with the chosen flux surface lie on the HFS")
                return
            s_res = s_res_list[omega_c_ax >= omega_c_res][np.argmax(s_res_list[omega_c_ax >= omega_c_res])]
        self.omega_c = omega_c_spl(s_res)
        theta_spl = InterpolatedUnivariateSpline(svec_dict["s"], svec_dict["theta"])
        self.theta = theta_spl(s_res)
        if(Te_ext <= 0):
            self.Te = Te_spl(self.rhop)
        else:
            self.Te = Te_ext
        ne = ne_spl(self.rhop)
        svec = s_vec(self.rhop, self.Te, ne, self.omega_c / np.pi, self.theta)
        print("svec", self.rhop, self.Te, ne, self.omega_c / np.pi, self.theta)
        self.u_par = np.linspace(-pmax, +pmax, npts)
        self.u_perp = np.linspace(0., pmax, npts)
        self.log10_f = np.zeros([len(self.u_par), len(self.u_perp)])
        self.Te_perp = np.zeros([len(self.u_par), len(self.u_perp)])
        self.Te_par = np.zeros([len(self.u_par), len(self.u_perp)])
        self.resonance_lines = []
        self.ray_resonance_lines = []
        self.beam_resonance_lines = []
        self.log10_f_along_res = []
        self.log10_f_therm_along_res = []
        self.Trad_nth_along_res = []
        self.tau_along_res = []
        self.harmonic = []
        self.N_abs = Alb.refr_index(svec, self.omega, mode)
        if(dstf != "Ge"):
            Alb.dist_mode = "ext"
        else:
            Alb.dist_mode = "gene"
        for n in range(n_min, n_max + 1):
            if(dstf != "Ge"):
                Alb.dist_mode = "ext"
            else:
                Alb.dist_mode = "gene"
            ray_list = glob.glob(os.path.join(ecfm_data, "Ich" + dstf, "BPD_ray*ch{0:03d}_{1:s}.dat".format(int(channelno), mode_str)))
            for ray_file in ray_list:
                if(not "ray001" in ray_file):
                    Raydata = np.loadtxt(ray_file)
                    Raydata = Raydata[Raydata.T[4] != -1.0]
                    Raydata = Raydata[::-1]
                    i = 0
                    while(self.rhop < Raydata.T[4][i] and i < len(Raydata.T[4]) - 1):
                        i += 1
                    if(i == len(Raydata.T[4])):
                        continue
                    freq_2X_ray = freq_2X_spl(np.sqrt(Raydata.T[1][i] ** 2 + Raydata.T[2][i] ** 2), Raydata.T[3][i], grid=False)
                    print(freq_2X_ray / svec.freq_2X)
                    svec_ray = s_vec(self.rhop, self.Te, ne, svec.freq_2X, Raydata.T[-1][i])
                    u_par_ray, u_perp_ray, c_abs_ray, j_ray = Alb.abs_Albajar_along_res(svec_ray, self.omega, mode, n)
                    if(Alb.resonant):
                        self.ray_resonance_lines.append([u_par_ray, u_perp_ray])
            if(Gray_beam is not None):
                for ibeam in range(len(Gray_beam.rays)):
                    for iray in range(len(Gray_beam.rays[ibeam])):
                        rhop_spline = InterpolatedUnivariateSpline(Gray_beam.rays[ibeam][iray]["s"], Gray_beam.rays[ibeam][iray]["rhop"] - self.rhop)
                        roots = rhop_spline.roots()
                        omega_c_beam_spl = InterpolatedUnivariateSpline(Gray_beam.rays[ibeam][iray]["s"], Gray_beam.rays[ibeam][iray]["omega_c"])
                        if(np.isscalar(roots)):
                            s_inter = roots
                        elif(len(roots) == 0):
                            continue
                        elif(not HFS):
                            s_inter = roots[0]
                        elif(np.any(omega_c_beam_spl(roots) > omega_c_ax)):
                            for root in roots:
                                if(omega_c_ax < omega_c_beam_spl(root)):
                                    s_inter = root
                                    break
                        else:
                            continue
                        omega_c_beam = omega_c_beam_spl(s_inter)
                        N_par_beam = InterpolatedUnivariateSpline(Gray_beam.rays[ibeam][iray]["s"], Gray_beam.rays[ibeam][iray]["Npar"])(s_inter)
                        omega_beam = 2.0 * np.pi * 140.e9
                        omega_bar_beam = omega_beam / omega_c_beam
                        if(N_par_beam ** 2 >= 1.0):
                            continue
                        m_0 = np.sqrt(1.e0 - N_par_beam ** 2) * omega_bar_beam
                        t = np.linspace(-1.0, 1.0, 60)
                        u_par_beam = 1.e0 / np.sqrt(1.e0 - N_par_beam ** 2) * (float(n) / m_0 * N_par_beam + \
                                               np.sqrt((float(n) / m_0) ** 2 - 1.e0) * t)
                        u_perp_beam_sq = ((float(n) / m_0) ** 2 - 1.e0) * (1.e0 - t ** 2)
                        u_perp_beam_sq[u_perp_beam_sq < 0] += 1.e-7
                        if(np.all(u_perp_beam_sq >= 0)):
                            u_perp_beam = np.sqrt(u_perp_beam_sq)
                        else:
                            continue
                        self.beam_resonance_lines.append([u_par_beam, u_perp_beam])
            c_abs_int, j_int = Alb.abs_Albajar(svec, self.omega, mode, n_max=n, n_min=n)
            print("Total absorption n = " + str(n), c_abs_int)
            u_par, u_perp, c_abs, j = Alb.abs_Albajar_along_res(svec, self.omega, mode, n)
            if(Alb.resonant):
                self.resonance_lines.append([u_par, u_perp])
                dist_vals = Alb.dist(u_par, u_perp, \
                                                     (cnst.m_e * cnst.c ** 2) / (svec.Te * cnst.e), svec)
                dist_vals[dist_vals < 1.e-20] = 1.e-20
                self.log10_f_along_res.append(np.log10(dist_vals))
#                self.log10_f_along_res.append(Alb.dist(u_par, u_perp, \
#                                                     (cnst.m_e * cnst.c ** 2) / (svec.Te * cnst.e), svec))
                self.tau_along_res.append(c_abs * 1.e-2)
                self.Trad_nth_along_res.append(np.zeros(len(u_par)))
                self.Trad_nth_along_res[-1][c_abs > c_lower_lim] = j[c_abs > c_lower_lim] / c_abs[c_abs > c_lower_lim] * cnst.c ** 2 / ece_freq ** 2 / cnst.e
                self.Trad_nth_along_res[-1][c_abs <= c_lower_lim] = 0.0
                self.Trad_nth_along_res[-1][self.Trad_nth_along_res[-1] <= self.Te * 0.1] = self.Te * 0.1
                Alb.dist_mode = "thermal"
                self.log10_f_therm_along_res.append(np.log10(Alb.dist(u_par, u_perp, \
                                                     (cnst.m_e * cnst.c ** 2) / (svec.Te * cnst.e), svec)))
#                self.log10_f_therm_along_res.append(Alb.dist(u_par, u_perp, \
#                                                     (cnst.m_e * cnst.c ** 2) / (svec.Te * cnst.e), svec))
                self.harmonic.append(n)
        u_par_cnst = np.zeros(len(self.u_perp))
        if(dstf != "Ge"):
            Alb.dist_mode = "ext"
        else:
            Alb.dist_mode = "gene"
        for i in range(len(self.u_par)):
            u_par_cnst[:] = self.u_par[i]
            self.log10_f[i] = Alb.dist(u_par_cnst, self.u_perp, \
                                       (cnst.m_e * cnst.c ** 2) / (svec.Te * cnst.e), svec)
            if(Alb.dist_mode == "gene"):
                # Assumes a non-rel. BiMaxwellian
                vpar, mu = Alb.v_par_mu_to_cyl(u_par_cnst, self.u_perp, svec)
                self.Te_perp[i] = -Alb.ext_dist.eval_dist_gene_dmu(vpar, mu, mode="spline") / Alb.B_min
                self.Te_perp[i][np.logical_or(np.logical_or(vpar <= Alb.ext_dist.vpar_min, \
                                                vpar >= Alb.ext_dist.vpar_max), \
                                                np.logical_or(mu <= Alb.ext_dist.mu_min, \
                                                mu >= Alb.ext_dist.mu_max))] = 0.0
                self.Te_par[i] = -Alb.ext_dist.eval_dist_gene_dvpar(vpar, mu, mode="spline") / vpar
                self.Te_par[i][np.logical_or(np.logical_or(vpar <= Alb.ext_dist.vpar_min, \
                                                vpar >= Alb.ext_dist.vpar_max), \
                                                np.logical_or(mu <= Alb.ext_dist.mu_min, \
                                                mu >= Alb.ext_dist.mu_max))] = 0.0
                self.Te_perp[i] = (cnst.m_e * cnst.c ** 2) / cnst.e / self.Te_perp[i]
                self.Te_par[i] = (cnst.m_e * cnst.c ** 2) / cnst.e / self.Te_par[i]
            else:
                # Assumes rel. BiMaxwellian
                # Poses a system of coupled non-linear equations
                # Could be quite time consuming...:
#                print("Working on step {0:d}/{1:d}".format(i, len(self.u_par)))
                f = Alb.dist(u_par_cnst, self.u_perp, cnst.m_e * cnst.c ** 2 / (cnst.e * svec.Te), svec)
                dfduperp = Alb.f_deriv_u_perp(u_par_cnst, self.u_perp, svec) / f
                dfdupar = Alb.f_deriv_u_par(u_par_cnst, self.u_perp, svec) / f
                for j in range(len(self.u_perp)):
                    if(np.abs(dfdupar[j]) < 1.e-4 or np.abs(self.u_perp[j]) < 1.e-4 or f[j] < 1.e-8):
                        self.Te_perp[i][j] = 0.0
                        self.Te_par[i][j] = 0.0
                    else:
                        self.Te_perp[i][j], self.Te_par[i][j] = fsolve(solve_Te_perp_Te_par, [svec.Te, svec.Te], \
                                                                       args=[dfduperp[j], dfdupar[j], self.u_perp[j], \
                                                                             u_par_cnst[j]])
                # Assumes rel. BiMaxwellian
            self.log10_f[i, self.log10_f[i] < 1.e-20] = 1.e-20
            self.log10_f[i] = np.log10(self.log10_f[i])
        self.success = True
        return

def test_resonance():
#    reso = resonance("/ptmp1/work/sdenk/nssf/33697/4.80/OERT/ed_39/", 33697, 4.80, 43, "Re", 1, 0.05, False, [0.0], \
#                                    eq_exp="AUGD", eq_diag="IDE", eq_ed=0, bt_vac_correction=1.005)nssf/33585/3.00/OERT/ed_4/
    reso = resonance("/tokp/work/sdenk/ECRadw/", 34663, 3.60, 4, "Th", 1, 0.25, False, [0.0], \
                                    eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005)
#    Te_perp = np.zeros(len(reso.u_par))
#    for i in range(len(Te_perp)):
#        Te_perp[i] = reso.Te_perp[i, i]
#    plt.plot(reso.u_par, Te_perp)
    cmap = plt.cm.get_cmap("plasma")
    levels = np.linspace(500, 700, 30)
    cont1 = plt.contourf(reso.u_par, reso.u_perp, reso.Te_perp.T, levels=levels, cmap=cmap)
    cb = plt.gcf().colorbar(cont1, ax=plt.gca(), ticks=[500, 550, 600, 650, 700])  # ticks = levels[::6] #,
    cb.set_label(r"$T_{\mathrm{e},\perp}$")
    cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([500, 700, 10.0])))
    cb.ax.minorticks_on()
    plt.gca().set_xlabel(r"$u_{\Vert}$")
    plt.gca().set_ylabel(r"$u_{\perp}$")
    plt.show()
#    cmap = plt.cm.get_cmap("plasma")
#    levels = np.linspace(0, 1.e5, 30)
#    cont1 = plt.contourf(reso.u_par, reso.u_perp, reso.Te_perp.T, levels=levels, cmap=cmap)  # , norm=LogNorm()
#    cb = plt.gcf().colorbar(cont1, ax=plt.gca(), ticks=np.linspace(0, 1.e5, 5))  # ticks = levels[::6] #,
#    cb.set_label(r"$T_{\mathrm{e},\perp}$")
#    cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.linspace(0, 1.e5, 5)))  # 10.0 ** np.linspace(3, 5, 3)
#    cb.ax.minorticks_on()
#    plt.gca().set_xlabel(r"$u_{\Vert}$")
#    plt.gca().set_ylabel(r"$u_{\perp}$")
#    plt.show()
# test_resonance()

def exampl_reso(fs, f_cs, thetas, Te, ne, n=2):
    Alb = em_abs_Alb()
    rhop = 0.0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    u_par_range = [np.inf, -np.inf]
    for f, f_c, theta in zip(fs, f_cs, thetas):
        omega = 2.0 * np.pi * f
        omega_c = 2.0 * np.pi * f_c
        svec = s_vec(rhop, Te, ne, omega_c / np.pi, theta)
        u_par, u_perp, c_abs, j = Alb.abs_Albajar_along_res(svec, omega, 1, n)
        print("Resonant: ", Alb.resonant)
        ax.plot(u_perp, u_par)
    u_par_mesh = np.linspace(-1.5, 1.5, 200)
    u_perp_mesh = np.linspace(0, 1.5, 100)
    u_par_dummy = np.zeros(100)
    f = np.zeros((200, 100))
    for i in range(len(u_par_mesh)):
        u_par_dummy[:] = u_par_mesh[i]
        f[i, :] = Juettner2D(u_par_dummy, u_perp_mesh, Te)
    f[f < 1.e-20] = 1.e-20
    cont1 = ax.contourf(u_perp_mesh, u_par_mesh, np.log10(f), cmap=plt.cm.get_cmap("plasma"))
    ax.set_aspect("equal")
    cb = fig.colorbar(cont1, ax=ax, ticks=[-10, -5, 0, 5])  # ticks = levels[::6] #,
    cb.set_label(r"$\mathrm{Log}_\mathrm{10}(f_\mathrm{MJ})$")
    ax.set_xlabel("$u_\perp$")
    ax.set_ylabel("$u_\parallel$")
    ax.get_xaxis().set_major_locator(MaxNLocator(nbins=3))
    plt.show()

def benchmark_absorption_rel(working_dir):
    a = np.loadtxt(os.path.join(working_dir, "v_high_ne_abs_prof.dat"))
    b = np.loadtxt(os.path.join(working_dir, "high_ne_abs_prof.dat"))
    c = np.loadtxt(os.path.join(working_dir, "low_ne_abs_prof.dat"))
    limit = 1.e-3
    i_1 = 1  # Albajar
    # i_2 = 3  # Gray
    i_2 = 4  # Crude Approx.
    a = a[a.T[0] < 2.005]
    a = a[(a.T[i_1] + a.T[i_2]) > limit]
    b = b[b.T[0] < 2.005]
    b = b[(b.T[i_1] + b.T[i_2]) > limit]
    c = c[c.T[0] < 2.005]
    c = c[(c.T[i_1] + c.T[i_2]) > limit]
    plt.plot(a.T[0][(a.T[i_1] + a.T[i_2]) > limit], (2.0 * np.abs(a.T[i_1] - a.T[i_2]) / (a.T[i_1] + a.T[i_2]))[(a.T[i_1] + a.T[i_2]) > limit], "-r", label=r"$n_\mathrm{e} = \SI{1.0e20}{\per\cubic\meter}$")
    plt.plot(b.T[0][(b.T[i_1] + b.T[i_2]) > limit], (2.0 * np.abs(b.T[i_1] - b.T[i_2]) / (b.T[i_1] + b.T[i_2]))[(b.T[i_1] + b.T[i_2]) > limit], "--c", label=r"$n_\mathrm{e} = \SI{8.0e19}{\per\cubic\meter}$")
    plt.plot(c.T[0][(c.T[i_1] + c.T[i_2]) > limit], (2.0 * np.abs(c.T[i_1] - c.T[i_2]) / (c.T[i_1] + c.T[i_2]))[(c.T[i_1] + c.T[i_2]) > limit], ":m", label=r"$n_\mathrm{e} = \SI{6.0e19}{\per\cubic\meter}$")
    plt.legend()
    plt.gca().set_xlabel(r"$\frac{\omega_\mathrm{c}}{\omega}$")
    plt.gca().set_ylabel(r"$\frac{\delta\alpha_\omega}{\overline{\alpha}}$")
    plt.figure()
    a = np.loadtxt(os.path.join(working_dir, "v_high_ne_abs_prof_low_f.dat"))
    b = np.loadtxt(os.path.join(working_dir, "high_ne_abs_prof_low_f.dat"))
    c = np.loadtxt(os.path.join(working_dir, "low_ne_abs_prof_low_f.dat"))
    limit = 1.e-3
    a = a[a.T[0] < 2.005]
    a = a[(a.T[i_1] + a.T[i_2]) > limit]
    b = b[b.T[0] < 2.005]
    b = b[(b.T[i_1] + b.T[i_2]) > limit]
    c = c[c.T[0] < 2.005]
    c = c[(c.T[i_1] + c.T[i_2]) > limit]
    plt.plot(a.T[0], 2.0 * np.abs(a.T[i_1] - a.T[i_2]) / (a.T[i_1] + a.T[i_2]), "-r", label=r"$n_\mathrm{e} = \SI{6.0e19}{\per\cubic\meter}$")
    plt.plot(b.T[0], 2.0 * np.abs(b.T[i_1] - b.T[i_2]) / (b.T[i_1] + b.T[i_2]), "--c", label=r"$n_\mathrm{e} = \SI{4.0e19}{\per\cubic\meter}$")
    plt.plot(c.T[0], 2.0 * np.abs(c.T[i_1] - c.T[i_2]) / (c.T[i_1] + c.T[i_2]), ":m", label=r"$n_\mathrm{e} = \SI{2.0e19}{\per\cubic\meter}$")
    plt.legend()
    plt.gca().set_xlabel(r"$\frac{\omega_\mathrm{c}}{\omega}$")
    plt.gca().set_ylabel(r"$\frac{\Delta\alpha_\omega}{\overline{\alpha}}$")
    plt.show()

def evaluate_shift(beta, rhop, Te_spline, rhop_spline, R, z):
    rhop_new = rhop_spline(R + beta[0], z + beta[1], grid=False)
    return Te_spline(rhop_new)

def plot_resonance_line(fig, fig2, rpath, rhop_in, ich, shot, time, dstf, rel_res, beta, hfs=False, mode="X"):
    rpath_data = os.path.join(rpath, "ecfm_data")
    gy_path = os.path.join(rpath_data, "gy_data")
    if(dstf == "Re"):
        te_mod_filename = os.path.join(rpath_data, "TRadM_RELAX.dat")
    elif(dstf == "Lu"):
        te_mod_filename = os.path.join(rpath_data, "TRadM_LUKE.dat")
    else:
        te_mod_filename = os.path.join(rpath_data, "TRadM_therm.dat")
    rhop_mod, Te_mod = read_file(te_mod_filename)
    ch_num = len(rhop_mod)
#    if(rel_res):
#        rhop_mod = np.loadtxt(os.path.join(rpath_data, "rhopres_rel.dat")).T[0]
#        rhop_res = rhop_mod[ich - 1]
#    else:
#        rhop_res = rhop_mod[ich - 1]
    svec, ece_freq = read_svec_from_file(rpath_data, ich, mode=mode)
    # ece_freq = 135.e9
    flag_use_ASCII = True
    flag_show_ECRH = False
    flag_show_ECE = True  # True #
    flag_show_j = False
    gy_freq = 140.e9
    gy_mode = +1
    c_lower_lim = 1.e-20
    print("ECE Freq", ece_freq)
    Te_ext = -1  # 8.0e3
    # print("Warning Te HARDCODED")
    try:
        Te_filename = os.path.join(rpath, "te_ida.res")
        rhop_vec_Te, Te_vec = read_file(Te_filename)
        Te_vec = Te_vec * 1.e3
        ne_filename = os.path.join(rpath, "ne_ida.res")
        rhop_vec_ne, ne_vec = read_file(ne_filename)
        ne_vec = ne_vec * 1.e20
    except IOError:
        Te_filename = os.path.join(rpath, "ecfm_data", "Te_file.dat")
        Te_data = np.loadtxt(Te_filename, skiprows=1)
        rhop_vec_Te = Te_data.T[0]
        Te_vec = Te_data.T[1]
        ne_filename = os.path.join(rpath, "ecfm_data", "ne_file.dat")
        ne_data = np.loadtxt(ne_filename, skiprows=1)
        rhop_vec_ne = ne_data.T[0]
        ne_vec = ne_data.T[1]
    Te_spl = InterpolatedUnivariateSpline(rhop_vec_Te, Te_vec)
    ne_spl = InterpolatedUnivariateSpline(rhop_vec_ne, ne_vec)
    Gy_Res = glob(os.path.join(gy_path, "Resonance*.dat"))
    print("Found resonance data for {0:} gyrotron(s)".format(len(Gy_Res)))
    Gy_Pj = glob(os.path.join(gy_path, "P_n_j*.dat"))
    print("Power and current data for {0:} gyrotron(s)".format(len(Gy_Res)))
    GyLabels = []
    # beta = [0.01,500,500,50000,50000,0.0,0.0,1.0, 0.0]
    # beta = np.array([0.0,5000,5000,7000,5000000,0.0,0.0,1.8, 0.0])
    rhop = 0.0
    B0 = 0.0
    Alb = em_abs_Alb()
    if(dstf == "Re" or dstf == "Lu"):
        if(flag_use_ASCII):
            f_folder = os.path.join(rpath, "ecfm_data", "f" + dstf)
            dist_obj = load_f_from_ASCII(f_folder, rhop_in)
            rhop = dist_obj.rhop[0]
            x = dist_obj.u
            y = dist_obj.pitch
            Fe = np.log(dist_obj.f)
            rhop_in = rhop
            Fe = Fe[0]
            # Fe = np.exp(Fe)
            print("Distribution shape", x.shape, y.shape, Fe.shape)
        elif(dstf == "Lu"):
            try:
                ne_filename = os.path.join(rpath, "ne_ida.res")
                rhop_vec_ne, ne = read_file(ne_filename)
                ne = ne * 1.e20
            except IOError:
                ne_filename = os.path.join(rpath, "ecfm_data", "ne_file.dat")
                ne_data = np.loadtxt(ne_filename, skiprows=1)
                rhop_vec_ne = ne_data.T[0]
                ne = ne_data.T[1]
            dist_obj = read_LUKE_data(rpath)
            rhop = dist_obj.rhop
            x = dist_obj.u
            y = dist_obj.rhop
            Fe = dist_obj.f
        else:
            ipsi, psi, x, y, Fe = read_Fe(rpath + "/ecfm_data/")
            irhop = np.argmin(np.abs(rhop - rhop_in))
            rhop_in = rhop[irhop]
            Fe = Fe[irhop]
        rhop_B_min, B_min = get_B_min_from_file(os.path.join(rpath, "ecfm_data"))
        B_min_spline = InterpolatedUnivariateSpline(rhop_B_min, B_min)
        B0 = B_min_spline(rhop_in)
    else:
        x = np.linspace(0.0, 3.0, 200)
        y = np.linspace(0.0, np.pi, 200)
        Fe = np.zeros((len(x), len(y)))
        rhop = rhop_in
        if((dstf != "SH" and dstf != "MS") and beta[0] == 0.0):
            for i in range(len(x)):
                u_perp = np.cos(y) * x[i]
                u_par = np.sin(y) * x[i]
                Fe[i] = Juettner2D(u_par, u_perp, Te_spl(rhop_in))
        elif(dstf != "SH" and dstf != "MS" and dstf != "RA" and beta[0] != 0.0):
            for i in range(len(x)):
                u_perp = np.cos(y) * x[i]
                u_par = np.sin(y) * x[i]
                Fe[i] = (1.e0 - beta[0]) * (Juettner2D(u_par, u_perp, Te_spl(rhop_in)) + \
                                beta[0] * Juettner2D_bidrift(u_par, u_perp, beta[3], beta[4], beta[5], beta[6]))
        elif(dstf == "MS"):
            for i in range(len(x)):
                u_perp = np.cos(y) * x[i]
                u_par = np.sin(y) * x[i]
                Fe[i] = multi_slope(u_par, u_perp, Te_spl(rhop_in), \
                                           beta[1], beta[0] * beta[2])
        elif(dstf == "RA"):
            for i in range(len(x)):
                u_perp = np.cos(y) * x[i]
                u_par = np.sin(y) * x[i]
                Fe[i] = (RunAway2D(u_par, u_perp, Te_spl(rhop_in), ne_spl(rhop_in), \
                                           beta[0], beta[3], beta[4]))
        else:
            print("The chosen dstf is not supported", dstf)
        Fe[Fe < 1.e-30] = 1.e-30
        Fe = np.log(Fe)
#            vd = -j_cur / (cnst.e * ne_cur)
#            f_res[i] = (1.e0 + SpizerHaermDist2D_static(ece_res_arr[i][0], \
#                    ece_res_arr[i][1], vd, Te_cur))
#            f_res[i] *= Juettner2D(ece_res_arr[i][0], \
#                                   ece_res_arr[i][1], Te_cur)
    # irhop = ipsi
        # norm = max(Fe[irhop,:,:].flatten())
        # Fe[irhop,:,:] = Fe[irhop,:,:] / norm
    dist_obj = distribution_interpolator(x, y, RectBivariateSpline(x, y, Fe))
    Alb.dist_mode = "ext"
    Alb.ext_dist = dist_obj
    Alb.B_min = B0
    print("B_min on flux surface:", B0)
    pmax = max(x)
    # print(pmax)
    npts = 400
    ull = np.linspace(-pmax, +pmax, npts)
    uxx = np.linspace(0., pmax, npts)
        # ull = np.linspace(-0.9, 0.9,npts)
        # uxx = np.linspace(0.,0.9, npts)
    # beta = np.array([0.025,5000,5000,500,10000,0.0,0.0,-0.68, 0.0]) #0.02
    if(dstf == "Re"):
        beta[0] = 0.0
        dist = r"RELAX"
        dist_basic = "MJ"
    elif(dstf == "Lu"):
        beta[0] = 0.0
        dist = r"LUKE"
        dist_basic = "MJ"
    elif(dstf == "Mx"):
        beta[0] = 0.0
        dist = r"MJ"
        dist_basic = "M"
    elif(dstf == "DM"):
        dist = r"Drifting MJ"
        dist_basic = "MJ"
        beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
    elif(dstf == "BJ"):
        dist = r"Bi-Maxwell-Juettner"
        dist_basic = "MJ"
        beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
        beta[5] = 0.0
        beta[6] = 0.0
    elif(dstf == "BM"):
        dist = r"Bi-Maxwell"
        dist_basic = "MJ"
        beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
        beta[5] = 0.0
        beta[6] = 0.0
    elif(dstf == "MS"):
        dist = r"Two Slope MJ"
        dist_basic = "MJ"
        beta[0] = Gauss_not_norm(rhop, [1.0, 0.0, beta[0]])
    elif(dstf == "SH"):
        dist = r"Spitzer-Haerm"
        dist_basic = "MJ"
        rhop_vec_j, j_vec, err = get_current(int(shot), float(time), 120)
        if(err != 0):
            print("Error when reading current. Erno: " + err)
            return rhop
        j_cur = search_interpolate(rhop, rhop_vec_j, j_vec)
    elif(dstf == "RA"):
        dist = r"Runaway"
        dist_basic = "MJ"
        beta[0] = Gauss_norm(rhop, [beta[0], beta[1], beta[2]])
    else:
        beta[0] = 0.0
        dist = r"MJ"
        dist_basic = "MJ"
    dist = dstf
    # beta = [0.01,500,500,2.e4,3.e5,0.0,0.0,0.00, 0.0]
    for i in Gy_Res:
        GyLabels.append(i.rsplit(".dat", 1)[0].split("_gy", 1)[1])
    P_gy = []
    j_gy = []
    rhop_vec_P_gy = []
    n_omega_bar_gy = []
    N_par_gy = []
    rhop_vec_N_par_gy = []
    R_gy = []
    z_gy = []
    R_ax, z_ax = get_axis(int(shot), float(time))
    if(len(Gy_Pj) == len(Gy_Res)):
        for i in range(len(Gy_Pj)):
            x_gy, y_gy = read_file(Gy_Pj[i], 0, 1)
            rhop_vec_P_gy.append(x_gy)
            P_gy.append(y_gy)
            x_gy, y_gy = read_file(Gy_Pj[i], 0, 2)
            j_gy.append(y_gy)
            x_gy, y_gy = read_file(Gy_Res[i], 0, 1)
            R_gy.append(x_gy)
            z_gy.append(y_gy)
            x_gy, y_gy = read_file(Gy_Res[i], 0, 2)
            rhop_vec_N_par_gy.append(y_gy)
            x_gy, y_gy = read_file(Gy_Res[i], 0, 3)
            n_omega_bar_gy.append(2.0 * y_gy)
            x_gy, y_gy = read_file(Gy_Res[i], 0, 4)
            N_par_gy.append(-y_gy)
    try:
        if(hfs):
            if(np.all(svec.T[3][svec.T[3] > 0][svec.T[1][svec.T[3] > 0] < R_ax] > rhop) or np.all(svec.T[3][svec.T[3] > 0][svec.T[1][svec.T[3] > 0] < R_ax] < rhop)):
                    raise IndexError
            else:
                rhop_res = svec.T[3][svec.T[1] < R_ax][np.argmin(np.abs(svec.T[3][svec.T[1] < R_ax] - rhop))]
                print("Closest rhopol", rhop_res)
                print("Corresponding index", np.where(rhop_res == svec.T[3][0].item()))
                ece_res_ind = [np.where(rhop_res == svec.T[3])[0].item()]
        else:
            if(np.all(svec.T[3][svec.T[3] > 0][svec.T[1][svec.T[3] > 0] > R_ax] > rhop) or \
               np.all(svec.T[3][svec.T[3] > 0][svec.T[1][svec.T[3] > 0] > R_ax] < rhop)):
                    raise IndexError
            else:
                rhop_res = svec.T[3][svec.T[1] > R_ax][np.argmin(np.abs(svec.T[3][svec.T[1] > R_ax] - rhop))]
                print("Clostest rhop pol", rhop_res)
                print("Corresponding index", np.where(rhop_res == svec.T[3])[0].item())
                ece_res_ind = [np.where(rhop_res == svec.T[3])[0].item()]
        # i_alternate = [find_value_occurence(svec.T[3], 0.104)[-1]][0]
    except IndexError as e:
        print(e)
        print("LOS does not cross flux surface")
        print("Smallest rhop_pol:", np.min(svec.T[3][svec.T[3] > 0]))
        return fig, fig2, rhop
    print("R at resonance" , svec.T[1][ece_res_ind])
    gy_res_ind_arr = []
    try:
        for i in range(len(rhop_vec_N_par_gy)):
            if(hfs):
                if(np.all(rhop_vec_N_par_gy[i][R_gy[i] < R_ax * 100.0] > rhop) or np.all(rhop_vec_N_par_gy[i][R_gy[i] < R_ax * 100.0] < rhop)):
                    raise IndexError
                else:
                    rhop_res = rhop_vec_N_par_gy[i][R_gy[i] < R_ax * 100.0 ][np.argmin(np.abs(rhop_vec_N_par_gy[i][R_gy[i] < R_ax * 100.0] - rhop))]
                    cur_indx = np.where(rhop_vec_N_par_gy[i] == rhop_res)[0].item()
            else:
                if(np.all(rhop_vec_N_par_gy[i][R_gy[i] > R_ax * 100.0] > rhop) or np.all(rhop_vec_N_par_gy[i][R_gy[i] > R_ax * 100.0] < rhop)):
                    raise IndexError
                else:
                    rhop_res = rhop_vec_N_par_gy[i][R_gy[i] > R_ax * 100.0][np.argmin(np.abs(rhop_vec_N_par_gy[i][R_gy[i] > R_ax * 100.0] - rhop))]
                    cur_indx = np.where(rhop_vec_N_par_gy[i] == rhop_res)[0].item()
            gy_res_ind_arr.append([cur_indx])
    except IndexError as e:
        print(e)
        print("Los of gyrtron does not cross flux surface")
        print("Smallest rhop_pol:", np.min(rhop_vec_N_par_gy[i]))
#    if(len(gy_res_ind_arr) == 0):
#        print("Gyrotrons not resonant on flux surface")
#    else:
#        print("gy_res", gy_res_ind_arr)
    temp_gy_res_ind = []
    temp_GyLabels = []
    temp_rhop_vec_N_par_gy = []
    temp_n_omega_bar_gy = []
    temp_N_par_gy = []
    temp_rhop_vec_P_gy = []
    temp_P_gy = []
    temp_j_gy = []
    temp_R_gy = []
    temp_z_gy = []
    for i in range(len(gy_res_ind_arr)):
        if(gy_res_ind_arr[i] != []):
            temp_gy_res_ind.append(gy_res_ind_arr[i])
            temp_GyLabels.append(GyLabels[i])
            temp_rhop_vec_N_par_gy.append()
            temp_n_omega_bar_gy.append(n_omega_bar_gy[i])
            temp_N_par_gy.append(N_par_gy[i])
            temp_rhop_vec_P_gy.append(rhop_vec_P_gy[i])
            temp_P_gy.append(P_gy[i])
            temp_j_gy.append(j_gy[i])
            temp_R_gy.append(R_gy[i])
            temp_z_gy.append(z_gy[i])
    gy_res_ind_arr = temp_gy_res_ind
    GyLabels = temp_GyLabels
    P_gy = temp_P_gy
    j_gy = temp_j_gy
    rhop_vec_P_gy = temp_rhop_vec_P_gy
    rhop_vec_N_par_gy = temp_rhop_vec_N_par_gy
    N_par_gy = -np.array(temp_N_par_gy)
    n_omega_bar_gy = temp_n_omega_bar_gy
    R_gy = temp_R_gy
    z_gy = temp_z_gy
    gy_N_par_mat = []
    gy_n_omega_bar_mat = []
    gy_R_mat = []
    gy_z_mat = []
    last_gy_N_par = 0.0
    svec_gy = []
    for i in range (len(gy_res_ind_arr)):
        gy_N_par_mat.append(np.zeros(len(gy_res_ind_arr[i])))
        gy_n_omega_bar_mat.append(np.zeros(len(gy_res_ind_arr[i])))
        gy_R_mat.append(np.zeros(len(gy_res_ind_arr[i])))
        gy_z_mat.append(np.zeros(len(gy_res_ind_arr[i])))
    for i in range(len(gy_res_ind_arr)):  # loop over the gyrotrons
        svec_gy.append([])
        for j in range(len(gy_res_ind_arr[i])):  # loop over the intersections
                                                # of los and rhop
            gy_n_omega_bar_mat[i][j] = interpolate(rhop, \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j]], \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j] + 1], \
                n_omega_bar_gy[i][gy_res_ind_arr[i][j]], \
                n_omega_bar_gy[i][gy_res_ind_arr[i][j] + 1])
            gy_N_par_mat[i][j] = interpolate(rhop, \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j]], \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j] + 1], \
                N_par_gy[i][gy_res_ind_arr[i][j]], \
                N_par_gy[i][gy_res_ind_arr[i][j] + 1])
            gy_R_mat[i][j] = interpolate(rhop, \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j]], \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j] + 1], \
                R_gy[i][gy_res_ind_arr[i][j]], \
                R_gy[i][gy_res_ind_arr[i][j] + 1])
            gy_z_mat[i][j] = interpolate(rhop, \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j]], \
                rhop_vec_N_par_gy[i][gy_res_ind_arr[i][j] + 1], \
                z_gy[i][gy_res_ind_arr[i][j]], \
                z_gy[i][gy_res_ind_arr[i][j] + 1])
            svec_gy[-1].append(s_vec(rhop_vec_N_par_gy[i], Te_spl(rhop), ne_spl(rhop), \
                                 gy_n_omega_bar_mat[i][j] * gy_freq, np.arccos(gy_N_par_mat[i][j])))  # assumes N = 1
    gy_res_array = []
    for i in range(len(gy_res_ind_arr)):  # loop over the gyrotrons
        gy_res_array.append([])
        for j in range(len(gy_res_ind_arr[i])):  # loop over the intersections
                                                # of los and rhop
            print(GyLabels[i])
            u_par, u_perp = Alb.abs_Albajar_resonance_line(svec, gy_freq, gy_mode, m=2)
            gy_res_array[i].append(np.array([u_par, u_perp]))
            # get_resonance_N(gy_n_omega_bar_mat[i][j], \
            #    gy_N_par_mat[i][j])
            print("n_omega_bar, N_par", gy_n_omega_bar_mat[i][j], \
                gy_N_par_mat[i][j])
            print("R,z", gy_R_mat[i][j], \
                gy_z_mat[i][j])
            # print(gy_res_array[i])
    for i in range(len(gy_res_array)):
        while None in gy_res_array[i]:
            gy_res_array[i].remove(None)
    while [] in gy_res_array:
        gy_res_array.remove([])
    if(len(gy_res_array) > 0):
        theta_gy = gy_N_par_mat[0][-1]
        last_gy_N_par = gy_N_par_mat[0][-1]
    else:
        last_gy_N_par = 0
    n_reso_line = 3
    ece_theta_arr = np.zeros(len(ece_res_ind) * n_reso_line)
    ece_freq_2X_arr = np.zeros(len(ece_res_ind) * n_reso_line)
    R_arr = np.zeros(len(ece_res_ind) * n_reso_line)
    z_arr = np.zeros(len(ece_res_ind) * n_reso_line)
    ece_N_arr = np.zeros(len(ece_res_ind) * n_reso_line)
    ece_res_arr = []
    actual_freq = []
    m = np.zeros(len(ece_res_ind) * n_reso_line, np.int)
    svec_ece = []
    for i in range(0, len(ece_res_ind) * n_reso_line, n_reso_line):
        cur_i = ece_res_ind[i / n_reso_line]
        for j in range(n_reso_line):
            m[i + j] = 2 + j
            ece_theta_arr[i + j] = interpolate(rhop, svec[cur_i][3], svec[cur_i + 1][3] \
                , svec[cur_i][6], svec[cur_i + 1][6])
            ece_N_arr[i + j] = interpolate(rhop, svec[cur_i][3], svec[cur_i + 1][3] \
                , svec[cur_i][7], svec[cur_i + 1][7])
            ece_freq_2X_arr[i] = interpolate(rhop, svec[cur_i][3], svec[cur_i + 1][3] \
                , svec[cur_i][8], svec[cur_i + 1][8])
            R_arr[i + j] = interpolate(rhop, svec[cur_i][3], svec[cur_i + 1][3] \
                , svec[cur_i][1], svec[cur_i + 1][1])
            z_arr[i + j] = interpolate(rhop, svec[cur_i][3], svec[cur_i + 1][3] \
                , svec[cur_i][2], svec[cur_i + 1][2])
            svec_ece.append(s_vec(rhop, Te_spl(rhop), ne_spl(rhop), ece_freq_2X_arr[i], ece_theta_arr[i]))  # for n = 2
            u_par, u_perp = Alb.abs_Albajar_resonance_line(svec_ece[-1], ece_freq * 2.0 * np.pi, -1, 2 + j)  # mode does not matter
#        res = get_resonance_N(ece_freq_2X_arr[i] / ece_freq, \
#            ece_N_arr[i] * np.cos(ece_theta_arr[i]))
            if(Alb.resonant):
                ece_res_arr.append(np.array([u_par, u_perp]))
                actual_freq.append(ece_freq_2X_arr[i + j] * (j + 2) / 2)
                print("ECE")
                print("n_omega_bar,\
                N_par", ece_freq_2X_arr[i + j] / ece_freq, \
                ece_N_arr[i + j] * np.cos(ece_theta_arr[i + j]))
                print("R,z", R_arr[i + j], \
                z_arr[i + j])
                print("omega_c", ece_freq_2X_arr[i] / 2.0)
                # ece_theta_arr[i] = 80.0 / 180.0 * np.pi
                print("theta", ece_theta_arr[i] * 180 / np.pi)
                print("N_par", np.cos(ece_theta_arr[i]) * ece_N_arr[i])
    print("Found {0:} resonance curves for ECE".format(len(ece_res_arr)))
    # theta_ECE =  ece_theta_arr[0]
    if(len(ece_res_arr) == 0):
        print("no ECE resonance\n")
        return fig, fig2, rhop
    cur_i = ece_res_ind[0]  # rhop change
    # cur_i = i_alternate
    Te_cur = interpolate(rhop , svec[cur_i][3], svec[cur_i + 1][3] \
            , svec[cur_i][5], svec[cur_i + 1][5])
    if(not np.isscalar(Te_cur)):
        Te_cur = Te_cur[0]
    ne_cur = interpolate(rhop, svec[cur_i][3], svec[cur_i + 1][3] \
            , svec[cur_i][4], svec[cur_i + 1][4])
    if(not np.isscalar(ne_cur)):
        ne_cur = ne_cur[0]
    if(Te_ext > 0):
        Te_cur = Te_ext
#    levels = np.zeros(50)
    # levels = np.arange(0,1, .01)
    # for i in range(len(levels)):
    #    if(levels[i] < 0.0):
    #        levels[i] = - levels[i] **2
    #    else:
    #        levels[i] = levels[i] **2
    # levels[47:62] = np.arange(5,20, 1)
    # levels[16:24] = np.arange(2,5, 0.4)
    # levels = np.exp(levels)

    # ECRH_res_array = []
    min_ull_res = -0.5
    min_uxx_res = 0.0
    max_ull_res = 0.5
    max_uxx_res = 0.5
    grid = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(grid[0])
    # ax1.get_xaxis().set_major_locator(MaxNLocator(prune='lower'))
    if(flag_show_j):
        grid2 = gridspec.GridSpec(3, 1)
    else:
        grid2 = gridspec.GridSpec(1, 2)
    ax2 = fig2.add_subplot(grid2[0])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
    # plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax3 = fig2.add_subplot(grid2[1], sharex=ax2)
    ax3_twin = ax3.twinx()
    if(flag_show_j):
        ax4 = fig2.add_subplot(grid2[2])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
    steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    steps_y = steps
    f_res = []
    sf_res = []
    sf_impact = []
    f_therm_res = []
    sf_therm_res = []
    gy_label = ""
    if(flag_show_ECRH):
        for i in range(len(gy_res_array)):  # loop over the gyrotrons
            for j in range(len(gy_res_array[i])):  # loop over the intersections
                                                # of los and rhop
                if(j >= len(gy_res_array[i]) / 2.0):
                    gy_label = r"Gyrotron {0:n} resonance - ".format(i + 1) + \
                        r"$f_\mathrm{ECRH} = 140$ GHz" + \
                                r", $N_\parallel" + r"=$ {0:1.2f}".format(theta_gy)
                else:
                    gy_label = r"Gyrotron {0:n} resonance - ".format(i + 1) + \
                        r"$f_\mathrm{ECRH} = 140$ GHz" + \
                                 r", $N_\parallel" + r"=$ {0:1.2f}".format(theta_gy)
                        # + GyLabels[i] +
                ax1.plot(gy_res_array[i][j][0], gy_res_array[i][j][1], "--k", label=gy_label)
                # i_02 = find_index(gy_res_array[i][j][0],0.2)
                # ax4.vlines(gy_res_array[i][j][1][i_02],-1e5,1e5, color = "r",label = gy_label)
            P_gy_label = r"Gyrotron " + GyLabels[i] + \
                        " Power deopsition"
            j_gy_label = r"Gyrotron " + GyLabels[i] + \
                        " driven current density"
            # ax3.plot(rhop_vec_P_gy[i],P_gy[i], label = P_gy_label)
            # ax3.plot(rhop_vec_P_gy[i],j_gy[i], label = j_gy_label)
    max_sf = 0.0
    best_f = 0.0
    marker_list = ["-k", "--b", ":g", "-.r"]
    max_impact = 0.0
    max_Trad = 0.0
    for i in range(len(ece_res_arr)):
        if(min_ull_res > min(ece_res_arr[i][0])):
            min_ull_res = min(ece_res_arr[i][0])
        if(min_uxx_res > min(ece_res_arr[i][1])):
            min_uxx_res = min(ece_res_arr[i][1])
        if(max_ull_res < max(ece_res_arr[i][0])):
            max_ull_res = max(ece_res_arr[i][0])
        if(max_uxx_res < max(ece_res_arr[i][1])):
            max_uxx_res = max(ece_res_arr[i][1])
        label = r"ECE resonance - $f_\mathrm{ECE} \approx " + " $ {0:3.1f} GHz".format(int(ece_freq * 1.e-9)) + \
                r", $N_\parallel" + r" = $ {0:1.2f} ".format(np.cos(ece_theta_arr[i]) * ece_N_arr[i]) + \
                " {0:d}.".format(m[i])  # 2X lfs, cold resonance at $\rho = {0:1.3f}$".format(rhop_res)
        label_f_res = r"$f_\mathrm{" + dist + "}$" + " {0:d}.".format(m[i])
        label_sf_res = r"$\alpha_\mathrm{" + dist + "}$" + " {0:d}.".format(m[i])
        label_f_therm_res = r"$f_\mathrm{" + dist_basic + "}$" + " {0:d}. ".format(m[i])
        label_sf_therm_res = r"$sf_\mathrm{" + dist_basic + "}$" + " {0:d}. ".format(m[i])
        if(i < len(marker_list)):
            marker = marker_list[i]
        else:
            marker = marker_list[0]
        if(flag_show_ECE):
            ax1.plot(ece_res_arr[i][0], ece_res_arr[i][1], marker, label=label)
        # ax2.plot(ece_res_arr[i][0],ece_res_arr[i][1], label = label)
        f_res.append(np.zeros(len(ece_res_arr[i][0])))
        sf_res.append(np.zeros(len(ece_res_arr[i][0])))
        sf_impact.append(np.zeros(len(ece_res_arr[i][0])))
        f_therm_res.append(np.zeros(len(ece_res_arr[i][0])))
        sf_therm_res.append(np.zeros(len(ece_res_arr[i][0])))
#        if(dstf != "Th" and (beta[0] != 0 or dstf == "MS" or dstf == "Lu" or dstf == "Re")):
#            Alb.dist_mode = "ext"
#        else:
#            Alb.dist_mode = "thermal"
        Alb.dist_mode = "ext"
        f_res[i] = Alb.dist(ece_res_arr[i][0], \
            ece_res_arr[i][1], (cnst.m_e * cnst.c ** 2) / (svec_ece[i].Te * cnst.e)  , svec_ece[i])
        f_res[i] = np.log10(f_res[i])
        if(dstf == "Mx"):
            f_therm_res[i] = Maxwell2D(ece_res_arr[i][0], \
                                                 ece_res_arr[i][1], Te_cur)
        else:
            Alb.dist_mode = "thermal"
            for j in range(len(f_therm_res[i])):
                f_therm_res[i][j] = Alb.dist(ece_res_arr[i][0][j], \
                    ece_res_arr[i][1][j], (cnst.m_e * cnst.c ** 2) / (svec_ece[i].Te * cnst.e)  , svec_ece[i])
#            f_therm_res[i] = Juettner2D(ece_res_arr[i][0], \
#                                                 ece_res_arr[i][1], Te_cur)
            f_therm_res[i] = np.log10(f_therm_res[i])
            Alb.dist_mode = "ext"
            # print(Te_cur)
#        if(dstf != "Th" and (beta[0] != 0 or dstf == "MS" or dstf == "Lu" or dstf == "Re")):
        if(mode == "X"):
            u_par, u_perp, c_abs, j = Alb.abs_Albajar_along_res(svec_ece[i], ece_freq * 2.0 * np.pi, +1, m=m[i])
        else:
            u_par, u_perp, c_abs, j = Alb.abs_Albajar_along_res(svec_ece[i], ece_freq * 2.0 * np.pi, -1, m=m[i])
        Alb.dist_mode = "thermal"
        if(mode == "X"):
            u_par_th, u_perp, c_abs_th, j_th = Alb.abs_Albajar_along_res(svec_ece[i], ece_freq * 2.0 * np.pi, +1, m=m[i])
        else:
            u_par_th, u_perp, c_abs_th, j_th = Alb.abs_Albajar_along_res(svec_ece[i], ece_freq * 2.0 * np.pi, -1, m=m[i])
        Alb.dist_mode = "ext"
        if(not np.isscalar(c_abs_th) and not np.isscalar(c_abs)):
            if(np.any(c_abs[c_abs_th != 0] <= 0)):
                print("Found positive gradients in the distribution")
                print("c_abs", c_abs[c_abs[c_abs_th != 0] <= 0])
                print("u_par", u_par[c_abs[c_abs_th != 0] <= 0])
                print("u_perp", u_perp[c_abs[c_abs_th != 0] <= 0])
            print(label_sf_res, simps(c_abs, u_par))
            print(label_sf_therm_res, simps(sf_therm_res[i], u_par_th))
            sf_res[i][c_abs > c_lower_lim] = j[c_abs > c_lower_lim] / c_abs[c_abs > c_lower_lim] * cnst.c ** 2 / ece_freq ** 2 / cnst.e
            sf_res[i][c_abs <= c_lower_lim] = 0.0
            sf_impact[i] = c_abs * 5.e-3
            if(max_Trad < np.max(sf_res[i])):
                max_Trad = np.max(sf_res[i])
            if(max_impact < np.max(sf_impact[i])):
                max_impact = np.max(sf_impact[i])
            # j * 5.e-3 * cnst.c ** 2 / ece_freq ** 2 / cnst.e / Te_cur  # 5 mm thickness of the flux surface
            sf_therm_res[i][c_abs_th > c_lower_lim] = j_th[c_abs_th > c_lower_lim] / c_abs_th[c_abs_th > c_lower_lim] * cnst.c ** 2 / ece_freq ** 2 / cnst.e
            sf_therm_res[i][c_abs_th <= c_lower_lim] = 0.0
            ax2.plot(ece_res_arr[i][0], f_res[i], label=label_f_res)
    #        print("i", i)
    #        print("res", ece_res_arr[i][0][0], ece_res_arr[i][0][-1])
    #        print("f therm", f_therm_res[i])
    #        print("Te", svec_ece[i].Te)
            ax2.plot(ece_res_arr[i][0], f_therm_res[i], "--", label=label_f_therm_res)
    #        ax3.plot(u_par, np.log10(sf_res[i]), label=label_sf_res)
    #        ax3.plot(u_par_th, np.log10(sf_therm_res[i]), label=label_sf_therm_res)
            ax3.plot(u_par, sf_res[i] / 1.e3, label=label_sf_res)  # eV -> keV
            ax3_twin.plot(u_par, sf_impact[i], "--")
        # ax3.plot(u_par_th, sf_therm_res[i], label=label_sf_therm_res)
    # step_ll = (max_ull_res-  min_ull_res) / 400.0
    # zstep_xx = (max_uxx_res-  min_uxx_res) / 400.0

    # ull = np.arange(min_ull_res,max_ull_res,step_ll)
    # print(min_ull_res,max_ull_res)
    # uxx = np.arange(min_uxx_res,max_uxx_res,step_xx)

    # ull, uxx = np.meshgrid(ull, uxx)
    f = np.zeros([len(ull), len(uxx)])
    # f_loc =np.zeros(len(uxx))
    # i_02 = find_index(ece_res_arr[i][0],0.2)
    #    if(i_02 is None):
    #        ax4.vlines(ece_res_arr[i][1][i_02],-1e5,1e5, color = "b", label = label)
    #    ax4.plot(uxx, f_loc,label = r"Distribution function at $u_\Vert = 0.2$" )
    #    ax4.set_ylim(-9,f_loc[0])
    #    ax4.plot(uxx, f_loc,label = r"Distribution function at $u_\Vert = 0.2$" )
    #    ax4.set_ylim(-9,f_loc[0])
    # print(ece_freq_2X_arr)
    if(dstf == "Re"):
        f = remap_f_Maj_single(x, y, Fe, ull, uxx, f, min(ece_freq_2X_arr), B0, False)
    elif(dstf == "Lu"):
        f = remap_f_Maj_single(x, y, Fe, ull, uxx, f, min(ece_freq_2X_arr), B0, False)
    else:
        # for k in range(len(ull)):
        #    for j in range(len(uxx)):
        #        f[k,j] =np.log10(Juettner2D(ull[k], uxx[j], Te_cur))
        if(dstf != "SH" and dstf != "MS" and beta[0] == 0.0):
            for k in range(len(ull)):
                for j in range(len(uxx)):
                    f[k, j] = Juettner2D(ull[k] , uxx[j], Te_cur)  # _beta
                    f[k, j] = np.log(f[k, j])
        elif(dstf != "SH" and dstf != "MS" and dstf != "RA" and beta[0] != 0.0):
            for k in range(len(ull)):
                for j in range(len(uxx)):
                    f[k, j] = (Juettner2D(ull[k], uxx[j], Te_cur) + \
                        beta[0] * Juettner2D_bidrift(ull[k], uxx[j], \
                        beta[3], beta[4], beta[5], beta[6]))
                    f[k, j] = np.log(f[k, j])
        elif(dstf == "MS"):
            for k in range(len(ull)):
                for j in range(len(uxx)):
                    f[k, j] = multi_slope(ull[k], uxx[j], Te_cur, \
                                               beta[1], beta[0] * beta[2])
                    f[k, j] = np.log(f[k, j])
        elif(dstf == "RA"):
            for k in range(len(ull)):
                for j in range(len(uxx)):
                    f[k, j] = (RunAway2D(ull[k], uxx[j], Te_cur, ne_cur, \
                                               beta[0], beta[3], beta[4]))
                    f[k, j] = np.log(f[k, j])
        else:
            vd = -j_cur / (cnst.e * ne_cur)
            for k in range(len(ull)):
                for j in range(len(uxx)):
                    f[k, j] = (1.e0 + SpizerHaermDist2D_static(ull[k], uxx[j], vd, Te_cur))
                    f[k, j] *= Juettner2D(ull[k], uxx[j], Te_cur)
                    f[k, j] = np.log(f[k, j])
    if(dstf == "Re" or dstf == "Lu" or dstf == "MS"and flag_show_j):
        j_tot = 0.e0
        f_circ = np.zeros(200)
        for u_circ in np.linspace(0.3, 1.5, 50):
            u_par_circ = np.linspace(-u_circ, u_circ, 200)
            root = u_circ ** 2 - u_par_circ ** 2
            if(root[0] < 0.e0):
                root[0] = 0.e0
            if(root[-1] < 0.e0):
                root[-1] = 0.e0
            u_perp_circ = np.sqrt(root)
            if(dstf == "Re"):
                f_circ = remap_f_Maj_res_single(x, y, Fe, u_par_circ, u_perp_circ, f_circ, min(ece_freq_2X_arr), B0, False)
            elif(dstf == "Lu"):
                f_circ = remap_f_Maj_res_single(x, y, Fe, u_par_circ, u_perp_circ, f_circ, min(ece_freq_2X_arr), B0, False)
            else:
                for k in range(len(u_par_circ)):
                    f_circ[k] = np.log10(multi_slope(u_par_circ[k], \
                                               u_perp_circ[k], Te_cur, \
                                               beta[1], beta[0] * beta[2]))
            # f_int_circ = simps(f_circ * 2.e0 * np.pi * u_circ,u_par_circ)
            # print("f_tot",f_int_circ)
            j_tot = j_tot + 0.02 * simps(np.exp(f_circ) * u_par_circ / np.sqrt(1.e0 + u_par_circ ** 2 + u_perp_circ ** 2) * \
                            2.e0 * np.pi * u_circ, u_par_circ) * ne_cur * cnst.e * cnst.c
        print("u_par_drif", j_tot / (ne_cur * cnst.e * cnst.c))
        print("Current density", j_tot)
        j_circ = 0.e0
        u_circ = 0.3
        u_par_circ = np.linspace(-u_circ, u_circ, 200)
        f_circ = np.zeros([len(u_par_circ)])
        u_perp_circ = np.sqrt(u_circ ** 2 - u_par_circ ** 2)
        j_circ_arr = [0.0]
        u_circ_arr = [0.0]
        while (not (np.abs(j_circ) > np.abs(j_tot) * 0.8 and np.sign(j_circ) == np.sign(j_tot))) and u_circ < 3.0:
            # print(np.abs(j_circ)/ np.abs(tot_j) * 0.8 )
            u_par_circ = np.linspace(-u_circ, u_circ, 200)
            root = u_circ ** 2 - u_par_circ ** 2
            if(root[0] < 0.e0):
                root[0] = 0.e0
            if(root[-1] < 0.e0):
                root[-1] = 0.e0
            u_perp_circ = np.sqrt(root)
            if(dstf == "Re"):
                f_circ = remap_f_Maj_res_single(x, y, Fe, u_par_circ, u_perp_circ, f_circ, min(ece_freq_2X_arr), B0, False)
            elif(dstf == "Lu"):
                f_circ = remap_f_Maj_res_single(x, y, Fe, u_par_circ, u_perp_circ, f_circ, min(ece_freq_2X_arr), B0, False)
            else:
                for k in range(len(u_par_circ)):
                    f_circ[k] = np.log10(multi_slope(u_par_circ[k], \
                                               u_perp_circ[k], Te_cur, \
                                               beta[1], beta[0] * beta[2]))
            # f_int_circ = simps(f_circ * 2.e0 * np.pi * u_circ,u_par_circ)
            # print("f_tot",f_int_circ)
            dj_circ = simps(np.exp(f_circ) * u_par_circ / np.sqrt(1.e0 + u_par_circ ** 2 + u_perp_circ ** 2) * \
                            2.e0 * np.pi * u_circ, u_par_circ) * ne_cur * cnst.e * cnst.c
            # print("j_circ",j_circ)
            j_circ += dj_circ * 0.05
            j_circ_arr.append(dj_circ)
            u_circ_arr.append(u_circ)
            u_circ += 0.05
        # print("u",u_circ)
        if(flag_show_j):
            ax1.plot(u_par_circ, u_perp_circ, "--k", label="Enclosed electrons carry 80% of total current density")
            ax4.plot(u_circ_arr, np.array(j_circ_arr) / max(j_circ_arr), label="Current density")
        j_circ = 0.e0
        u_circ = 0.3
        u_par_circ = np.linspace(-u_circ, u_circ, 200)
        f_circ = np.zeros([len(u_par_circ)])
        u_perp_circ = np.sqrt(u_circ ** 2 - u_par_circ ** 2)
        j_circ_arr = [0.0]
        u_circ_arr = [0.0]
        while (not (np.abs(j_circ) > np.abs(j_tot) * 0.05 and np.sign(j_circ) == np.sign(j_tot))) and u_circ < 3.0:
            u_par_circ = np.linspace(-u_circ, u_circ, 200)
            root = u_circ ** 2 - u_par_circ ** 2
            if(root[0] < 0.e0):
                root[0] = 0.e0
            if(root[-1] < 0.e0):
                root[-1] = 0.e0
            u_perp_circ = np.sqrt(root)
            if(dstf == "Re"):
                _circ = remap_f_Maj_res_single(x, y, Fe, u_par_circ, u_perp_circ, f_circ, min(ece_freq_2X_arr), B0, False)
            elif(dstf == "Lu"):
                f_circ = remap_f_Maj_res_single(x, y, Fe, u_par_circ, u_perp_circ, f_circ, min(ece_freq_2X_arr), B0, False)
            else:
                for k in range(len(u_par_circ)):
                    f_circ[k] = f_circ[k] = multi_slope(u_par_circ[k], \
                                               u_perp_circ[k], Te_cur, \
                                               beta[1], beta[0] * beta[2])
            # f_int_circ = simps(f_circ * 2.e0 * np.pi * u_circ,u_par_circ)
            # print("f_tot",f_int_circ)
            dj_circ = simps(np.exp(f_circ) * u_par_circ / np.sqrt(1.e0 + u_par_circ ** 2 + u_perp_circ ** 2) * \
                            2.e0 * np.pi * u_circ, u_par_circ) * ne_cur * cnst.e * cnst.c
            # print("j_circ",j_circ)
            j_circ += dj_circ * 0.05
            j_circ_arr.append(dj_circ)
            u_circ_arr.append(u_circ)
            u_circ += 0.05

        # print("u",u_circ)

        if(flag_show_j):
            ax1.plot(u_par_circ, u_perp_circ, ":k", label="Enclosed electrons carry 5% of total current density")
    if(flag_show_j):
        ax4.set_xlabel(r"$u$")
        ax4.set_ylabel(r"$j / \mathrm{a.u.}$")
    # cmap = plt.cm.get_cmap("Greys")
    # cmap.set_under("magenta")
    # cmap.set_over("yellow")
    levels = np.linspace(-13, 5, 60)
    cmap = plt.cm.get_cmap("gnuplot")
    cont1 = ax1.contourf(ull, uxx, f.T / np.log(10), levels=levels, cmap=cmap)  # ,norm = LogNorm()
    cont2 = ax1.contour(ull, uxx, f.T / np.log(10), levels=levels, colors='k',
                        hold='on', alpha=0.25, linewidths=1)
    for c in cont2.collections:
        c.set_linestyle('solid')
    cb = fig.colorbar(cont1, ax=ax1, ticks=[-10, -5, 0, 5])  # ticks = levels[::6] #,
    cb.set_label(r"$\mathrm{Log}_\mathrm{10}\left(f_\mathrm{" + dist + r"}\right)$")  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
    # cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
    cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([1.0, 5.0, 10.0])))
    cb.ax.minorticks_on()
    ax1.get_xaxis().set_major_locator(MaxNLocator(nbins=4, steps=steps, prune='lower'))
    ax1.get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 4.0))
    ax1.get_yaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps_y))
    ax1.get_yaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps_y / 4.0))
    # cb.set_label(r"$\mathrm{Log}_\mathrm{10}(f_\mathrm{" + dist + r"}(\rho_\mathrm{pol}=" + "{:1.2}".format(rhop) + r",u_\perp,u_\Vert))$")
    # cb.set_label(r"$\mathrm{Log}_\mathrm{10}(f_\mathrm{" + dist + r"}(\beta_\perp,\beta_\Vert)$)")
    # plt.clabel(cont2, fmt = "%1.2e")
    if(flag_show_ECE or flag_show_ECRH):
        handles, labels = ax1.get_legend_handles_labels()
        leg = ax1.legend(handles, labels, loc="best", fontsize=22)  # ,         loc = "upper left"
    handles, labels = ax2.get_legend_handles_labels()
    leg = ax2.legend(handles, labels, loc="best")
    handles, labels = ax3.get_legend_handles_labels()
    leg = ax3.legend(handles, labels, loc="best")
    leg.get_frame().set_alpha(0.5)
    ax1.set_xlabel(r"$u_{\Vert}$")
    ax1.set_ylabel(r"$u_{\perp}$")
    # ax1.set_ylabel(r"$\beta_{\perp}$")
    # ax1.set_xlabel(r"$\beta_{\Vert}$")
    #
    # ax1.set_ylabel(r"$\beta_{\perp}$")
    # ax1.get_xaxis().set_major_locator(MaxNLocator(prune='lower'))
    ax2.set_ylabel(r"$u_{\perp}$")
    ax2.set_xlabel(r"$u_{\Vert}$")
    ax3.set_xlabel(r"$u_{\Vert}$")
    ax2.set_ylabel(r"$log_{10}(f)$")
    # ax3.set_ylabel(r"$log_{10}(\alpha)$ / $\mathrm{au}$")
    ax3.set_ylabel(r"$T_\mathrm{ibb, non-therm} \, [\si{\kilo\electronvolt}]$")
    ax3.set_ylim(0.0, max_Trad * 1.005e-3)
    ax3_twin.set_ylabel(r"$\tau_{\mathrm{d}\rho}$")
    ax3_twin.set_ylim(0.0, max_impact * 1.005)
    # ax1.set_xlim(min_ull_res,max_ull_res)
    # ax1.set_ylim(min_uxx_res,max_uxx_res)
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    # ax3.set_xlabel(r"$\rho_{pol}$")
    # ax3.set_ylabel(r"$\frac{\partial P}{\partial s}/j / MW / MA$")
    grid.tight_layout(fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                     rect=[0.05, 0.05, 0.95, 0.9])
    grid2.tight_layout(fig2, pad=0.0, h_pad=1.0, w_pad=1.0,
                     rect=[0.05, 0.05, 0.95, 0.9])
    # fig.suptitle(r"Distribution function for $\#$" + shot + r" $t$ = " + time + r" s @ $R \approx " + \
    #             r"{0:1.1f}$ m".format(R_arr[i]))#") #
    print(Te_cur)
    print(ece_freq_2X_arr, np.min(ece_freq_2X_arr))
    fig.suptitle(r"$T_\mathrm{e}= $ \SI{" + "{0:2.2f}".format(Te_cur * 1.e-3)\
                  + "}{\kilo\electronvolt}, $2\cdot f_\mathrm{c,0} = $ \SI{" + "{0:3.0f}".format(np.min(ece_freq_2X_arr) * 1.e-9) + \
                  "}{\giga\hertz}")  #  \& ECRH    s
    # @ $\rho_\mathrm{pol} = " + \ r"{0:1.3f}$".format(rhop))#") #

    #    fig.suptitle("Rhop = {0:1.3f}".format(rhop))
    return fig, fig2, rhop

def make_iso_flux(folder, shot, time, eq_diag="EQH", eq_experiment="AUGD", eq_ed=0, bt_vac_correction=1.005):
    EQ_obj = EQData(int(shot), EQ_exp=eq_experiment, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
    EQ_slice = EQ_obj.GetSlice(float(time))
    rhop_spline = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.rhop)
    beta = np.zeros(2)
    beta[0] = 0.03
    te_stuff = np.loadtxt(os.path.join(folder, "Te_file.dat"), skiprows=1)
    rhop_Te = te_stuff.T[0]
    Te = te_stuff.T[1]
    ECE_Diag = Diag("ECE", "AUGD", "RMD", 0)
    std_vec_ECE, ECE_data = get_data_calib(ECE_Diag, shot=shot, time=time, eq_exp=eq_experiment, eq_diag=eq_diag, eq_ed=eq_ed)
    resonances = np.loadtxt(os.path.join(folder, "sres.dat"))
    rhop_res = resonances.T[-1]
    R = resonances.T[1][rhop_res < 0.3]
    z = resonances.T[2][rhop_res < 0.3]
    Trad = ECE_data[1][rhop_res < 0.3] * 1.e3
    rhop_Trad = rhop_res[rhop_res < 0.3]
    ifixb = np.array([1, 1])
    TeSpline = InterpolatedUnivariateSpline(rhop_Te, Te)
#    i_Rmin = np.argmin(R)
#    h = 1.e-3
#    dR_drhop = (-rhop_spline(R[i_Rmin] + 2.e0 * h, z[i_Rmin], grid=False) + 8.e0 * rhop_spline(R[i_Rmin] + h, z[i_Rmin], grid=False) - \
#                            8.e0 * rhop_spline(R[i_Rmin] - h, z[i_Rmin], grid=False) + rhop_spline(R[i_Rmin] - 2.e0 * h, z[i_Rmin], grid=False)) / (12.e0 * h)
#    print(R[i_Rmin], dR_drhop)
#    rhop_new = rhop_spline(R - 2.5e-2, z, grid=False)
    plt.plot(rhop_Trad, Trad, "+")
    plt.plot(rhop_Trad, TeSpline(rhop_Trad), "^")
    plt.show()
    Te_data = odr.Data(rhop_Trad, Trad)
    eval_shift = odr.Model(evaluate_shift, extra_args=[TeSpline, rhop_spline, R, z])
    ODR_Fit = odr.ODR(Te_data, eval_shift, beta, iprint=1111, ifixb=ifixb)
    ODR_Fit.set_job(fit_type=2)
    results = ODR_Fit.run()
    print(results.beta)
    rhop_new = rhop_spline(R + results.beta[0], z + results.beta[1], grid=False)
    plt.plot(rhop_new, Trad, "+")
    plt.plot(rhop_new, TeSpline(rhop_new), "^")
    plt.show()
    print("shift complete")


def calculate_resonant_fraction(folder, channelno, width, Te_scale=1.0):
    svec, freq = read_svec_dict_from_file(folder, channelno)
    freq2X_spl = InterpolatedUnivariateSpline(svec["s"], svec["freq_2X"])
    Te_spl = InterpolatedUnivariateSpline(svec["s"], svec["Te"] * Te_scale)
    sres = np.loadtxt(os.path.join(folder, "sres.dat"))[channelno - 1][0]
    s_range = np.linspace(sres - width, sres - 1.e-6, 1000)
    frac = []
    for s in s_range:
        u = np.sqrt((freq2X_spl(s) / freq) ** 2 - 1.0)
        frac.append(2.0 * u ** 3 * 4.0 * np.pi / np.sqrt(1.0 + u ** 2) * Juettner1D(u, Te_spl(s)))
    total = []
    s_range_tot = np.linspace(0.0, sres - 1.e-6, 10000)
    for s in s_range_tot:
        u = np.sqrt((freq2X_spl(s) / freq) ** 2 - 1.0)
        total.append(2.0 * u ** 3 * 4.0 * np.pi / np.sqrt(1.0 + u ** 2) * Juettner1D(u, Te_spl(s)))
    plt.plot(s_range_tot, np.array(total))
    plt.plot(s_range, np.array(frac))
    print(simps(np.array(frac), s_range) / simps(np.array(total), s_range_tot))



def Convolute_Te_perp_Te_par(working_dir, ch, shot, time, eq_exp="AUGD", eq_diag="EQH", eq_ed=0):
    rhop_Gene, R_Gene, z_Gene, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(working_dir, shot, time, eq_exp=eq_exp, eq_diag=eq_diag, eq_ed=eq_ed)
    Te_perp, Te_par = get_dist_moments_non_rel(rhop_Gene, beta_par, mu_norm, f, Te, ne, B0, slices=1)
    filename_BPD = os.path.join(working_dir, "ecfm_data", "IchGe", "BPD_ray001ch" + "{0:0>3}_X.dat".format(ch))
    BPD_data = np.loadtxt(filename_BPD)
    Te_data = np.loadtxt(os.path.join(working_dir, "ecfm_data", "Te_file.dat"), skiprows=1)
    Te_spl = InterpolatedUnivariateSpline(Te_data.T[0], Te_data.T[1])
    Te_perp_spl = InterpolatedUnivariateSpline(rhop_Gene, Te_perp)
    Te_par_spl = InterpolatedUnivariateSpline(rhop_Gene, Te_par)
    rhop_max_Gene = np.max(np.abs(rhop_Gene))
    rhop_min_Gene = np.min(np.abs(rhop_Gene))
    rhop_BPD = BPD_data.T[4]
    rhop_BPD = rhop_BPD[rhop_BPD != -1]
    s = BPD_data.T[0][rhop_BPD != -1]
    Te_perp_BPD = np.zeros(len(rhop_BPD))
    Te_par_BPD = np.zeros(len(rhop_BPD))
    Te_los = np.zeros(len(rhop_BPD))
    D = BPD_data.T[5][rhop_BPD != -1]
    Trad = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_GENE.dat")).T[1][ch - 1]
    Trad_comp = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_therm.dat")).T[1][ch - 1]
    tau_comp = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_therm.dat")).T[2][ch - 1]
    for i in range(len(rhop_BPD)):
        if(rhop_BPD[i] > rhop_min_Gene and rhop_BPD[i] < rhop_max_Gene):
            Te_perp_BPD[i] = Te_perp_spl(rhop_BPD[i])
            Te_par_BPD[i] = Te_par_spl(rhop_BPD[i])
            Te_los[i] = Te
        else:
            Te_perp_BPD[i] = Te_spl(rhop_BPD[i])
            Te_par_BPD[i] = Te_spl(rhop_BPD[i])
            Te_los[i] = Te_spl(rhop_BPD[i])
    plt.plot(rhop_BPD, Te_los)
    plt.show()
    plt.plot(rhop_Gene, (Te_perp / Te - 1.0) * 1.e2, "r", label=r"$\tilde{T}_{\mathrm{e},\perp} / T_\mathrm{e} - 1$")
    ax1 = plt.gca()
    ax1.plot(rhop_Gene, (Te_par / Te - 1.0) * 1.e2, "b", label=r"$\tilde{T}_{\mathrm{e},\parallel} / T_\mathrm{e} - 1$")
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"$\tilde{T}_\mathrm{e} / T_\mathrm{e} - 1$\,[\%]")
    ax2 = ax1.twinx()
    ax2.plot(rhop_BPD, D, "g", label=r"$D_\omega$")
    ax2.set_ylabel(r"$D_\omega$")
    D_norm_spl = InterpolatedUnivariateSpline(s, D)
    D_norm = D_norm_spl.integral(s[0], s[-1])
    Te_D_spl = InterpolatedUnivariateSpline(s, D * Te_los / D_norm)
    Te_rad = Te_D_spl.integral(s[0], s[-1])
    Te_perp_D_spl = InterpolatedUnivariateSpline(s, D * Te_perp_BPD / D_norm)
    Te_perp_rad = Te_perp_D_spl.integral(s[0], s[-1])
    Te_par_D_spl = InterpolatedUnivariateSpline(s, D * Te_par_BPD / D_norm)
    Te_par_rad = Te_par_D_spl.integral(s[0], s[-1])
    print("Normalized Te_perp fluctuation [%]", 2.e2 * (Te_perp_rad - Te_rad) / (Te_perp_rad + Te_rad))
    print("Normalized Te_par fluctuation [%]", 2.e2 * (Te_par_rad - Te_rad) / (Te_par_rad + Te_rad))
    print("Trad background from integration of BPD", Te_rad)
    print("Trad background from radiation Transport", Trad_comp * 1.e3)
    print("Trad pertubation from radiation Transport [%]", 2.e2 * (Trad - Trad_comp) / (Trad + Trad_comp))
    lns = ax1.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs)
    plt.show()


def Check_Trad_vs_Te_BPD(working_dir, ch, shot, time, eq_exp="AUGD", eq_diag="EQH", eq_ed=0):
    filename_BPD = os.path.join(working_dir, "ecfm_data", "IchTB", "BPD_ray001ch" + "{0:0>3}_X.dat".format(ch))
    BPD_data = np.loadtxt(filename_BPD)
    Te_data = np.loadtxt(os.path.join(working_dir, "ecfm_data", "Te_file.dat"), skiprows=1)
    Te_spl = InterpolatedUnivariateSpline(Te_data.T[0], Te_data.T[1])
    rhop_BPD = BPD_data.T[4]
    rhop_BPD = rhop_BPD[rhop_BPD != -1]
    s = BPD_data.T[0][rhop_BPD != -1]
    D = BPD_data.T[5][rhop_BPD != -1]
    Trad_comp = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_therm.dat")).T[1][ch - 1]
    tau_comp = np.loadtxt(os.path.join(working_dir, "ecfm_data", "TRadM_therm.dat")).T[2][ch - 1]
    Te_los = Te_spl(rhop_BPD)
    plt.plot(rhop_BPD, Te_los, label=r"$T_\mathrm{e}$")
    ax1 = plt.gca()
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"$T_\mathrm{e}$")
    ax2 = ax1.twinx()
    ax2.plot(rhop_BPD, D, "g", label=r"$D_\omega$")
    ax2.set_ylabel(r"$D_\omega$")
    D_norm_spl = InterpolatedUnivariateSpline(s, D)
    D_norm = D_norm_spl.integral(s[0], s[-1])
    Te_D_spl = InterpolatedUnivariateSpline(s, D * Te_los / D_norm)
    Te_rad = Te_D_spl.integral(s[0], s[-1])
    print("Normalization of BPD: ", D_norm)
    print("Trad background from integration of BPD with reflections", Te_rad)
    print("Trad background from radiation Transport", Trad_comp * 1.e3)  # * (1.0 - 0.9 * np.exp(-tau_comp)) / (1.0 - np.exp(-tau_comp)))
    lns = ax1.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs)
    plt.show()

def compare_Trad_difs(shot, parent_dir, editions, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markes = ["+", "*", "^", "v", "d", "s", "<"]
    for i in range(len(editions)):
        path = os.path.join(parent_dir, editions[i], "ecfm_data")
        if(i == 0):
            rhop_Gene, R_Gene, z_Gene, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(path, 33585, 3.0)
            Te_perp, Te_par = get_dist_moments_non_rel(rhop_Gene, beta_par, mu_norm, f, Te, ne, B0, slices=1)
            plt.plot(rhop_Gene, (Te_perp - Te), "-", label=r"GENE $\tilde{T}_{\perp,\mathrm{e}}$", markersize=20)
            plt.plot(rhop_Gene, (Te_par - Te), "--", label=r"GENE $\tilde{T}_{\parallel,\mathrm{e}}$", markersize=20)
        Trad = np.loadtxt(os.path.join(path, "TRadM_GENE.dat"))
        Trad_comp = np.loadtxt(os.path.join(path, "TRadM_therm.dat"))
        rel_res = np.loadtxt(os.path.join(path, "sres_rel.dat"))
        rhop_warm = rel_res.T[3]
        plt.plot(rhop_warm, (Trad.T[1] - Trad_comp.T[1]) * 1.e3, markes[i], label=labels[i], markersize=20)
    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax.set_ylabel(r"$\tilde{T_\mathrm{rad}}$ [eV]")
    ax.legend(ncol=3)
    plt.show()

def compare_Trad_difs_light(parent_dir, editions, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markes = ["+", "*", "^", "v", "d", "s", "<"]
    for i in range(len(editions)):
        path = os.path.join(parent_dir, editions[i], "ecfm_data")
        Trad = np.loadtxt(os.path.join(path, "TRadM_GENE.dat"))
        Trad_comp = np.loadtxt(os.path.join(path, "TRadM_therm.dat"))
        rel_res = np.loadtxt(os.path.join(path, "sres_rel.dat"))
        rhop_warm = rel_res.T[3]
        plt.plot(rhop_warm, (Trad.T[1] - Trad_comp.T[1]) * 1.e3, markes[i], label=labels[i], markersize=20)
    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax.set_ylabel(r"$\tilde{T_\mathrm{rad}}$ [eV]")
    ax.legend(ncol=3)
    plt.show()

def check_BPD_norm(BPD_file, n1=1, n2=2):
    BPD_data = np.loadtxt(BPD_file)
    x_BPD = BPD_data.T[0]
    D = BPD_data.T[n1]
    D_2 = BPD_data.T[n2]
    D_spline = InterpolatedUnivariateSpline(x_BPD, D)
    D_2_spline = InterpolatedUnivariateSpline(x_BPD, D_2)
    x1 = np.min(x_BPD)
    x2 = np.max(x_BPD)
    print(D_spline.integral(x1, x2), D_2_spline.integral(x1, x2))

def compare_Trad_difs_mat(filename_list, gene_directory, labels):
    markers = ["+", "*"]  # , "^", "v", "d", "s", "<"]
    shot = -1
    result_list = []
    for filename in filename_list:
        result_list.append(ECRadResults())
        result_list[-1].from_mat_file(filename)
    if(shot == -1):
        shot = result_list[0].Scenario.shot
        time = result_list[0].time[0]
    for it in range(len(result_list[0].time)):
        for result, label, marker in zip(result_list, labels, markers):
            plt.plot(result.resonance["rhop_warm"][it], (result.Trad[it] - result.Trad_comp[it]) * 1.e3, \
                     marker, label=label + " {0:d}".format(it), markersize=20)
        gene_obj = Gene_BiMax(gene_directory, shot, time=time, it=it)
        gene_obj.make_bi_max()
        plt.plot(gene_obj.rhop, (gene_obj.Te_perp - gene_obj.Te), "-", label=r"GENE $\tilde{T}_{\perp,\mathrm{e}}$", markersize=20)
        plt.plot(gene_obj.rhop, (gene_obj.Te_par - gene_obj.Te), "--", label=r"GENE $\tilde{T}_{\parallel,\mathrm{e}}$", markersize=20)
        plt.show()

def dif_vs_difs(filename_gene, filename_bimax):
    gene_result = ECRadResults()
    bimax_result = ECRadResults()
    gene_result.from_mat_file(filename_gene)
    bimax_result.from_mat_file(filename_bimax)
    filename_bimax
    for it in range(len(gene_result.time)):
        plt.plot((bimax_result.Trad[it] - bimax_result.Trad_comp[it]) * 1.e3, \
                 (gene_result.Trad[it] - gene_result.Trad_comp[it]) * 1.e3, \
                     "+", label="time {0:d}".format(it), markersize=20)
    ones = np.zeros(bimax_result.Trad.flatten().shape)
    ones[:] = 1.0
    slope, intercept, rvalue, pvalue, stderr = linregress((bimax_result.Trad.flatten() - bimax_result.Trad_comp.flatten()), \
                          (gene_result.Trad.flatten() - gene_result.Trad_comp.flatten()))
    Trad_ax = np.linspace(np.min((bimax_result.Trad.flatten() - bimax_result.Trad_comp.flatten())) * 1.2e3, \
                          np.max((bimax_result.Trad.flatten() - bimax_result.Trad_comp.flatten())) * 1.2e3, 100)
    art_data = Trad_ax * slope + intercept
    print(slope, intercept)
    plt.plot(Trad_ax, art_data)
    plt.gca().set_xlabel(r"$\tilde{T}_\mathrm{rad}[\mathrm{BiMax}]$ [eV]")
    plt.gca().set_ylabel(r"$\tilde{T}_\mathrm{rad}[\mathrm{Gene}]$ [eV]")
    plt.show()


def check_IDA_Te_curv(curv_scale, rp1, rp1_decay, rp1_scal, rp2, rp2_decay, rp2_scal):
    rhop = np.linspace(0, 1.01, 100)
    curv_prior = np.zeros(100)
    curv_prior[rhop < rp1] = curv_scale * (rp1_scal + (1.e0 - rp1_scal) * np.exp((rhop[rhop < rp1] - rp1) / rp1_decay))
    curv_prior[rhop > rp2] = curv_scale * (rp2_scal + (1.e0 - rp2_scal) * np.exp((rhop[rhop > rp2] - rp2) / rp2_decay))
    curv_prior[np.logical_and(rhop > rp1, rhop < rp2)] = curv_scale
    plt.plot(rhop, curv_prior)
    plt.show()

def fast_ECE_2D_plot(shot, time_window):
    time, freq, Te = get_RMC_data_calib(shot, time_window)
    t_c = time_window[0] + 0.5 * (time_window[1] - time_window[0])
    levels = np.linspace(1.0, 5.0, 25)
    cont1 = plt.contourf((time - t_c) * 1000, freq * 1.e-9, np.log10(Te.T), levels, cmap=plt.cm.plasma)
    cont2 = plt.contour((time - t_c) * 1000, freq * 1.e-9, np.log10(Te.T), levels, cmap=plt.cm.plasma)
    plt.gca().set_xlabel(r"$t [\si{\milli\second}]$")
    plt.gca().set_ylabel(r"$f [\si{\giga\hertz}]$")
    for c in cont2.collections:
        c.set_linestyle('solid')
    cb = plt.gcf().colorbar(cont1, ax=plt.gca(), ticks=[np.linspace(1.0, 5.0, 5)])
    cb.set_label(r"$\log_{10}(T_\mathrm{rad}[\si{\electronvolt}])$")
    cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50])))
    cb.ax.minorticks_on()
    plt.show()

def correlate(filenames, labels):
    figure_1 = plt.figure()
    ax1 = figure_1.add_subplot(111)
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"Radial cross correlation")
    figure_2 = plt.figure()
    ax2 = figure_2.add_subplot(111)
    ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax2.set_ylabel(r"Radial cross correlation [\%]")
    figure_3 = plt.figure()
    ax3 = figure_3.add_subplot(111)
    ax3.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax3.set_ylabel(r"Normalized radial cross cor.")
    for filename, label in zip(filenames, labels):
        ECRad_res = ECRadResults()
        ECRad_res.from_mat_file(filename)
        Trads = (ECRad_res.Trad - ECRad_res.Trad_comp) / ECRad_res.Trad_comp
        sets = []
        std_devs = []
        for ich in range(len(Trads.T) / 2):
            sets.append(np.zeros(len(ECRad_res.time)))
            for it in range(len(ECRad_res.time)):
                sets[ich][it] = Trads[it][ich] * Trads[it][ich + 1]
            std_devs.append(np.std(Trads.T[ich]) * np.std(Trads.T[ich + 1]))
                # sets[ich][it] = ECRad_res.Trad_comp[it][ich] * ECRad_res.Trad_comp[it][ich + 1]
        sets = np.array(sets)
        std_devs = np.array(std_devs)
        ax1.plot(ECRad_res.resonance["rhop_warm"][0][::2], np.mean(sets, axis=1), "+", label=label)
        ax2.plot(ECRad_res.resonance["rhop_warm"][0][::2], np.sqrt(np.abs(np.mean(sets, axis=1))) * 100.0, "+", label=label)
        ax3.plot(ECRad_res.resonance["rhop_warm"][0][::2], np.mean(sets, axis=1) / std_devs, "+", label=label)
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax3.legend(loc=0)
    plt.show()

def correlate_with_one(filenames, labels):
    figure_1 = plt.figure()
    ax1 = figure_1.add_subplot(111)
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"Radial cross correlation")
    figure_2 = plt.figure()
    ax2 = figure_2.add_subplot(111)
    ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax2.set_ylabel(r"Radial cross correlation [\%]")
    figure_3 = plt.figure()
    ax3 = figure_3.add_subplot(111)
    ax3.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax3.set_ylabel(r"Normalized radial cross cor.")
    for filename, label in zip(filenames, labels):
        ECRad_res = ECRadResults()
        ECRad_res.from_mat_file(filename)
        Ch_max = len(ECRad_res.resonance["rhop_warm"][0])
        Trads = (ECRad_res.Trad - ECRad_res.Trad_comp) / ECRad_res.Trad_comp
        sets = []
        std_devs = []
        for ich in range(1, len(Trads.T)):
            sets.append(np.zeros(len(ECRad_res.time)))
            for it in range(len(ECRad_res.time)):
                sets[ich - 1][it] = Trads[it][0] * Trads[it][ich]
            std_devs.append(np.std(Trads.T[0]) * np.std(Trads.T[ich]))
                # sets[ich][it] = ECRad_res.Trad_comp[it][ich] * ECRad_res.Trad_comp[it][ich + 1]
        sets = np.array(sets)
        std_devs = np.array(std_devs)
        ax1.plot(ECRad_res.resonance["rhop_warm"][0][1:Ch_max], np.mean(sets, axis=1), "+", label=label)
        ax2.plot(ECRad_res.resonance["rhop_warm"][0][1:Ch_max], np.sqrt(np.abs(np.mean(sets, axis=1))) * 100.0, "+", label=label)
        ax3.plot(ECRad_res.resonance["rhop_warm"][0][1:Ch_max], np.mean(sets, axis=1) / std_devs, "+", label=label)
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax3.legend(loc=0)
    plt.show()

def benchmark_absorption(working_dir):
    a = np.loadtxt(os.path.join(working_dir, "v_high_ne_abs_prof.dat"))
    b = np.loadtxt(os.path.join(working_dir, "high_ne_abs_prof.dat"))
    c = np.loadtxt(os.path.join(working_dir, "low_ne_abs_prof.dat"))
    i_1 = 1  # Albajar
    # i_2 = 3  # Gray
    i_2 = 4  # Crude Approx.
    fig1 = plt.figure(figsize=(8.5, 8.5))
    fig2 = plt.figure(figsize=(8.5, 8.5))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax1.semilogy(a.T[0], a.T[i_1], "-r", label="$n_\mathrm{e} = \SI{1.0e20}{\per\cubic\meter}$")
    ax1.semilogy(a.T[0], a.T[i_2], "--b")  # , label="fully relativistic $n_\mathrm{e} = \SI{8.0e19}{\per\cubic\meter}$"
    # ax1.semilogy(a.T[0], a.T[i_2], "-m")
    # plt.figure()
    ax1.semilogy(b.T[0], b.T[i_1], "-m", label="$n_\mathrm{e} = \SI{8.0e19}{\per\cubic\meter}$")
    ax1.semilogy(b.T[0], b.T[i_2], "--c")  # , label="fully relativistic $n_\mathrm{e} = \SI{4.0e19}{\per\cubic\meter}$"
    # plt.semilogy(b.T[0], b.T[i_2], "--m")
    # plt.figure()
    ax1.semilogy(c.T[0], c.T[i_1], "-y", label="$n_\mathrm{e} = \SI{6.0e19}{\per\cubic\meter}$")
    ax1.semilogy(c.T[0], c.T[i_2], "--k")  # , label="fully relativistic $n_\mathrm{e} = \SI{2.0e19}{\per\cubic\meter}$"
    # plt.semilogy(c.T[0], c.T[i_2], ":m")
    ax1.legend()
    ax1.set_xlabel(r"$\frac{\omega_\mathrm{c,0}}{\omega}$")
    ax1.set_ylabel(r"$\alpha_\omega\,[\si{\per\metre}]$")
    ax1.set_xlim(1.68, 2.01)
    # ax1.set_ylim(2.e0, 10 ** np.ceil(np.log10(np.max([np.max(a.T[i_1]), np.max(a.T[i_2]), np.max(b.T[i_1]), np.max(b.T[i_2]), np.max(c.T[i_1]), np.max(c.T[i_2])]))))
    ax1.set_ylim(2.e0, 10 ** 4)
    a = np.loadtxt(os.path.join(working_dir, "v_high_ne_abs_prof_low_f.dat"))
    b = np.loadtxt(os.path.join(working_dir, "high_ne_abs_prof_low_f.dat"))
    c = np.loadtxt(os.path.join(working_dir, "low_ne_abs_prof_low_f.dat"))
    ax2.semilogy(a.T[0], a.T[i_1], "-r", label="$n_\mathrm{e} = \SI{6.0e19}{\per\cubic\meter}$")
    ax2.semilogy(a.T[0], a.T[i_2], "--b")  # , label="fully relativistic $n_\mathrm{e} = \SI{6.0e19}{\per\cubic\meter}$"
    # ax2.semilogy(a.T[0], a.T[i_2], "-m")
    # plt.figure()
    ax2.semilogy(b.T[0], b.T[i_1], "-m", label="$n_\mathrm{e} = \SI{4.0e19}{\per\cubic\meter}$")
    ax2.semilogy(b.T[0], b.T[i_2], "--c")  # , label="fully relativistic $n_\mathrm{e} = \SI{3.0e19}{\per\cubic\meter}$"
    # ax2.semilogy(b.T[0], b.T[3], "--m")
    # plt.figure()
    ax2.semilogy(c.T[0], c.T[i_1], "-y", label="$n_\mathrm{e} = \SI{2.0e19}{\per\cubic\meter}$")
    ax2.semilogy(c.T[0], c.T[i_2], "--k")  # , label="fully relativistic $n_\mathrm{e} = \SI{1.0e19}{\per\cubic\meter}$"
    ax2.legend()
    ax2.set_xlabel(r"$\frac{\omega_\mathrm{c,0}}{\omega}$")
    ax2.set_ylabel(r"$\alpha_\omega\,[\si{\per\metre}]$")
    ax2.set_xlim(1.68, 2.01)
    # ax2.set_ylim(2.e0, 10 ** np.ceil(np.log10(np.max([np.max(a.T[i_1]), np.max(a.T[i_2]), np.max(b.T[i_1]), np.max(b.T[i_2]), np.max(c.T[i_1]), np.max(c.T[i_2])]))))
    ax2.set_ylim(2.e0, 10 ** 4)
    plt.show()

def plot_BPD_for_all_rays(result_file, time, ch_list, mode="X"):
    result = ECRadResults()
    result.from_mat_file(result_file)
    itime = np.argmin(result.time - time)
    for ch in ch_list:
        for ir in range(result.Config.N_ray):
            plt.plot(result.ray["rhop" + mode][itime][ch - 1][ir], result.ray["BPD" + mode][itime][ch - 1][ir])
        plt.show()

def Energy_los(beta, B=2.5):
    gamma = np.sqrt(1.0 / (1.0 - beta ** 2))
    U_B = B ** 2 / (2.0 * cnst.mu_0)
    sigma_T = cnst.mu_0 * cnst.e ** 4 / (6.0 * np.pi * cnst.epsilon_0 * cnst.c ** 2 * cnst.m_e ** 2)
    P_los = 4.0 / 3.0 * beta ** 2 * gamma ** 2 * cnst.c * sigma_T * U_B
    E_kin = cnst.c ** 2 * cnst.m_e * (gamma - 1.0)
    print(E_kin / P_los)


def refr_index_coupling(ne, omega, omega_c, mode= -1):
    theta = np.linspace(0, np.pi, 100)
    N_omega = np.zeros(100)
    N_s = np.zeros(100)
    em_alb = em_abs_Alb()
    for i in range(len(N_omega)):
        svec = s_vec(0.9, 400, ne, omega_c / np.pi, theta[i])
        N_omega[i] = em_alb.refr_index(svec, omega, mode)
        X = cnst.e ** 2 * svec.ne / (cnst.epsilon_0 * cnst.m_e) / omega ** 2
        Y = svec.freq_2X * np.pi / omega
        N_par = N_omega[i] * svec.cos_theta
        Delta = (1.e0 - (N_par) ** 2) ** 2 + 4.e0 * N_par ** 2 * (1.e0 - X) / Y ** 2
        N_s[i] = 1.e0 - X + ((1.e0 + float(mode) * Delta + N_par ** 2) * X * Y ** 2) / (2.e0 * (-1.e0 + X + Y ** 2))
    plt.plot(np.rad2deg(theta), N_omega ** 2)
    plt.plot(np.rad2deg(theta), N_s)
    plt.show()
# def bin_BPD_to_common_rhop(working_dir, ch, mode, N=1000):
#    BPD_name = "BPD_ray*ch{0:03d}_{1:1s}".format(ch, mode)
#    BPD_files = glob.glob(os.path.join(working_dir, BPD_name))
#    rays_s = []
#    rays_rhop = []
#    rays_BPD = []
#    rhop_max = -np.inf
#    for BPD_file in BPD_files:
#        BPD_data = np.loadtxt(BPD_file)
#        rays_s.append(BPD_data.T[0])
#        rays_rhop.append(BPD_data.T[4])
#        rays_BPD.append(BPD_data.T[5])
#        if(np.max(rays_rhop[-1]) > rhop_max):
#            rhop_max = np.max(rays_rhop[-1])
#    rhop = np.linspace(-rhop_max, rhop_max, N)
#    BPD = np.zeros(len(rhop))
#    for i in range(len(rays_s)):
#        if(np.sum(rays_BPD[i]) < 1.e-5):
#            print("Skipping ray file " + BPD_files[i], " because BPD too tiny")
#        # Scan the along s to indetify peaks in BPD
#        j_seg_end = 0
#        j_seg_start = 9
#        while j_seg_end < len(rays_s[i]):
#             if(rays_BPD)

def boundary_dependence_tranmittance(folder, shot, time, dist, ich, mode_str="X", eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005):
    Npts = 200
    em_abs_obj = em_abs_Alb()
    ray_dict = read_ray_dict_from_file(folder, dist, ich, mode=mode_str)
    svec_dict, freq = read_svec_dict_from_file(folder, ich, mode=mode_str)
    rhop_ne, ne = np.loadtxt(os.path.join(folder, "ne_file.dat"), skiprows=1, unpack=True)
    launch_file = np.loadtxt(os.path.join(folder, "ray_launch.dat"), skiprows=1)
    make_perp = True
    x_launch = np.zeros(3)
    x_launch[0] = launch_file[ich - 1][2] * 1.e-2
    x_launch[1] = launch_file[ich - 1][3] * 1.e-2
    x_launch[2] = launch_file[ich - 1][4] * 1.e-2
    omega = freq * 2.0 * np.pi
#    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
#    EQ_slice = EQ_obj.GetSlice(float(time))
#    EQ_obj.add_ripple_to_slice(time, EQ_slice)
    if(mode_str == "O"):
        mode = -1
    else:
        mode = 1
#    print(em_abs_obj.test_filter_transmittance(svec_dict, ray_dict, EQ_slice, omega, mode))
    transmittance = np.zeros(Npts)
    new_transmittance = np.zeros(Npts)
    mismatch_filter_mag = np.zeros(Npts)
    boundary = np.linspace(0.90, 1.15, Npts)
    x_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
    y_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
    z_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
    Nx_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Nx"])
    Ny_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Ny"])
    Nz_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Nz"])
    Bx_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bx"])
    By_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["By"])
    Bz_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bz"])
    ne_spl = InterpolatedUnivariateSpline(rhop_ne, np.log(ne / 1.e19))
    x_vec = np.zeros(3)
    R_vec = np.zeros(3)
    B_vec = np.zeros(3)
    N_vec = np.zeros(3)
    for i in range(len(boundary)):
        rhop_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["rhop"] - boundary[i])
        s_sep = np.max(rhop_spl.roots())
        x_vec[0] = x_spl(s_sep)
        x_vec[1] = y_spl(s_sep)
        x_vec[2] = z_spl(s_sep)
        N_vec[0] = Nx_spl(s_sep)
        N_vec[1] = Ny_spl(s_sep)
        N_vec[2] = Nz_spl(s_sep)
        B_vec[0] = Bx_spl(s_sep)
        B_vec[1] = By_spl(s_sep)
        B_vec[2] = Bz_spl(s_sep)
        ne = 1.e19 * np.exp(ne_spl(boundary[i]))
        R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        R_vec[1] = np.arctan2(x_vec[1], x_vec[0])
        R_vec[2] = x_vec[2]
        omega_c = cnst.e / cnst.m_e * np.sqrt(np.sum(B_vec ** 2))
        X = cnst.e ** 2 * ne / (cnst.m_e * cnst.epsilon_0 * omega ** 2)
        Y = omega_c / omega
        phi = np.arctan2(x_vec[1], x_vec[0])
        B_pol = B_vec[2]  # np.sqrt((B_vec[0] * np.cos(phi) + B_vec[1] * np.sin(phi)) ** 2 + B_vec[2] ** 2)
        B_tor = B_vec[1] * np.cos(phi) - B_vec[0] * np.sin(phi)
        if(make_perp):
            # Eliminate E_r
            B_vec[0] = -B_tor * np.sin(R_vec[1])
            B_vec[1] = +B_tor * np.cos(R_vec[1])
            B_vec[2] = B_pol
            # Make N perpendicular to B for testing purposes
            N_vec = N_vec / np.linalg.norm(N_vec) - np.dot(N_vec / np.linalg.norm(N_vec), B_vec / np.linalg.norm(B_vec)) * B_vec / np.linalg.norm(B_vec)
            N_vec /= np.linalg.norm(N_vec)
#        print(B_pol, B_tor)
#        print("Pitch angle at point of decoupling", np.arctan2(B_pol, B_tor))
#        print("Square of pitch angle", np.arctan2(B_pol, B_tor) ** 2)
        mismatch_filter_mag[i] = np.cos(np.arctan2(B_pol, B_tor)) ** 2
        transmittance[i] = em_abs_obj.get_filter_transmittance_new(omega, X, Y, mode, x_vec, N_vec, B_vec)
        new_transmittance[i] = em_abs_obj.get_filter_transmittance_correct_filer(omega, X, Y, mode, x_vec, N_vec, B_vec, x_launch)
    plt.plot(boundary, mismatch_filter_mag, label=r"From aligment $\vec{B} \cdot \vec{f}$")
    plt.plot(boundary, transmittance, "--", label=r"Rotation of $\vec{e}$ to ray cood. system")
    plt.plot(boundary, new_transmittance, ":", label=r"Rotation of $\vec{f}$ to stix cood. system")
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$ boundary")
    plt.gca().set_ylabel(r"$X$-mode fraction")
    plt.gca().legend()
    plt.show()
    return transmittance

def polarizer_angle_dependence_tranmittance(folder, shot, time, dist, ich, mode_str="X"):
    Npts = 200
    em_abs_obj = em_abs_Alb()
    ray_dict = read_ray_dict_from_file(folder, dist, ich, mode=mode_str)
    svec_dict, freq = read_svec_dict_from_file(folder, ich, mode=mode_str)
    rhop_ne, ne = np.loadtxt(os.path.join(folder, "ne_file.dat"), skiprows=1, unpack=True)
    launch_file = np.loadtxt(os.path.join(folder, "ray_launch.dat"), skiprows=1)
    x_launch = np.zeros(3)
    x_launch[0] = launch_file[ich - 1][2] * 1.e-2
    x_launch[1] = launch_file[ich - 1][3] * 1.e-2
    x_launch[2] = launch_file[ich - 1][4] * 1.e-2
    omega = freq * 2.0 * np.pi
#    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
#    EQ_slice = EQ_obj.GetSlice(float(time))
#    EQ_obj.add_ripple_to_slice(time, EQ_slice)
    if(mode_str == "O"):
        mode = -1
    else:
        mode = 1
#    print(em_abs_obj.test_filter_transmittance(svec_dict, ray_dict, EQ_slice, omega, mode))
    transmittance = np.zeros(Npts)
    x_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
    y_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
    z_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
    Bx_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bx"])
    By_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["By"])
    Bz_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bz"])
    ne_spl = InterpolatedUnivariateSpline(rhop_ne, np.log(ne / 1.e19))
    x_vec = np.zeros(3)
    B_vec = np.zeros(3)
    N_vec = np.zeros(3)
    boundary = 1.05
    rhop_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["rhop"] - boundary)
    s_sep = np.max(rhop_spl.roots())
    x_vec[0] = x_spl(s_sep)
    x_vec[1] = y_spl(s_sep)
    x_vec[2] = z_spl(s_sep)
    x_vec[0] = x_spl(s_sep)
    x_vec[1] = y_spl(s_sep)
    x_vec[2] = z_spl(s_sep)
    N_vec[:] = x_launch[:]
    N_vec /= np.linalg.norm(N_vec)
    N_vec = np.array([1.0, 0.0, 0.0])
    B_vec = np.array([0.0, -2.1, 0.1])
#    B_vec[0] = Bx_spl(s_sep)
#    B_vec[1] = By_spl(s_sep)
#    B_vec[2] = Bz_spl(s_sep)
    ne = 1.e19 * np.exp(ne_spl(boundary))
    omega_c = cnst.e / cnst.m_e * np.sqrt(np.sum(B_vec ** 2))
    X = cnst.e ** 2 * ne / (cnst.m_e * cnst.epsilon_0 * omega ** 2)
    Y = omega_c / omega
    polarizer_angle = np.linspace(-np.pi / 4.0, np.pi / 4.0, Npts)
    for i in range(Npts):
        transmittance[i] = em_abs_obj.get_filter_transmittance_reverse(omega, X, Y, mode, x_vec, N_vec, B_vec, polarizer_angle[i])
    plt.plot(np.rad2deg(polarizer_angle), transmittance)
    plt.gca().set_xlabel(r"$\theta_\mathrm{polarizer}$ boundary")
    plt.gca().set_ylabel(r"$X$-mode fraction")
    plt.show()
    return transmittance

def phi_tor_dependence_tranmittance(folder, shot, time, dist, ich, mode_str="X", eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005):
    em_abs_obj = em_abs_Alb()
    N = 180
    phi_tor = np.linspace(-45.0, 45.0, N)
    transmittance = np.zeros((N))
    new_transmittance = np.zeros((N))
    ray_dict = read_ray_dict_from_file(folder, dist, ich, mode=mode_str)
    svec_dict, freq = read_svec_dict_from_file(folder, ich, mode=mode_str)
    rhop_ne, ne = np.loadtxt(os.path.join(folder, "ne_file.dat"), skiprows=1, unpack=True)
    launch_file = np.loadtxt(os.path.join(folder, "ray_launch.dat"), skiprows=1)
    x_launch = np.zeros(3)
    x_launch[0] = launch_file[ich - 1][2] * 1.e-2
    x_launch[1] = launch_file[ich - 1][3] * 1.e-2
    x_launch[2] = launch_file[ich - 1][4] * 1.e-2
    theta_pol = launch_file[ich - 1][6]
    omega = freq * 2.0 * np.pi
#    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
#    EQ_slice = EQ_obj.GetSlice(float(time))
#    EQ_obj.add_ripple_to_slice(time, EQ_slice)
    if(mode_str == "O"):
        mode = -1
    else:
        mode = 1
#    print(em_abs_obj.test_filter_transmittance(svec_dict, ray_dict, EQ_slice, omega, mode))
    x_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
    y_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
    z_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
    ne_spl = InterpolatedUnivariateSpline(rhop_ne, np.log(ne / 1.e19))
    x_vec = np.zeros(3)
    boundary = 1.05
    rhop_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["rhop"] - boundary)
    s_sep = np.max(rhop_spl.roots())
    x_vec[0] = x_spl(s_sep)
    x_vec[1] = y_spl(s_sep)
    x_vec[2] = z_spl(s_sep)
    B_vec = np.array([0.0, -2.1, 0.0])
    N_vec = np.zeros(3)
#    N_vec[:] = x_launch[:]
#    N_vec /= np.linalg.norm(N_vec)
#    B_vec[0] = Bx_spl(s_sep)
#    B_vec[1] = By_spl(s_sep)
#    B_vec[2] = Bz_spl(s_sep)
    ne = 1.e19 * np.exp(ne_spl(boundary))
    omega_c = cnst.e / cnst.m_e * np.sqrt(np.sum(B_vec ** 2))
    X = cnst.e ** 2 * ne / (cnst.m_e * cnst.epsilon_0 * omega ** 2)
    Y = omega_c / omega
#    print("rot angle theta", np.rad2deg(angle))
#    theta_check = np.arccos(np.dot(N_vec_par_theta, B_vec) / np.linalg.norm(N_vec_par_theta) / np.linalg.norm(B_vec))
#    print("theta after initial theta rotation - should be 90", np.rad2deg(theta_check))
    for i in range(len(phi_tor)):
        N_vec[:] = x_vec / np.linalg.norm(x_vec)
        temp1 = np.arctan2(N_vec[2], N_vec[1])
        N_vec[0] = 1.e0
        N_vec[1] = temp1
        # Poloidal angle is aligned with R direction
        N_vec[2] = np.pi / 2.e0
        N_vec[1] += np.deg2rad(phi_tor[i])
        N_vec[2] += np.deg2rad(theta_pol)
        temp = np.cos(N_vec[1]) * np.sin(N_vec[2])
        temp1 = np.sin(N_vec[1]) * np.sin(N_vec[2])
        temp2 = np.cos(N_vec[2])
        N_vec[0] = temp
        N_vec[1] = temp1
        N_vec[2] = temp2
        N_vec /= np.linalg.norm(N_vec)
        transmittance[i] = em_abs_obj.get_filter_transmittance_correct_filter(omega, X, Y, mode, x_vec, N_vec, B_vec, x_launch)
        new_transmittance[i] = em_abs_obj.get_filter_transmittance_reverse_correct_filter(omega, X, Y, mode, x_vec, N_vec, B_vec, x_launch)
    print("Max new:", np.max(new_transmittance))
    plt.plot(phi_tor, transmittance)
    plt.plot(phi_tor, new_transmittance)
    plt.gca().set_xlabel(r"Toroidal launch angle [$^\circ$]")
    plt.gca().set_ylabel(r"X-mode fraction")
    plt.show()
    return transmittance, new_transmittance


def theta_dependence_tranmittance(folder, shot, time, dist, ich, mode_str="X", eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005):
    em_abs_obj = em_abs_Alb()
    N = 101
    phi = np.linspace(-np.pi * 0.25, np.pi * 0.25, N)
    theta = np.linspace(-np.pi * 0.25, np.pi * 0.25, N)
    ThetaCoords = np.zeros((N, N))
    PhiCoords = np.zeros((N, N))
    transmittance = np.zeros((N, N))
    new_transmittance = np.zeros((N, N))
    transmittance_B_rot = np.zeros((N, N))
    new_transmittance_B_rot = np.zeros((N, N))
    ray_dict = read_ray_dict_from_file(folder, dist, ich, mode=mode_str)
    svec_dict, freq = read_svec_dict_from_file(folder, ich, mode=mode_str)
    rhop_ne, ne = np.loadtxt(os.path.join(folder, "ne_file.dat"), skiprows=1, unpack=True)
    launch_file = np.loadtxt(os.path.join(folder, "ray_launch.dat"), skiprows=1)
    x_launch = np.zeros(3)
    x_launch[0] = launch_file[ich - 1][2] * 1.e-2
    x_launch[1] = launch_file[ich - 1][3] * 1.e-2
    x_launch[2] = launch_file[ich - 1][4] * 1.e-2
    omega = freq * 2.0 * np.pi
#    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
#    EQ_slice = EQ_obj.GetSlice(float(time))
#    EQ_obj.add_ripple_to_slice(time, EQ_slice)
    if(mode_str == "O"):
        mode = -1
    else:
        mode = 1
#    print(em_abs_obj.test_filter_transmittance(svec_dict, ray_dict, EQ_slice, omega, mode))
    x_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
    y_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
    z_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
    Bx_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bx"])
    By_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["By"])
    Bz_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bz"])
    ne_spl = InterpolatedUnivariateSpline(rhop_ne, np.log(ne / 1.e19))
    x_vec = np.zeros(3)
    R_vec = np.zeros(3)
    B_vec = np.zeros(3)
    N_vec = np.zeros(3)
    boundary = 1.05
    rhop_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["rhop"] - boundary)
    s_sep = np.max(rhop_spl.roots())
    x_vec[0] = x_spl(s_sep)
    x_vec[1] = y_spl(s_sep)
    x_vec[2] = z_spl(s_sep)
    N_vec = np.array([1.0, 0.0, 0.0])
    B_vec = np.array([0.0, -2.1, 0.0])
#    N_vec[:] = x_launch[:]
#    N_vec /= np.linalg.norm(N_vec)
#    B_vec[0] = Bx_spl(s_sep)
#    B_vec[1] = By_spl(s_sep)
#    B_vec[2] = Bz_spl(s_sep)
    ne = 1.e19 * np.exp(ne_spl(boundary))
    R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
    R_vec[1] = np.arctan2(x_vec[1], x_vec[0])
    R_vec[2] = x_vec[2]
    omega_c = cnst.e / cnst.m_e * np.sqrt(np.sum(B_vec ** 2))
    X = cnst.e ** 2 * ne / (cnst.m_e * cnst.epsilon_0 * omega ** 2)
    Y = omega_c / omega
    rot_base_theta = np.cross(N_vec, B_vec)
    rot_base_theta /= np.linalg.norm(rot_base_theta)
    angle = -np.pi / 2.0 + np.arccos(np.dot(N_vec, B_vec) / np.linalg.norm(N_vec) / np.linalg.norm(B_vec))
#    print("rot angle theta", np.rad2deg(angle))
    N_vec_par_theta = rotate_vec_around_axis(N_vec, rot_base_theta, angle)
#    theta_check = np.arccos(np.dot(N_vec_par_theta, B_vec) / np.linalg.norm(N_vec_par_theta) / np.linalg.norm(B_vec))
#    print("theta after initial theta rotation - should be 90", np.rad2deg(theta_check))
    rot_base_phi = B_vec / np.linalg.norm(B_vec)
    angle = -np.pi / 2.0 + np.arccos(np.dot(N_vec_par_theta, rot_base_theta) / np.linalg.norm(N_vec_par_theta) / np.linalg.norm(rot_base_theta))
#    print("Phi initial", np.rad2deg(angle))
    N_vec_par_phi = rotate_vec_around_axis(N_vec_par_theta, rot_base_phi, angle)
    phi_check = np.arccos(np.round(np.dot(N_vec_par_phi, rot_base_theta) / np.linalg.norm(N_vec_par_phi) / np.linalg.norm(rot_base_theta), 7))
    theta_check = np.arccos(np.dot(N_vec_par_phi, B_vec) / np.linalg.norm(N_vec_par_phi) / np.linalg.norm(B_vec))
    print("Phi after initial phi rotation - should be 90", np.rad2deg(phi_check))
    print("theta after initial theta and phi rotation - should still be 90", np.rad2deg(theta_check))
    for i in range(len(phi)):
        N_vec_phi_rot = rotate_vec_around_axis(N_vec_par_phi, rot_base_phi, -phi[i])
        print("before phi rot", rot_base_theta)
        phi_check = np.arccos(np.round(np.dot(N_vec_phi_rot, rot_base_theta) / np.linalg.norm(N_vec_phi_rot) / np.linalg.norm(rot_base_theta), 7))
        print("phi check before theta rot", phi_check)
        rot_base_theta = np.cross(N_vec_phi_rot, B_vec)
        rot_base_theta /= np.linalg.norm(rot_base_theta)
        print("after phi rot", rot_base_theta)
#        print("phi", np.round(np.rad2deg(phi[i]), 2), np.round(np.rad2deg(phi_check), 2))
#        theta_check = np.arccos(np.dot(N_vec_phi_rot, B_vec) / np.linalg.norm(N_vec_phi_rot) / np.linalg.norm(B_vec))
#        print("theta initial should be 90", np.round(np.rad2deg(theta_check), 2))
        for j in range(len(theta)):
            N_vec_theta_rot = rotate_vec_around_axis(N_vec_par_phi, rot_base_theta, -theta[j])
            theta_check = np.arccos(np.round(np.dot(N_vec_theta_rot, B_vec) / np.linalg.norm(N_vec_theta_rot) / np.linalg.norm(B_vec), 7))
            phi_check = np.arccos(np.round(np.dot(N_vec_phi_rot, rot_base_theta) / np.linalg.norm(N_vec_phi_rot) / np.linalg.norm(rot_base_theta), 7))
            print("theta", np.round(np.rad2deg(theta[j]), 2), np.round(np.rad2deg(theta_check), 2))
            print("phi", np.round(np.rad2deg(phi[i]), 2), np.round(np.rad2deg(phi_check), 2))
            PhiCoords[i, j] = phi_check
            ThetaCoords[i, j] = theta_check
            transmittance[i, j] = em_abs_obj.get_filter_transmittance(omega, X, Y, mode, x_vec, N_vec_theta_rot, B_vec, x_launch)
            new_transmittance[i, j] = em_abs_obj.get_filter_transmittance_new(omega, X, Y, mode, x_vec, N_vec_theta_rot, B_vec)
        plt.plot(np.rad2deg(theta), np.rad2deg(ThetaCoords[i]))
        plt.show()
        theta_spl = InterpolatedUnivariateSpline(np.rad2deg(theta), np.rad2deg(ThetaCoords[i]))
        plt.plot(np.rad2deg(theta), theta_spl(np.rad2deg(theta), nu=1))
        plt.show()
#    for i in range(len(phi)):
#        rot_base = np.cross(B_vec, N_vec)
#        rot_base /= np.linalg.norm(rot_base)
#        angle = -np.pi / 2.0 + np.arccos(np.dot(N_vec, B_vec) / np.linalg.norm(N_vec) / np.linalg.norm(B_vec))
#        B_vec_par = rotate_vec_around_axis(B_vec, rot_base, angle)
#        theta_check = np.arccos(np.dot(N_vec, B_vec_par) / np.linalg.norm(N_vec_par) / np.linalg.norm(B_vec))
#        print(np.rad2deg(angle), np.dot(N_vec, B_vec_par), np.rad2deg(theta_check))
#        B_vec_new = rotate_vec_around_axis(B_vec_par, rot_base, -phi[i])
#        phi_check = np.arccos(np.dot(N_vec, B_vec_new) / np.linalg.norm(N_vec) / np.linalg.norm(B_vec_new))
#        print(np.round(np.rad2deg(phi[i]), 2), np.round(np.rad2deg(phi_check), 2))
#        transmittance_B_rot[i] = em_abs_obj.get_filter_transmittance(omega, X, Y, mode, x_vec, N_vec, B_vec_new, x_launch)
#        new_transmittance_B_rot[i] = em_abs_obj.get_filter_transmittance_new(omega, X, Y, mode, x_vec, N_vec, B_vec_new)
#    plt.plot(np.rad2deg(phi + np.pi / 2.0), transmittance)
#    plt.plot(np.rad2deg(phi + np.pi / 2.0), new_transmittance)
#    plt.plot(np.rad2deg(phi + np.pi / 2.0), transmittance_B_rot, "--")
#    plt.plot(np.rad2deg(phi + np.pi / 2.0), new_transmittance_B_rot, ":")
    print("Max new:", np.max(new_transmittance))
    contour_new = plt.contourf(np.rad2deg(PhiCoords), np.rad2deg(ThetaCoords), new_transmittance, levels=np.linspace(0.0, 1.0, 21), \
                               cmap=plt.cm.get_cmap("plasma"))
    plt.gca().set_xlabel(r"$\phi$ $[^\circ]$")
    plt.gca().set_ylabel(r"$\theta$ $[^\circ]$")
    plt.gcf().colorbar(contour_new, ax=plt.gca())
    plt.figure()
    contour_old = plt.contourf(np.rad2deg(PhiCoords), np.rad2deg(ThetaCoords), transmittance, levels=np.linspace(0.0, 1.0, 21), \
                               cmap=plt.cm.get_cmap("plasma"))
    print("Max old:", np.max(transmittance))
    plt.gca().set_xlabel(r"$\phi$ $[^\circ]$")
    plt.gca().set_ylabel(r"$\theta$ $[^\circ]$")
    plt.gcf().colorbar(contour_old, ax=plt.gca())
    plt.show()
    return transmittance, new_transmittance


def theta_dependence_tranmittance_better(folder, shot, time, dist, ich, mode_str="X", eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005):
    em_abs_obj = em_abs_Alb()
    N = 26
    t = np.linspace(-0.5, 0.5, N)
#    t = np.array([0.0])
    ThetaCoords = np.zeros((N, N))
    PhiCoords = np.zeros((N, N))
    transmittance = np.zeros((N, N))
    new_transmittance = np.zeros((N, N))
    ray_dict = read_ray_dict_from_file(folder, dist, ich, mode=mode_str)
    svec_dict, freq = read_svec_dict_from_file(folder, ich, mode=mode_str)
    rhop_ne, ne = np.loadtxt(os.path.join(folder, "ne_file.dat"), skiprows=1, unpack=True)
    launch_file = np.loadtxt(os.path.join(folder, "ray_launch.dat"), skiprows=1)
    x_launch = np.zeros(3)
    x_launch[0] = launch_file[ich - 1][2] * 1.e-2
    x_launch[1] = launch_file[ich - 1][3] * 1.e-2
    x_launch[2] = launch_file[ich - 1][4] * 1.e-2
    omega = freq * 2.0 * np.pi
#    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
#    EQ_slice = EQ_obj.GetSlice(float(time))
#    EQ_obj.add_ripple_to_slice(time, EQ_slice)
    if(mode_str == "O"):
        mode = -1
    else:
        mode = 1
#    print(em_abs_obj.test_filter_transmittance(svec_dict, ray_dict, EQ_slice, omega, mode))
    x_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
    y_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
    z_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
    Bx_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bx"])
    By_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["By"])
    Bz_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["Bz"])
    ne_spl = InterpolatedUnivariateSpline(rhop_ne, np.log(ne / 1.e19))
    x_vec = np.zeros(3)
    R_vec = np.zeros(3)
    B_vec = np.zeros(3)
    N_vec = np.zeros(3)
    boundary = 1.05
    rhop_spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["rhop"] - boundary)
    s_sep = np.max(rhop_spl.roots())
    x_vec[0] = x_spl(s_sep)
    x_vec[1] = y_spl(s_sep)
    x_vec[2] = z_spl(s_sep)
#    N_vec = np.array([1.0, 0.0, 0.0])
#    B_vec = np.array([0.0, -2.1, 0.0])
    N_vec[:] = x_launch[:]
    N_vec /= np.linalg.norm(N_vec)
    B_vec[0] = Bx_spl(s_sep)
    B_vec[1] = By_spl(s_sep)
    B_vec[2] = Bz_spl(s_sep)
    N_vec_norm = np.copy(N_vec) / np.linalg.norm(N_vec)
    B_vec_norm = np.copy(B_vec) / np.linalg.norm(B_vec)
    ne = 1.e19 * np.exp(ne_spl(boundary))
    R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
    R_vec[1] = np.arctan2(x_vec[1], x_vec[0])
    R_vec[2] = x_vec[2]
    omega_c = cnst.e / cnst.m_e * np.sqrt(np.sum(B_vec ** 2))
    X = cnst.e ** 2 * ne / (cnst.m_e * cnst.epsilon_0 * omega ** 2)
    Y = omega_c / omega
    N_vec_perp = N_vec_norm - np.dot(N_vec_norm, B_vec_norm) * B_vec_norm  # Create an N_vec that is perpendicular to B
    x1 = np.cross(N_vec_perp, B_vec_norm)
    x2 = B_vec_norm
    print(np.dot(N_vec_perp, x1))
    print(np.dot(N_vec_perp, x2))
    perp_N_comp = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            N_vec_cur = N_vec_perp * np.sqrt(1.0 - t[i] ** 2 - t[j] ** 2) + t[i] * x1 + t[j] * x2
            perp_N_comp[j] = N_vec_cur - N_vec_perp * np.dot(N_vec_cur, N_vec_perp)
            theta_check = np.arccos(np.round(np.dot(N_vec_cur, B_vec_norm), 7))
            phi_check = np.arccos(np.round(np.dot(N_vec_cur, x1), 7))
            PhiCoords[i, j] = phi_check
            ThetaCoords[i, j] = theta_check
            transmittance[i, j] = em_abs_obj.get_filter_transmittance_reverse_correct_filter(omega, X, Y, mode, x_vec, N_vec_cur, B_vec, x_launch)
            # em_abs_obj.get_filter_transmittance(omega, X, Y, mode, x_vec, N_vec_cur, B_vec, x_launch)
            new_transmittance[i, j] = em_abs_obj.get_filter_transmittance_correct_filter(omega, X, Y, mode, x_vec, N_vec_cur, B_vec, x_launch)
#            em_abs_obj.get_filter_transmittance_new(omega, X, Y, mode, x_vec, N_vec_cur, B_vec)
#            plt.plot(np.rad2deg(np.arccos(np.dot(perp_N_comp, x1))), np.rad2deg(np.arccos(np.dot(perp_N_comp, x2))), "+k")
#    plt.show()
    print("Max new:", np.max(new_transmittance))
    contour_new = plt.contourf(np.rad2deg(PhiCoords), np.rad2deg(ThetaCoords), new_transmittance, levels=np.linspace(0.0, 1.0, 21), \
                               cmap=plt.cm.get_cmap("plasma"))
    plt.gca().set_xlabel(r"$\phi$ $[^\circ]$")
    plt.gca().set_ylabel(r"$\theta$ $[^\circ]$")
    plt.gcf().colorbar(contour_new, ax=plt.gca())
    plt.figure()
    contour_old = plt.contourf(np.rad2deg(PhiCoords), np.rad2deg(ThetaCoords), transmittance, levels=np.linspace(0.0, 1.0, 21), \
                               cmap=plt.cm.get_cmap("plasma"))
    print("Max old:", np.max(transmittance))
    plt.gca().set_xlabel(r"$\phi$ $[^\circ]$")
    plt.gca().set_ylabel(r"$\theta$ $[^\circ]$")
    plt.gcf().colorbar(contour_old, ax=plt.gca())
    plt.show()
    return transmittance, new_transmittance

def test_transmittance():
    abs_obj = em_abs_Alb()
    abs_obj.get_filter_transmittance_new(105.e9 * 2.0 * np.pi, 0.001, 0.4, 1, np.array([1.0, 0, 0]), np.array([0.9, 0.1, 0.1]), np.array([-0.1, 0.9, 0.0]))

def polarization_vector_behavior():
    X = np.logspace(-8, -1.0, 100)
    Y = 0.4
    N_vec = np.array([0.8, 0.2, 0.0])
    N_vec /= np.linalg.norm(N_vec)
    B_vec = np.array([0.01450288, -2.13216371, 0.31611364])
    B_abs = np.linalg.norm(B_vec)
    theta = np.arccos(np.dot(N_vec, B_vec) / B_abs)
    print(np.rad2deg(theta))
    mode = 1
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    e = []
    N = []
    for X_val in X:
        N_val, e_val = N_with_pol_vec(X_val, Y, sin_theta, cos_theta, mode)
        N.append(N_val)
        e.append(e_val)
    e = np.array(e)
    e_x = N_vec - cos_theta * B_vec / B_abs
    e_x = e_x / np.sqrt(sum(e_x ** 2))
    e_z = B_vec / B_abs
    # e_y = e_z x e_x
    e_y = np.cross(e_x, e_z)
    pol_vector_lab = np.outer(e.T[0], e_x) + \
                     np.outer(e.T[1], e_y) + \
                     np.outer(e.T[2], e_z)
    plt.semilogx(X, np.abs(np.dot(pol_vector_lab, N_vec)))
    plt.semilogx(X, np.abs(np.dot(pol_vector_lab, B_vec / B_abs)), "--")
    plt.semilogx(X, np.abs(np.dot(pol_vector_lab, np.cross(N_vec, B_vec / B_abs))), ":")
    plt.figure()
    plt.semilogx(X, np.abs(e.T[0]))
    plt.semilogx(X, np.abs(e.T[1]), "--")
    plt.semilogx(X, np.abs(e.T[2]), ":")
    plt.show()

def plot_delta_f(folder, rhop_in):
    RELAX_dist = load_f_from_ASCII(os.path.join(folder, "fRe"), rhop_in)
    f_RELAX = RELAX_dist.f_cycl[0]
    Te = RELAX_dist.Te[0]
    Fe_th = np.zeros(f_RELAX.shape)
    for i in range(len(RELAX_dist.ull)):
        Fe_th[i, :] = Juettner2D(RELAX_dist.ull[i], RELAX_dist.uxx, Te)
    mask = Fe_th > 1.e-2
    plt.imshow(((f_RELAX[mask] - Fe_th[mask]) / Fe_th[mask]).T)
#    i_ull_0 = np.argmin(np.abs(RELAX_dist.ull))
#    mask = Fe_th[i_ull_0] > 1.e-2
#    plt.plot(RELAX_dist.uxx[mask], (f_RELAX[i_ull_0][mask] - Fe_th[i_ull_0][mask]) / Fe_th[i_ull_0][mask])
#    plt.plot(RELAX_dist.uxx, Fe_th[i_ull_0][Fe_th[i_ull_0] > 1.e-2])
    plt.show()

def benchmark_rad_reac_force(folder, rhop_in, shot, time, EQ_Exp, EQ_diag, EQ_ed, bt_vac_correction):
    EQ_obj = EQData(int(shot), EQ_exp=EQ_Exp, EQ_diag=EQ_diag, EQ_ed=int(EQ_ed), bt_vac_correction=bt_vac_correction)
    EQ_slice = EQ_obj.GetSlice(float(time))
    B_tot_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, np.sqrt(EQ_slice.Br ** 2 + EQ_slice.Bt ** 2 + EQ_slice.Bz ** 2))
    s, cont = EQ_obj.get_Rz_contour(time, rhop_in)
    B_tot_spl = InterpolatedUnivariateSpline(s, B_tot_spl(cont.T[0], cont.T[1], grid=False))
    av_B = B_tot_spl.integral(0, s[-1]) / s[-1]
    print(av_B)
    RELAX_dist = load_f_from_ASCII(os.path.join(folder, "fRe"))
    i_relax = np.argmin(np.abs(rhop_in - RELAX_dist.rhop))
    rhop_RELAX = RELAX_dist.rhop[i_relax]
    f_RELAX = RELAX_dist.f[i_relax]
    Te_spl = InterpolatedUnivariateSpline(RELAX_dist.rhop_1D_profs, RELAX_dist.Te_init)
    ne_spl = InterpolatedUnivariateSpline(RELAX_dist.rhop_1D_profs, RELAX_dist.ne_init)
    Te = Te_spl(rhop_RELAX)
    ne = ne_spl(rhop_RELAX)
    print("RELAX ne", RELAX_dist.ne[i_relax] / RELAX_dist.ne_init[i_relax])
    Fe_ana = np.zeros(f_RELAX.shape)
    Fe_th = np.zeros(f_RELAX.shape)
    Fe_ana_cycl = np.zeros(RELAX_dist.f_cycl[0].shape)
    for i in range(len(RELAX_dist.u)):
        Fe_ana[i, :] = Juettner2D(RELAX_dist.u[i], 0.0, Te)
        Fe_th[i, :] = Juettner2D(RELAX_dist.u[i], 0.0, Te)
        if(RELAX_dist.u[i] > 0.05):
            g0, g2, f = SynchrotonDistribution(RELAX_dist.u[i], np.cos(RELAX_dist.pitch), Te, ne, av_B, 1.5)
            Fe_ana[i, :] *= (1.0 + f)
    remap_f_Maj(RELAX_dist.u, RELAX_dist.pitch, [Fe_ana], 0, RELAX_dist.ull, RELAX_dist.uxx, Fe_ana_cycl, 1, 1, LUKE=True)
    norm_ana, Te_ana = get_0th_and_2nd_moment(RELAX_dist.ull, RELAX_dist.uxx, Fe_ana_cycl)
    print("norm_ana", norm_ana)
    j_slice = np.argmin(np.abs(RELAX_dist.pitch))
    u = RELAX_dist.u
    Fe_th = Fe_th[:, j_slice]
    Fe_ana = Fe_ana[:, j_slice][Fe_th > 1.e-10]
    f_RELAX = f_RELAX[:, j_slice][Fe_th > 1.e-10]
    u = u[Fe_th > 1.e-10]
    Fe_th = Fe_th[Fe_th > 1.e-10]
    fig = plt.figure(figsize=(8.5, 8.5))
    plt.plot(u, 1.e2 * (Fe_ana / norm_ana - Fe_th) / Fe_th, "-", label="Analytical solution")
    plt.plot(u, 1.e2 * (f_RELAX * RELAX_dist.ne_init[i_relax] / RELAX_dist.ne[i_relax] - Fe_th) / Fe_th, "--", label="RELAX")
    plt.gca().set_xlabel(r"$u_\perp$")
    plt.gca().set_ylabel(r"$\frac{f_1 - f_0}{f_0}\,[\si{\percent}]$")
    leg = plt.legend()
    leg.draggable()
    plt.tight_layout()
    plt.show()  # semilogy

def g0_g2_rad_reac(alpha, Te=8000):
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111)
    u = np.linspace(0.00, 1.5, 200)
    Fe_th = np.zeros((len(u), len(u) / 2))
    for i in range(len(u)):
        Fe_th[i, :] = Juettner2D(u[i], 0.0, Te)
    ax.plot(u, Fe_th.T[0] * (g0_approx(alpha, u)), "-r", label=r"$g_0$")
#    ax.plot(u, g2_approx(alpha, u), "--b", label=r"$g_2$ for $Z=1$")
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$g_n$")
    ax.get_xaxis().set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    ax.get_xaxis().set_minor_locator(MaxNLocator(nbins=10))
    ax.get_yaxis().set_major_locator(MaxNLocator(nbins=5))
    ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=10))
    ax.set_xlim(0.0, 1.5)
    leg = ax.legend()
    leg.draggable()
    plt.tight_layout()
    plt.show()

def compare_thetas(folder, shot, time, dist, ich, mode_str="X", eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005):
    ray_dict = read_ray_dict_from_file(folder, dist, ich, mode=mode_str)
    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
    EQ_slice = EQ_obj.GetSlice(float(time))
    EQ_obj.add_ripple_to_slice(time, EQ_slice)
    B_r_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.Br)
    B_t_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.Bt)
    B_z_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.Bz)
    R = np.zeros(len(ray_dict["s"]))
    theta_calc = np.zeros(len(ray_dict["s"]))
    theta_ECRad_B = np.zeros(len(ray_dict["s"]))
    R_vec = np.zeros(3)
    B_vec = np.zeros(3)
    B_vec_ECRad = np.zeros(3)
    N_vec = np.zeros(3)
    B_x_vec = np.zeros(3)
    for i in range(len(theta_calc)):
        if(ray_dict["rhop"][i] != -1.0):
            R_vec[0] = np.sqrt(ray_dict["x"][i] ** 2 + ray_dict["y"][i] ** 2)
            R_vec[1] = np.arctan2(ray_dict["y"][i], ray_dict["x"][i])
            R_vec[2] = ray_dict["z"][i]
            R[i] = R_vec[0]
            B_vec[0] = B_r_spl(R_vec[0], R_vec[2], grid=False)
            B_vec[1] = B_t_spl(R_vec[0], R_vec[2], grid=False)
            B_vec[2] = B_z_spl(R_vec[0], R_vec[2], grid=False)
            B_vec += EQ_slice.ripple.get_ripple(R_vec)
            B_abs = np.sqrt(np.sum(B_vec ** 2))
            B_vec[:] /= B_abs
            B_x_vec[0] = B_vec[0] * np.cos(R_vec[1]) - B_vec[1] * np.sin(R_vec[1])
            B_x_vec[1] = B_vec[0] * np.sin(R_vec[1]) + B_vec[1] * np.cos(R_vec[1])
            B_x_vec[2] = B_vec[2]
            N_vec[0] = ray_dict["Nx"][i]
            N_vec[1] = ray_dict["Ny"][i]
            N_vec[2] = ray_dict["Nz"][i]
            N_abs = np.sqrt(np.sum(N_vec ** 2))
            N_vec[:] /= N_abs
            B_vec_ECRad[0] = ray_dict["Bx"][i]
            B_vec_ECRad[1] = ray_dict["By"][i]
            B_vec_ECRad[2] = ray_dict["Bz"][i]
            B_vec_ECRad[:] /= np.sqrt(np.sum(B_vec_ECRad ** 2))
            theta_calc[i] = np.arccos(np.sum(B_x_vec * N_vec))
            theta_ECRad_B[i] = np.arccos(np.sum(B_vec_ECRad * N_vec))
    plt.plot(np.sqrt(ray_dict["x"] ** 2 + ray_dict["y"] ** 2), np.rad2deg(ray_dict["theta"]), label=r"ECRad")
    plt.plot(R, np.rad2deg(theta_calc), "--", label=r"Python")
    plt.plot(R, np.rad2deg(theta_ECRad_B), "--", label=r"Python with ECRad B-Field")
    plt.gca().set_xlabel(r"$R$ \si{\metre}")
    plt.gca().set_ylabel(r"$\theta$ $[^\circ]$")
    plt.legend()
    plt.show()
    return theta_calc

def compare_channel_thetas(folder, ch_list, dist, color_list, linestyles, mode_str="X", labels=None):
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111)
    if(labels is None):
        labels = []
        for ch in ch_list:
            labels.append(r"channel" + " {0:d}".format(ch))
    for ch, label, color, linestyle in zip(ch_list, labels, color_list, linestyles):
        ray_dict = read_ray_dict_from_file(folder, dist, ch, mode=mode_str)
        success, s, R, z, rhop = find_cold_res(folder, ch, mode=mode_str, harmonic_number=2)
        ax.plot(np.sqrt(ray_dict["x"] ** 2 + ray_dict["y"] ** 2), ray_dict["N_cold"] * np.cos(ray_dict["theta"]), color=color, label=label, linestyle=linestyle)
        ax.vlines(R, 0.0, 1.2 , linestyle=":", color=color)
    ax.set_xlabel(r"$R$ [\si{\metre}]")
    ax.set_ylabel(r"$N_\parallel$")
    ax.set_ylim(0, 1.2)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return


def Faraday_stuff(folder, shot, time, dist, ich, eq_exp="AUGD", eq_diag="EQH", eq_ed=0, bt_vac_correction=1.005):
    Npts = 200
    ray_dict_X = read_ray_dict_from_file(folder, dist, ich, mode="X")
    svec_dict_X, freq = read_svec_dict_from_file(folder, ich, mode="X")
    ray_dict_O = read_ray_dict_from_file(folder, dist, ich, mode="O")
    svec_dict_O, freq = read_svec_dict_from_file(folder, ich, mode="O")
    svec_dict = [svec_dict_X, svec_dict_O]
    ray_dict = [ray_dict_X, ray_dict_O]
    rhop_ne, ne = np.loadtxt(os.path.join(folder, "ne_file.dat"), skiprows=1, unpack=True)
    launch_file = np.loadtxt(os.path.join(folder, "ray_launch.dat"), skiprows=1)
    make_perp = False
    x_launch = np.zeros(3)
    x_launch[0] = launch_file[ich - 1][2] * 1.e-2
    x_launch[1] = launch_file[ich - 1][3] * 1.e-2
    x_launch[2] = launch_file[ich - 1][4] * 1.e-2
    omega = freq * 2.0 * np.pi
#    EQ_obj = EQData(int(shot), EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=int(eq_ed), bt_vac_correction=bt_vac_correction)
#    EQ_slice = EQ_obj.GetSlice(float(time))
#    EQ_obj.add_ripple_to_slice(time, EQ_slice)
#    print(em_abs_obj.test_filter_transmittance(svec_dict, ray_dict, EQ_slice, omega, mode))
    N_abs_X = np.zeros(Npts)
    N_abs_O = np.zeros(Npts)
    s = np.zeros(Npts)
    hutch_pol_integrand = np.zeros(Npts)
    boundary = np.linspace(1.05, 1.15, Npts)
    x_spl = []
    y_spl = []
    z_spl = []
    Nx_spl = []
    Ny_spl = []
    Nz_spl = []
    Bx_spl = []
    By_spl = []
    Bz_spl = []
    ne_spl = []
    for mode in [0, 1]:
        x_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["x"]))
        y_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["y"]))
        z_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["z"]))
        Nx_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["Nx"]))
        Ny_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["Ny"]))
        Nz_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["Nz"]))
        Bx_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["Bx"]))
        By_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["By"]))
        Bz_spl.append(InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["Bz"]))
        ne_spl.append(InterpolatedUnivariateSpline(rhop_ne, np.log(ne / 1.e19)))
    x_vec = np.zeros(3)
    R_vec = np.zeros(3)
    B_vec = np.zeros(3)
    N_vec = np.zeros(3)
    for i in range(len(boundary)):
        for mode in [0, 1]:
            rhop_spl = InterpolatedUnivariateSpline(ray_dict[mode]["s"], ray_dict[mode]["rhop"] - boundary[i])
            s_sep = np.max(rhop_spl.roots())
            x_vec[0] = x_spl[mode](s_sep)
            x_vec[1] = y_spl[mode](s_sep)
            x_vec[2] = z_spl[mode](s_sep)
            N_vec[0] = Nx_spl[mode](s_sep)
            N_vec[1] = Ny_spl[mode](s_sep)
            N_vec[2] = Nz_spl[mode](s_sep)
            B_vec[0] = Bx_spl[mode](s_sep)
            B_vec[1] = By_spl[mode](s_sep)
            B_vec[2] = Bz_spl[mode](s_sep)
            ne = 1.e19 * np.exp(ne_spl[mode](boundary[i]))
            R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
            R_vec[1] = np.arctan2(x_vec[1], x_vec[0])
            R_vec[2] = x_vec[2]
            omega_c = cnst.e / cnst.m_e * np.sqrt(np.sum(B_vec ** 2))
            X = cnst.e ** 2 * ne / (cnst.m_e * cnst.epsilon_0 * omega ** 2)
            Y = omega_c / omega
            phi = np.arctan2(x_vec[1], x_vec[0])
            B_pol = B_vec[2]  # np.sqrt((B_vec[0] * np.cos(phi) + B_vec[1] * np.sin(phi)) ** 2 + B_vec[2] ** 2)
            B_tor = B_vec[1] * np.cos(phi) - B_vec[0] * np.sin(phi)
            if(make_perp):
                # Eliminate E_r
                B_vec[0] = -B_tor * np.sin(R_vec[1])
                B_vec[1] = +B_tor * np.cos(R_vec[1])
                B_vec[2] = B_pol
                # Make N perpendicular to B for testing purposes
                N_vec = N_vec / np.linalg.norm(N_vec) - np.dot(N_vec / np.linalg.norm(N_vec), B_vec / np.linalg.norm(B_vec)) * B_vec / np.linalg.norm(B_vec)
                N_vec /= np.linalg.norm(N_vec)
            cos_theta = np.dot(N_vec, B_vec) / (np.linalg.norm(N_vec) * np.linalg.norm(B_vec))
            hutch_pol_integrand[i] = X * cos_theta
            if(mode == 0):
                s[i] = s_sep
                N_abs_X[i], pol_vec_X = N_with_pol_vec(X, Y, np.sin(np.arccos(cos_theta)), cos_theta, 1)
            else:
                N_abs_O[i], pol_vec_O = N_with_pol_vec(X, Y, np.sin(np.arccos(cos_theta)), cos_theta, -1)
    delta_phi = (N_abs_X - N_abs_O) * omega / cnst.c
    delta_phi_spl = InterpolatedUnivariateSpline(s, delta_phi)
    hutch_pol_integrand_spl = InterpolatedUnivariateSpline(s, hutch_pol_integrand)
    alpha = np.zeros(Npts)
    Hutch_alpha = np.zeros(Npts)
    for i in range(Npts):
        alpha[i] = delta_phi_spl.integral(s[0], s[i])
        Hutch_alpha[i] = cnst.e / (2.e0 * cnst.m_e * cnst.c) * hutch_pol_integrand_spl.integral(s[0], s[i])
#    plt.plot(boundary, N_abs_X, label=r"$X$-mode")
#    plt.plot(boundary, N_abs_O, "--", label=r"$O$-mode")
    plt.plot(boundary, np.rad2deg(Hutch_alpha), "--", label=r"Faraday rotation approx.")
    plt.plot(boundary, np.rad2deg(alpha), "--", label=r"Faraday rotation precise")
    plt.gca().set_xlabel(r"$\rho_\mathrm{pol}$ boundary")
    plt.gca().set_ylabel(r"$\alpha$ $[^\circ]$")
    plt.gca().legend()
    plt.show()
    return

def calculate_radiation_loss():
    u_par = 0.01
    u_perp = 0.75
    gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
    omega_c = 2.0 * np.pi * 65.e9
    omega_p = np.sqrt(cnst.e ** 2 * 4.5e19 / (cnst.epsilon_0 * cnst.m_e))
    em_abs_obj = em_abs_Alb()
#    print(em_abs_obj.integrated_emissivity(u_par, u_perp, omega_c, omega_p))
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, 200)
    mode = 1
    n = 2
    omega_test = em_abs_obj.eval_omega(u_par, gamma, omega_c, 0, n)
    N_abs, f, a_sq, b_sq = em_abs_obj.abs_Al_N_with_pol_coeff(omega_test, omega_p ** 2 / omega_test ** 2, omega_c / omega_test, 1.0, 0.0, mode)
    if(N_abs <= 0):
        print("Emission for current harmonic is in cut-off")
        return
#    omega = em_abs_obj.eval_omega(u_par, gamma, omega_c, np.cos(theta) * N_abs_test, n) / (1.e9 * 2.0 * np.pi)
#    plt.plot(np.rad2deg(theta), omega)
    omega_X = np.zeros(len(theta))
    omega_O = np.zeros(len(theta))
    for i in range(len(theta)):
        omega_X[i] = em_abs_obj.calc_omega(u_par, gamma, theta[i], omega_c, omega_p, mode, n)
        omega_O[i] = em_abs_obj.calc_omega(u_par, gamma, theta[i], omega_c, omega_p, -mode, n)
    X = omega_p ** 2 / omega_O ** 2
    Y = omega_c / omega_O
    a, b = em_abs_obj.a_sq_frac(omega_O, X, Y, np.sin(theta), np.cos(theta), mode)
    plt.semilogy(theta, a)
    plt.semilogy(theta, b, "--")
    plt.show()
#    plt.plot(np.rad2deg(theta), omega_X / (1.e9 * 2.0 * np.pi), "-")
#    plt.plot(np.rad2deg(theta), omega_O / (1.e9 * 2.0 * np.pi), "--")
#    j = em_abs_obj.single_electron_emissivity(theta, args=[u_par, u_perp, omega_c, omega_p])
#    plt.plot(np.rad2deg(theta), j)
    plt.show()


def poject_full_wave_E_to_modes(full_wave_path, f_ECE, theta_pol, phi_tor, average=False):
    omega = 2.0 * np.pi * f_ECE
    waves = h5py.File(full_wave_path, 'r')
    E_x = waves['E_R'].value
    E_y = waves['E_tor'].value
    E_z = waves['E_z'].value
#    R_wave = np.linspace(1.492, 2.510, #waves['R'].value
#    -0.408 0.408
    z_wave = np.zeros(len(R_wave))
#    if(average):
#        E_x_spl = InterpolatedUnivariateSpline(R_wave, E_x)
#        E_z_spl = InterpolatedUnivariateSpline(R_wave, E_y, k=4)
#        E_z_spl = InterpolatedUnivariateSpline(R_wave, E_z)
#    EQ_obj = EQDataExt(32934, external_folder=Ext_data_path)
#    EQ_obj.read_EQ_from_Ext()
#    EQ_slice = EQ_obj.slices[0]
    ne = np.loadtxt(os.path.join(Ext_data_path, "ne0"))
    ne_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, ne)
    Br_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.Br.T)
    Bt_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.Bt.T)
    Bz_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.Bz.T)
    rhop_spl = RectBivariateSpline(EQ_slice.R, EQ_slice.z, EQ_slice.rhop.T)
    X_frac = np.zeros(R_wave.shape)
    O_frac = np.zeros(R_wave.shape)
    em_abs_obj = em_abs_Alb()
    x_vec = np.zeros(3)
    N_vec = np.zeros(3)
    B_vec = np.zeros(3)
    E_vec = np.zeros(3)
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512, sharex=ax1)
    ax3 = fig.add_subplot(513, sharex=ax2)
    ax4 = fig.add_subplot(514, sharex=ax3)
    ax5 = fig.add_subplot(515, sharex=ax4)
    # for j in range(len(z_wave)):
    sep_root_spl = InterpolatedUnivariateSpline(R_wave, rhop_spl(R_wave, z_wave, grid=False) - 1.0)
    R_sep = sep_root_spl.roots()[0]
    for i in range(len(R_wave)):
        j = i
        x_vec[0] = R_wave[i]  # Pure x component
        x_vec[1] = z_wave[j]
        E_R_vec = np.array([E_R[i], E_tor[i], E_z[i]])
        B_R_vec = np.array([Br_spl(R_wave[i], z_wave[j], grid=False), Bt_spl(R_wave[i], z_wave[j], grid=False), Bz_spl(R_wave[i], z_wave[j], grid=False)])
        # Convert B to Carthesian Coordinates
        phi = np.arctan2(x_vec[1], x_vec[0])
        B_vec[0] = np.cos(phi) * B_R_vec[0] - np.sin(phi) * B_R_vec[1]
        B_vec[1] = np.sin(phi) * B_R_vec[0] + np.cos(phi) * B_R_vec[1]
        B_vec[2] = B_R_vec[2]
        E_vec[0] = np.cos(phi) * E_R_vec[0] - np.sin(phi) * E_R_vec[1]
        E_vec[1] = np.sin(phi) * E_R_vec[0] + np.cos(phi) * E_R_vec[1]
        E_vec[2] = E_R_vec[2]
        E_vec /= np.linalg.norm(E_vec)
        X = cnst.e ** 2 * ne_spl(R_wave[i], z_wave[j], grid=False) / (cnst.epsilon_0 * cnst.m_e * omega ** 2)
        Y = cnst.e * np.linalg.norm(B_vec) / (cnst.m_e * omega)
        N_vec[:] = -x_vec / np.linalg.norm(x_vec)
        temp1 = np.arctan2(N_vec[2], N_vec[1])
        N_vec[0] = 1.e0
        N_vec[1] = temp1
        # Poloidal angle is aligned with R direction
        N_vec[2] = np.pi / 2.e0
        N_vec[1] += phi_tor
        N_vec[2] += theta_pol
        temp = np.cos(N_vec[1]) * np.sin(N_vec[2])
        temp1 = np.sin(N_vec[1]) * np.sin(N_vec[2])
        temp2 = np.cos(N_vec[2])
        N_vec[0] = temp
        N_vec[1] = temp1
        N_vec[2] = temp2
        N_vec /= np.linalg.norm(N_vec)
        pol_vec_X = em_abs_obj.get_pol_vec_carth(omega, X, Y, 1, N_vec, B_vec)
        pol_vec_O = em_abs_obj.get_pol_vec_carth(omega, X, Y, -1, N_vec, B_vec)
        X_frac[i] = np.abs(np.vdot(pol_vec_X, E_vec)) ** 2
        O_frac[i] = np.abs(np.vdot(pol_vec_O, E_vec)) ** 2
    ax1.plot(R_wave, E_R)
    ax1.vlines(R_sep, np.min(E_R), np.max(E_R), color="k", linestyle="--")
    ax1.set_ylabel(r"$E_\mathrm{r} \, [\mathrm{a.u,}]$")
    ax2.plot(R_wave, E_tor)
    ax2.vlines(R_sep, np.min(E_tor), np.max(E_tor), color="k", linestyle="--")
    ax2.set_ylabel(r"$E_\mathrm{t} \, [\mathrm{a.u,}]$")
    ax3.plot(R_wave, E_z)
    ax3.vlines(R_sep, np.min(E_z), np.max(E_z), color="k", linestyle="--")
    ax3.set_ylabel(r"$E_\mathrm{z} \, [\mathrm{a.u,}]$")
    ax4.plot(R_wave, X_frac)
    ax4.vlines(R_sep, np.min(E_z), np.max(E_z), color="k", linestyle="--")
    ax4.set_ylabel(r"$X_\mathrm{frac}$")
    ax5.plot(R_wave, O_frac)
    ax5.set_ylabel(r"$O_\mathrm{frac}$")
    ax5.vlines(R_sep, np.min(E_z), np.max(E_z), color="k", linestyle="--")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax5.set_xlabel(r"$R\, [\si{\metre}]$")
    plt.show()

def compare_BPD_to_Teperp(Te):
    u_par = np.linspace(-1, 1, 100)
    u_perp = np.linspace(0, 1, 50)
    Trad = []
    for u_perp_val in u_perp:
        Trad.append(Juettner2D(u_par, u_perp_val, Te) * u_perp_val ** 2)
    Trad = np.array(Trad)
    Trad[Trad < 1.e-20] = 1.e-20
    plt.contourf(u_perp, u_par, np.log10(Trad).T)
    plt.gca().set_aspect("equal")
    plt.show()

def test_IO(working_dir):
    rhop_X = np.loadtxt(os.path.join(working_dir, "X_TRadM_therm.dat")).T[0]
    rhop_O = np.loadtxt(os.path.join(working_dir, "O_TRadM_therm.dat")).T[0]
    I_X = np.loadtxt(os.path.join(working_dir, "X_TRadM_therm.dat")).T[1]
    I_O = np.loadtxt(os.path.join(working_dir, "O_TRadM_therm.dat")).T[1]
    T_X = np.loadtxt(os.path.join(working_dir, "X_TRadM_therm.dat")).T[2]
    T_O = np.loadtxt(os.path.join(working_dir, "O_TRadM_therm.dat")).T[2]
    reflec_X = 0.8
    reflec_O = 0.92
    mode_conv = 0.2
    vessel_plasma_ratio = 59.4 / 39.8
    make_I0(rhop_X, rhop_O, I_X, I_O, T_X, T_O, reflec_X, reflec_O, mode_conv, vessel_plasma_ratio)

def make_I0(rhop_X, rhop_O, I_X, I_O, T_X, T_O, reflec_X, reflec_O, mode_conv, vessel_plasma_ratio):
    I0_X = -((I_O * mode_conv + I_X * (1.e0 + vessel_plasma_ratio - reflec_O - T_O)) / \
                       (mode_conv ** 2 - (-1.e0 - vessel_plasma_ratio + reflec_O + T_O) * (-1.e0 - vessel_plasma_ratio + reflec_X + T_X)))
    I0_O = -((I_O + vessel_plasma_ratio * I_O + I_X * mode_conv - I_O * reflec_X - I_O * T_X) / \
            (-1.e0 - 2 * vessel_plasma_ratio - vessel_plasma_ratio ** 2 + reflec_O + vessel_plasma_ratio * reflec_O + \
            mode_conv ** 2 + reflec_X + vessel_plasma_ratio * reflec_X - reflec_O * reflec_X + \
            T_O + vessel_plasma_ratio * T_O - reflec_X * T_O + T_X + vessel_plasma_ratio * T_X - reflec_O * T_X - T_O * T_X))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    figO = plt.figure()
    axO = figO.add_subplot(111)
    ax.plot(rhop_X, I_X, "^")
    ax.plot(rhop_X, I0_X, "+")
    axO.plot(rhop_O, I_O, "^")
    axO.plot(rhop_O, I0_O, "+")
    plt.show()


if __name__ == "__main__":
    repair_ECRad_results("/tokp/work/sdenk/DRELAX_Results")
#    plot_delta_f("/tokp/work/sdenk/nssf/33697/4.80/OERT/ed_0/ecfm_data/", 0.15)
#    compare_BPD_to_Teperp(1000)
#    exampl_reso([120.e9, 120.e9], [70.e9, 70.e9], [0.99 * np.pi / 2.e0, np.pi / 4.e0], 8.e3, 5.e18, n=2)
#    make_iso_flux("/ptmp1/work/sdenk/nssf/32740/5.96/OERT/ecfm_data/", 32740, 5.964)
#    check_BPD_norm("/ptmp1/work/sdenk/ECRad7/ecfm_data/IchTB/BPD_ray001ch037_X.dat", n1=5, n2=6)  #  /ptmp1/work/sdenk/nssf/32740/5.07/OERT/ed_2
#    check_BPD_norm("/ptmp1/work/sdenk/ECRad7/ecfm_data/IchTB/BPDX042.dat")
#    Energy_los(0.4, B=2.5)
#    fast_ECE_2D_plot(31539, [2.8161, 2.8164])
#    refr_index_coupling(5.818838678157487E+019, 502654824574.367, 259604452669.565 / np.pi, +1)
#    compare_thetas("/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_4/ecfm_data/", 32934, 3.298, "TB", 42)
#    boundary_dependence_tranmittance("/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_4/ecfm_data/", 32934, 3.298, "TB", 1)
#    polarization_vector_behavior()
#    test_transmittance()
#    g0_g2_rad_reac(1.76)
#    test_IO("/tokp/work/sdenk/ECRad2/ecfm_data/")
#    benchmark_rad_reac_force("/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_16/ecfm_data/", 0.2, 31539, 2.814, "AUGD", "EQH", 0, 1.005)
#    compare_channel_thetas("/tokp/work/sdenk/ECRad/ecfm_data", [42, 70], "TB", [(0.6, 0.0, 0.0), (0.0, 0.0, 0.6)], ["-", "--"], labels=["radial ECE", "oblique ECE"])
#    poject_full_wave_E_to_modes("/afs/ipp/u/sdenk/Documentation/Data/full_wave_output.h5", \
#                                "/tokp/work/sdenk/ECRad/Ext_data/" , 91.93e9, np.deg2rad(1.455), np.deg2rad(2.18))
#    theta_dependence_tranmittance_better("/ptmp1/work/sdenk/ECRad/ecfm_data/", 32934, 3.298, "TB", 1, mode_str="X")  # "/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_4/ecfm_data/"
#    phi_tor_dependence_tranmittance("/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_8/ecfm_data/", 32934, 3.298, "TB", 20, mode_str="X")  #
#    boundary_dependence_tranmittance("/ptmp1/work/sdenk/ECRad/ecfm_data/", 32934, 3.298, "TB", 1)  # "/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_4/ecfm_data/"
#    polarizer_angle_dependence_tranmittance("/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_8/ecfm_data/", 32934, 3.298, "TB", 20)
#    Faraday_stuff("/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_8/ecfm_data/", 32934, 3.298, "TB", 6)
#    theta_dependence_tranmittance_better("/ptmp1/work/sdenk/nssf/32934/3.30/OERT/ed_8/ecfm_data/", 32934, 3.298, "TB", 20)
#    print(calculate_transmittance(
#    calculate_radiation_loss()
#    check_IDA_Te_curv(1000.0, 0.98, 0.02, 0.2, 1.005, 0.01, 0.05)
#    calculate_resonant_fraction("/ptmp1/work/sdenk/nssf/30907/0.73/OERT/ed_1/ecfm_data/", 23, 0.01, Te_scale=1)
#    calculate_resonant_fraction("/ptmp1/work/sdenk/nssf/30907/0.73/OERT/ed_1/ecfm_data/", 23, 0.01, Te_scale=0.5)
#    calculate_resonant_fraction("/ptmp1/work/sdenk/nssf/30907/0.73/OERT/ed_1/ecfm_data/", 23, 0.01, Te_scale=3)
#    plot_BPD_for_all_rays("/ptmp1/work/sdenk/ECRad7/ECRad_33697_EXT_ed4.mat", 4.80, [1, 2])
#    plt.show()
#    test_resonance()
#    test_resonance()
#    compare_Trad_difs_light("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                      ["ed_4", "ed_5", "ed_6", "ed_7", "ed_14", "ed_15", "ed_16"], \
#                       [r"$\tilde{T}_\mathrm{rad}$[GENE]", \
#                        r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian($T_\parallel=T_0$)]", \
#                        r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian($T_\perp=T_0$)]", \
#                        r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian]", \
#                        r"$\tilde{T}_\mathrm{rad}$[rel. BiMaxwellian($T_\parallel=T_0$)]", \
#                        r"$\tilde{T}_\mathrm{rad}$[rel. BiMaxwellian($T_\perp=T_0$)]", \
#                        r"$\tilde{T}_\mathrm{rad}$[rel. BiMaxwellian]"])
#     compare_Trad_difs_light("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                       ["ed_4", "ed_5", "ed_6", "ed_7"], \
#                        [r"$\tilde{T}_\mathrm{rad}$[GENE]", \
#                         r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian($T_\parallel=T_0$)]", \
#                         r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian($T_\perp=T_0$)]", \
#                         r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian]"])
#    compare_Trad_difs("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                      ["ed_4", "ed_23", "ed_7", "ed_24"], \
#                       [r"$\tilde{T}_\mathrm{rad}$[GENE perp.]", \
#                        r"$\tilde{T}_\mathrm{rad}$[GENE obl.]", \
#                        r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian perp.]", \
#                        r"$\tilde{T}_\mathrm{rad}$[BiMaxwellian obl.]"])
#    compare_Trad_difs("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                      ["ed_4", "ed_7", "ed_25", "ed_26"], \
#                       [r"$\tilde{T}_\mathrm{rad}$[GENE single ray radial]", \
#                        r"$\tilde{T}_\mathrm{rad}$[BiMax single ray radial]", \
#                        r"$\tilde{T}_\mathrm{rad}$[GENE single ray]", \
#                        r"$\tilde{T}_\mathrm{rad}$[GENE 36 rays]"])
#    compare_Trad_difs("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                      ["ed_25", "ed_27"], \
#                       [r"$\tilde{T}_\mathrm{rad}$[GENE w. single frequency]", \
#                        r"$\tilde{T}_\mathrm{rad}$[GENE w. 8 frequencies]"])
#    dif_vs_difs("/ptmp1/work/sdenk/ECRad7/ECRad_33585_EXT_ed6.mat", \
#                "/ptmp1/work/sdenk/ECRad6/ECRad_33585_EXT_ed 1.mat")
#    correlate(["/ptmp1/work/sdenk/ECRad7/ECRad_33585_EXT_ed6.mat", \
#               "/ptmp1/work/sdenk/ECRad6/ECRad_33585_EXT_ed 1.mat"], \
#               ["GENE", "BiMaxwellian"])
#    correlate_with_one(["/ptmp1/work/sdenk/ECRad7/ECRad_33585_EXT_ed6.mat", \
#               "/ptmp1/work/sdenk/ECRad6/ECRad_33585_EXT_ed 1.mat"], \
#               ["GENE", "BiMaxwellian"])
#    compare_Trad_difs("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                      ["ed_25", "ed_28"], \
#                       [r"$\tilde{T}_\mathrm{rad}$[GENE w. density pert.]", \
#                        r"$\tilde{T}_\mathrm{rad}$[GENE no density pert.]"])
#    Check_Trad_vs_Te_BPD("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_1/", 2, 33585, 3.0, eq_exp="AUGD", eq_diag="EQH", eq_ed=0)
#    Convolute_Te_perp_Te_par("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_1/", 2, 33585, 3.0, eq_exp="AUGD", eq_diag="EQH", eq_ed=0)
    # benchmark_absorption("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/Ecfm_Model_new")
