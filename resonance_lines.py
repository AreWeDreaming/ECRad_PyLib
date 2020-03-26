'''
Created on Dec 1, 2019

@author: sdenk
'''
from ECRad_Results import ECRadResults
from distribution_io import read_waves_mat_to_beam, load_f_from_mat
from scipy.io import loadmat
from plotting_configuration import plt, MaxNLocator
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from em_Albajar import em_abs_Alb, s_vec
from distribution_functions import Juettner2D

def plot_resonance_lines(rhop, ECRadResFile = None, wave_mat_file=None, dist_mat=None, art_launch=None, only_central=False, ECE_ch=[], colors=["k", "b"]):
    if(ECRadResFile is not None):
        Results = ECRadResults()
        Results.from_mat_file(ECRadResFile)
    if(wave_mat_file is not None):
        wave_mat = loadmat(wave_mat_file)
        linear_beam = read_waves_mat_to_beam(wave_mat, Results.Scenario.plasma_dict["eq_data"][0])
    if(dist_mat is None and art_launch is not None):
        uxx=np.linspace(0.0, 2.0, 200)
        ull=np.linspace(-2.0, 2.0,400)
        f = []
        for u_perp in uxx:
            f.append(np.log10(Juettner2D(ull, u_perp, art_launch["Te"])))
        f = np.array(f).T
    elif(dist_mat is not None):
        dist_obj = load_f_from_mat(dist_mat, use_dist_prefix=True)
        f_ind = np.argmin(np.abs(dist_obj.rhop - rhop))
        print("Rhop: ", dist_obj.rhop[f_ind])
        uxx = dist_obj.uxx
        ull = dist_obj.ull
        f = dist_obj.f_cycl_log10[f_ind]   
    else:
        raise ValueError("Either art_launch or dist_mat must not be None")
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    em_abs_alb = em_abs_Alb()
    if(wave_mat_file is not None):
        for ibeam in range(len(linear_beam.rays)):
            for iray in range(len(linear_beam.rays[ibeam])):
                if(only_central and iray != 0):
                    continue
                rhop_spline = InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["rhop"] - dist_obj.rhop[f_ind])
                R_spl =  InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["R"])
                roots = rhop_spline.roots()
                if(np.isscalar(roots)):
                    s_res_LFS = roots
                elif(len(roots) == 0):
                    continue
                else:
                    i_root = np.argmax(R_spl(roots))
                    print("R", R_spl(roots)[i_root])
                    s_res_LFS = roots[i_root]
                omega_c = InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["omega_c"])(s_res_LFS)
                N_par = InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["Npar"])(s_res_LFS)
                omega = 2.0 * np.pi * 105.e9
                omega_bar = omega / omega_c
                if(N_par ** 2 >= 1.0):
                    continue
                m_0 = np.sqrt(1.e0 - N_par ** 2) * omega_bar
                t = np.linspace(-1.0, 1.0, 60)
                for m in [2]:
                    u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                                       np.sqrt((float(m) / m_0) ** 2 - 1.e0) * t)
                    if(np.any(np.isnan(u_par))):
                        continue
                    u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - t ** 2)
                    u_perp_sq[u_perp_sq < 0] += 1.e-7
                    if(np.all(u_perp_sq >= 0)):
                        u_perp = np.sqrt(u_perp_sq)
                        ax.plot(u_perp, u_par, "--k")
    if(ECRadResFile is not None):
        itime = 0
        for ch,color in zip(ECE_ch,colors):
            ich = ch - 1
            freq = Results.Scenario.ray_launch[itime]["f"][ich]
            for iray in range(Results.Config.N_ray):
                svec = {}
                ray_index = iray - 1
                if(only_central and iray != 0):
                    continue
                if(Results.Config.N_ray == 1):
                    svec["rhop"] = Results.ray["rhopX"][itime][ich]
                    svec["s"] = Results.ray["sX"][itime][ich][svec["rhop"] != -1.0]
                    svec["R"] = np.sqrt(Results.ray["xX"][itime][ich][svec["rhop"] != -1.0] ** 2 + \
                                        Results.ray["yX"][itime][ich][svec["rhop"] != -1.0] ** 2)
                    svec["z"] = Results.ray["zX"][itime][ich][svec["rhop"] != -1.0]
                    svec["Te"] = Results.ray["TeX"][itime][ich][svec["rhop"] != -1.0]
                    svec["theta"] = Results.ray["thetaX"][itime][ich][svec["rhop"] != -1.0]
                    svec["freq_2X"] = Results.ray["YX"][itime][ich][svec["rhop"] != -1.0] * 2.0 * freq
                    svec["N_abs"] = Results.ray["NcX"][itime][ich][svec["rhop"] != -1.0]
                    svec["rhop"] = svec["rhop"][svec["rhop"] != -1.0]
                else:
                    svec["rhop"] = Results.ray["rhopX"][itime][ich][ray_index]
                    svec["s"] = Results.ray["sX"][itime][ich][ray_index][svec["rhop"] != -1.0]
                    svec["R"] = np.sqrt(Results.ray["xX"][itime][ich][ray_index][svec["rhop"] != -1.0] ** 2 + \
                                        Results.ray["yX"][itime][ich][ray_index][svec["rhop"] != -1.0] ** 2)
                    svec["z"] = Results.ray["zX"][itime][ich][ray_index][svec["rhop"] != -1.0]
                    svec["Te"] = Results.ray["TeX"][itime][ich][ray_index][svec["rhop"] != -1.0]
                    svec["theta"] = Results.ray["thetaX"][itime][ich][ray_index][svec["rhop"] != -1.0]
                    svec["freq_2X"] = Results.ray["YX"][itime][ich][ray_index][svec["rhop"] != -1.0] * 2.0 * freq
                    svec["N_abs"] = Results.ray["NcX"][itime][ich][ray_index][svec["rhop"] != -1.0]
                    svec["rhop"] = svec["rhop"][svec["rhop"] != -1.0]
                ne_spl = InterpolatedUnivariateSpline(Results.Scenario.plasma_dict["rhop_prof"][0], \
                                                      np.log(Results.Scenario.plasma_dict["ne"][0]))
                svec["ne"] = np.exp(ne_spl(svec["rhop"] ))
                rhop_root_spline = InterpolatedUnivariateSpline(svec["s"], svec["rhop"] - dist_obj.rhop[f_ind])
                R_spline = InterpolatedUnivariateSpline(svec["s"], svec["R"])
                roots = rhop_root_spline.roots()
                if(np.isscalar(roots)):
                    s_res_LFS = roots
                elif(len(roots) == 0):
                    continue
                else:
                    i_res = np.argmax(R_spline(roots))
                    s_res_LFS = roots[i_res] # Want LFS i.e. largest R
                rhop_spl = InterpolatedUnivariateSpline(svec["s"], svec["rhop"])
                Te_spl = InterpolatedUnivariateSpline(svec["s"], svec["Te"])
                ne_spl = InterpolatedUnivariateSpline(svec["s"], svec["ne"])
                freq_2X = InterpolatedUnivariateSpline(svec["s"], svec["freq_2X"])
                theta_spl = InterpolatedUnivariateSpline(svec["s"], svec["theta"])
                svec_point = s_vec(rhop_spl(s_res_LFS), Te_spl(s_res_LFS), ne_spl(s_res_LFS), \
                                   freq_2X(s_res_LFS), theta_spl(s_res_LFS))
                u_par, u_perp = em_abs_alb.abs_Albajar_resonance_line(svec_point, freq* 2.0 * np.pi, 1, 2)
                if(not np.any(np.isnan(u_perp))):
                    ax.plot(u_par, u_perp, "-" + color)
    if(art_launch is not None):
        svec_point = s_vec(art_launch["rhop"], art_launch["Te"], art_launch["ne"], \
                           art_launch["freq_2X"], art_launch["theta"])
        u_par, u_perp = em_abs_alb.abs_Albajar_resonance_line(svec_point, art_launch["freq"]* 2.0 * np.pi, 1, 2)
        if(not np.any(np.isnan(u_perp))):
            ax.plot(u_par, u_perp, "-k")    
    levels = np.linspace(-13, 5, 20)
    try:
        cmap = plt.cm.get_cmap("plasma")
    except ValueError:
        cmap = plt.cm.get_cmap("jet")
    cont1 = ax.contourf(ull, uxx, f.T, levels=levels, cmap=cmap)  # ,norm = LogNorm()
    cont2 = ax.contour(ull, uxx, f.T, levels=levels, colors='k',
                                     hold='on', alpha=0.25, linewidths=1)
    ax.set_xlabel(r"$u_{\Vert}$")
    ax.set_ylabel(r"$u_{\perp}$")
    for c in cont2.collections:
        c.set_linestyle('solid')
    cb = fig.colorbar(cont1, ax=ax, ticks=[-10, -5, 0, 5])  # ticks = levels[::6] #,
    cb.set_label(r"$\log_\mathrm{10}\left(f\right)$")  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
    cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
    cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([1.0, 5.0, 10.0])))
    cb.ax.minorticks_on()
    steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    steps_y = steps
    ax.get_yaxis().set_major_locator(MaxNLocator(nbins=2, steps=steps, prune='lower'))
    ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=4, steps=steps / 4.0))
    ax.get_xaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps_y))
    ax.get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps_y / 4.0))
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 1.125)
    ax.set_aspect("equal")
    if(art_launch is not None):
        ax.text(-0.6, 0.2,r"$\omega_\mathrm{c,0}/\omega = $" + "{0:1.3f}".format(art_launch["freq_2X"]/(2.0*art_launch["freq"])) + \
                                                                   r", $\theta=\ang{" + "{0:3.0f}".format(np.rad2deg(art_launch["theta"])) + \
                                                                                   r"}$", color="w")
    plt.show()
    
if(__name__ == "__main__"):
    plot_resonance_lines(0.18, ECRadResFile = "/tokp/work/sdenk/DRELAX_35662_rdiff_prof/ECRad_35662_ECECTCCTA_run3224.mat", \
                        wave_mat_file=None,  \
                        dist_mat="/tokp/work/sdenk/DRELAX_35662_rdiff_prof/ECRad_35662_ECECTCCTA_run3224.mat", \
                        only_central=True, ECE_ch=[94,144])   #94, 144 "/tokp/work/sdenk/Backup_PhD_stuff/DRELAX_Results_2nd_batch/GRAY_rays_35662_4.40.mat"
#     plot_resonance_lines(0.15, art_launch = {"rhop":0.15, "freq": 110.0e9, "freq_2X": 140.e9, "theta": np.deg2rad(89.9), "Te":8.e3, "ne":2.e19})   #94, 144 "/tokp/work/sdenk/Backup_PhD_stuff/DRELAX_Results_2nd_batch/GRAY_rays_35662_4.40.mat",
    