'''
Created on 11.06.2019

@author: sdenk
'''

from BDOP_3D import make_3DBDOP_for_ray, make_f_inter
from equilibrium_utils import EQDataExt
from ECRad_Results import ECRadResults
from distribution_io import load_f_from_mat, read_dist_mat_to_beam
from distribution_functions import Juettner2D
from matplotlib import cm                                        
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from plotting_configuration import *
from scipy.io import loadmat
from distribution_functions import rel_thermal_beta
import scipy.constants as cnst

def find_cell_interceps(u_par_grid, u_perp_grid, cur_BDOP, irhop):
    u_par_interceps = u_par_grid[np.logical_and(u_par_grid > np.min(cur_BDOP.u_par[irhop]),u_par_grid < np.max(cur_BDOP.u_par[irhop]))]
    u_perp_spl = InterpolatedUnivariateSpline(cur_BDOP.u_par[irhop], cur_BDOP.u_perp[irhop])
    u_perp_at_u_par_interceps = u_perp_spl(u_par_interceps)
    u_perp_interceps = []
    u_par_at_u_perp_inteceps = []
    for i, u_perp_intercep in enumerate(u_perp_grid):
        u_perp_interceps_spl = InterpolatedUnivariateSpline(cur_BDOP.u_par[irhop], cur_BDOP.u_perp[irhop] - u_perp_intercep)
        roots = u_perp_interceps_spl.roots()
        if(len(roots) > 0):
            for root in roots:
                if(root >= np.min(u_par_grid) and root <= np.max(u_par_grid)):
                    u_perp_interceps.append(u_perp_intercep)
                    u_par_at_u_perp_inteceps.append(root)
    # Now create list of all intercep points (u_par, u_perp) sorted by ascending u_par
    intercep_list = np.array([np.concatenate([u_par_interceps, u_par_at_u_perp_inteceps]), np.concatenate([u_perp_at_u_par_interceps, u_perp_interceps])])
    sortarray = np.argsort(intercep_list[0])
    return intercep_list.T[sortarray]
    
def diag_weight_stand_alone(Result_file, time_point, ch, DistWaveFile=None):
    Results = ECRadResults()
    Results.from_mat_file(Result_file)
    fig = plt.figure(figsize=(12.5,8.5))
    fig = diag_weight(fig, Results, time_point, ch, DistWaveFile=DistWaveFile)    

def diag_weight(fig, Results, time_point, ch, DistWaveFile=None):
    # Currently only RELAX/LUKE distributions supported
    # Extension for GENE trivial though
    ax = fig.add_subplot(111)
    harmonic_n = 2
    itime = np.argmin(np.abs(time_point - Results.Scenario.plasma_dict["time"]))
    time_cor = Results.Scenario.plasma_dict["time"][itime]
    EQObj= EQDataExt(Results.Scenario.shot)
    EQObj.insert_slices_from_ext(Results.Scenario.plasma_dict["time"], Results.Scenario.plasma_dict["eq_data"])
    B_ax = EQObj.get_B_on_axis(time_cor)
    if(DistWaveFile is not None):
        dist_obj = load_f_from_mat(DistWaveFile, True)
        f_inter, f_inter_scnd = make_f_inter(Results.Config.dstf, dist_obj=dist_obj, EQObj=EQObj, time=time_cor)
    else:
        dist_obj = None
        f_inter, f_inter_scnd  = make_f_inter("Th", EQObj=EQObj, time=time_cor)
    m= 40
    n= 80
    if(dist_obj is None):
        # Estimate grid for thermal distribution assuming X-mode
        rhop_Te_signed = np.concatenate([- Results.Scenario.plasma_dict["rhop_prof"][itime][::-1][:-1], Results.Scenario.plasma_dict["rhop_prof"][itime]])
        Te_signed_rhop = np.concatenate([Results.Scenario.plasma_dict["Te"][itime][::-1][:-1], Results.Scenario.plasma_dict["Te"][itime]])
        Te_spl = InterpolatedUnivariateSpline(rhop_Te_signed, Te_signed_rhop)
        Te_weighted_spl = InterpolatedUnivariateSpline(Results.BPD["rhopX"][itime][ch - 1], Results.BPD["BPDX"][itime][ch - 1] * \
                                                       Te_spl(Results.BPD["rhopX"][itime][ch - 1]))
        Te_av = Te_weighted_spl.integral(-1.0,1.0)
        beta_th = rel_thermal_beta(cnst.c**2 * cnst.electron_mass / cnst.e / Te_av)
        u_th = beta_th / np.sqrt(1.0 - beta_th**2)
        u_perp_grid = np.linspace(0.0, 5.0 * u_th, m)
        u_par_grid = np.linspace(-5.0 * u_th, 5.0 * u_th, n) # five sigma should be good enough!
    else:
        u_perp_grid = np.linspace(0.0, np.max(dist_obj.u), m)
        u_par_grid = np.linspace(-np.max(dist_obj.u), np.max(f_inter.x), n)
    diag_weight_f = np.zeros((m,n))
    diag_weight_rel = np.zeros((m,n))
    for ir in range(Results.Config.N_ray):
        cur_BDOP = make_3DBDOP_for_ray(Results, time_cor, ch, ir, harmonic_n, B_ax, f_inter=f_inter)
        for irhop, rhop in enumerate(cur_BDOP.rho):
            print(irhop + 1, " / ", len(cur_BDOP.rho))
            intercep_points = find_cell_interceps(u_par_grid, u_perp_grid, cur_BDOP, irhop)
            for i_intercep,intercep_point in enumerate(intercep_points[:-1]):
                i = np.argmin(np.abs(intercep_point[0] - u_par_grid))
                j = np.argmin(np.abs(intercep_point[1] - u_perp_grid))
                if(u_par_grid[i] > intercep_point[0]):
                    i -= 1
                if(u_perp_grid[j] > intercep_point[1]):
                    j -= 1
                if(i < 0 or j < 0):
                    continue # only happens at the lower bounds, where u_perp is very small and, therefore, also j is very small
                # Compute arclength
                t = np.zeros(cur_BDOP.u_par[irhop].shape)
                for i_res_line in range(1,len(cur_BDOP.u_par[irhop])):
                    t[i_res_line] = t[i_res_line - 1] + np.sqrt((cur_BDOP.u_par[irhop][i_res_line] - cur_BDOP.u_par[irhop][i_res_line - 1])**2 + \
                                                                (cur_BDOP.u_perp[irhop][i_res_line] - cur_BDOP.u_perp[irhop][i_res_line - 1])**2)
                t /= np.max(t) # Normalize this
                # Sort 
                t_spl = InterpolatedUnivariateSpline(cur_BDOP.u_par[irhop], t)
                try:
                    BPD_val_spl = InterpolatedUnivariateSpline(t, cur_BDOP.val[irhop])
                except Exception as e:
                    print(e)
                BPD_val_rel_spl = InterpolatedUnivariateSpline(t, np.abs(cur_BDOP.val[irhop] - cur_BDOP.val_back[irhop]))
                t1 = t_spl(intercep_point[0])
                t2 = t_spl(intercep_points[i_intercep + 1][0])
                diag_weight_f[j,i] += Results.weights["ray"][itime][ch][ir] * \
                                    BPD_val_spl.integral(t1, t2)
                diag_weight_rel[j,i] += Results.weights["ray"][itime][ch][ir] * \
                                    BPD_val_rel_spl.integral(t1, t2)
    ax.contour(u_perp_grid, u_par_grid, diag_weight_f.T / np.max(diag_weight_f.flatten()), \
                 levels = np.linspace(0.01,1,10), cmap = plt.get_cmap("plasma"))
    m = cm.ScalarMappable(cmap=plt.cm.get_cmap("plasma"))
    m.set_array(np.linspace(0.01, 1.0, 10))
    cb_diag = fig.colorbar(m, pad=0.15, ticks=[0.0, 0.5, 1.0])
    cb_diag.set_label(r"$D_\omega [\si{{a.u.}}]$")
    ax.set_ylabel(r"$u_\parallel$")
    ax.set_xlabel(r"$u_\perp$")
    ax.set_aspect("equal")
    return fig


def current_weight(DistWaveFile):
    dist_obj = load_f_from_mat(DistWaveFile, True)
    dist_mat = loadmat(DistWaveFile, squeeze_me=True)
    waves = read_dist_mat_to_beam(dist_mat, True)
    ECCD_weight = np.zeros(dist_obj.f_cycl[0].shape)
    j_spl = InterpolatedUnivariateSpline(waves.rhop, waves.j)
    j_tot = j_spl.integral(waves.rhop[0], waves.rhop[-1])
#     u_par_stop = np.max(dist_obj.ull[dist_obj.ull < 0])
    u_par_stop = -0.05
    for irhop, rhop in enumerate(dist_obj.rhop):
        weight = np.abs(waves.j[irhop] / j_tot)
        print(rhop, weight)
        for i, u_par in enumerate(dist_obj.ull):
            for j, u_perp  in enumerate(dist_obj.uxx):
                f_diff = dist_obj.f_cycl[irhop][i,j] - Juettner2D(u_par, u_perp, dist_obj.Te[irhop])
                ECCD_weight[i,j] += weight * f_diff * u_par 
    ECCD_weight += np.max(np.abs(ECCD_weight.flatten()))
    ECCD_weight /= np.max(np.abs(ECCD_weight.flatten()))
    ECCD_weight -= 0.5
    ECCD_weight *= 2.0
    plt.contourf(dist_obj.uxx, dist_obj.ull, ECCD_weight / np.max(ECCD_weight.flatten()), \
                 levels = np.linspace(-1,1,30), cmap = plt.get_cmap("coolwarm"))
    plt.gca().set_ylabel(r"$u_\parallel$")
    plt.gca().set_xlabel(r"$u_\perp$")
    m = cm.ScalarMappable(cmap=plt.cm.get_cmap("coolwarm"))
    m.set_array(np.linspace(-1.0, 1.0, 30))
    cb_j = plt.gcf().colorbar(m, pad=0.15, ticks=[-1.0, 0.0, 1.0])
    cb_j.set_label(r"$(f - f_0) u_\parallel [\si{{a.u.}}]$")
    plt.gca().set_aspect("equal")
#     plt.show()
if(__name__ == "__main__"):
    fig = plt.figure()
    #current_weight("/tokp/work/sdenk/ECRad/ECRad_35662_ECECTACTC_ed9.mat")
    current_weight('/tokp/work/sdenk/DRELAX_Results_2nd_batch/ECRad_35662_ECECTCCTA_run0006.mat')
    diag_weight_stand_alone('/tokp/work/sdenk/DRELAX_Results_2nd_batch/ECRad_35662_ECECTCCTA_run0006.mat', 4.4, 94, \
                            '/tokp/work/sdenk/DRELAX_Results_2nd_batch/ECRad_35662_ECECTCCTA_run0006.mat')
    diag_weight_stand_alone('/tokp/work/sdenk/DRELAX_Results_2nd_batch/ECRad_35662_ECECTCCTA_run0006.mat', 4.4, 144, \
                            '/tokp/work/sdenk/DRELAX_Results_2nd_batch/ECRad_35662_ECECTCCTA_run0006.mat')
    plt.show()
    