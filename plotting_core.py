# -*- coding: utf-8 -*-
"""
Created on Thu May 01 20:53:35 2014

@author: Severin Denk
"""
from plotting_configuration import *
import numpy as np
from subprocess import call
from scipy.integrate import quad
from glob import glob
from matplotlib import cm
import os
from shutil import copyfile
from GlobalSettings import AUG, TCV
if(AUG):
    from equilibrium_utils_AUG import EQData, make_rhop_signed_axis
    from shotfile_handling_AUG import get_diag_data_no_calib, get_data_calib, load_IDA_data, get_shot_heating, get_NPA_data, get_ECE_spectrum, get_Thomson_data
    import fconf
    from get_ECRH_config import get_ECRH_viewing_angles
elif(TCV):
    from equilibrium_utils_TCV import EQData, make_rhop_signed_axis
    from shotfile_handling_AUG import get_diag_data_no_calib, get_data_calib, load_IDA_data, get_shot_heating, get_NPA_data, get_ECE_spectrum, get_Thomson_data
else:
    print('Neither AUG nor TCV selected')
    raise(ValueError('No system selected!'))
from Diags import Diag
from electron_distribution_utils import read_svec_from_file, identify_LFS_channels, make_R_res, remap_rhop_R, get_B
from electron_distribution_utils import make_f_beta, Maxwell2D, find_cold_res, find_rel_res, make_f, BiJuettner, multi_slope_simpl, Gauss_norm, get_Te_ne_R, weighted_emissivity
from electron_distribution_utils import make_f_1D, Maxwell1D_beta, MJ_approx, rel_rhop_res_all_ch, get_omega_c_and_cutoff, cycl_distribution, cyc_momentum
from electron_distribution_utils import ratio_B, weighted_emissivity_along_s, Juettner2D, Juettner1D_beta, multi_slope, multi_slope_cyl_beta
import scipy.constants as cnst
from scipy.integrate import simps
from ECRad_utils import get_files_and_labels
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline, SmoothBivariateSpline
from scipy import stats
from wxEvents import ThreadFinishedEvt, Unbound_EVT_DONE_PLOTTING
import wx
from colorsys import hls_to_rgb
non_therm_dist_Relax = True
home = '/afs/ipp-garching.mpg.de/home/s/sdenk/'
# import matplotlib.gridspec as gridspec
class plotting_core:
    def __init__(self, fig, fig_2=None, title=True):
        self.fig = fig
        self.fig_2 = fig_2
        self.reset(title)
        self.diag_markers = ["o", "s", "*", "o", "s", "*", "o", "s", "*"]
        self.model_markers = ["^", "v", "+", "d", "^", "v", "+", "d", "^", "v", "+", "d"]
        self.line_markers = ["-", "--", ":", "-", "--", ":", "-", "--", ":"]
        self.diag_colors = [(0.0, 126.0 / 255, 0.0)]
        self.model_colors = [(1.0, 0.e0, 0.e0), (126.0 / 255, 0.0, 126.0 / 255), (0.0, 0.0, 0.0), (0.0, 126.0 / 255, 1.0), (126.0 / 255, 126.0 / 255, 0.0)]
        self.line_colors = [(0.0, 0.e0, 0.e0)]
        self.n_diag_colors = 8
        self.n_line_colors = 8
        self.n_model_colors = 12
        self.diag_cmap = plt.cm.ScalarMappable(plt.Normalize(0, 1), "Paired")
        self.line_cmap = plt.cm.ScalarMappable(plt.Normalize(0, 1), "Set1")
#        try:
#            self.model_cmap = plt.cm.ScalarMappable(plt.Normalize(0, 1), "nipy_spectral")
#        except:
        self.model_cmap = plt.cm.ScalarMappable(plt.Normalize(0, 1), "nipy_spectral")
        n_diag_steps = np.linspace(0.0, 1.0, self.n_diag_colors)
        n_line_steps = np.linspace(0.0, 1.0, self.n_line_colors)
        model_steps = np.linspace(0.0, 1.0, self.n_model_colors)
        self.diag_colors = np.concatenate([self.diag_colors, self.diag_cmap.to_rgba(n_diag_steps).T[0:3].T])
        self.line_colors = np.concatenate([self.line_colors, self.line_cmap.to_rgba(n_line_steps).T[0:3].T])
        self.model_colors = np.concatenate([self.model_colors, self.model_cmap.to_rgba(model_steps).T[0:3].T[::-1]])
        self.model_marker_index = [0]
        self.model_color_index = [0]
        self.diag_marker_index = [0]
        self.diag_color_index = [0]
        self.line_marker_index = [0]
        self.line_color_index = [0]
        self.model_marker_index_2 = [0]
        self.model_color_index_2 = [0]
        self.diag_marker_index_2 = [0]
        self.diag_color_index_2 = [0]
        self.line_marker_index_2 = [0]
        self.line_color_index_2 = [0]

    def reset(self, title=True):
        self.setup = False
        self.fig.clf()
        if(self.fig_2 is not None):
            self.fig_2.clf()
        self.gridspec = None
        self.gridsec_2 = None
        self.axlist = []
        self.axlist_2 = []
        self.first_invoke = False
        self.layout = [1, 1, 1]
        self.layout_2 = [1, 1, 1]
        self.x_share_list = [None]
        self.y_share_list = [None]
        self.y_range_list = [[np.inf, -np.inf]]
        self.x_share_list_2 = [None]
        self.y_share_list_2 = [None]
        self.y_range_list_2 = [[np.inf, -np.inf]]
        self.title = title
        self.x_step_min = [None]
        self.x_step_max = [None]
        self.y_step_min = [None]
        self.y_step_max = [None]
        self.x_step_min_2 = [None]
        self.x_step_max_2 = [None]
        self.y_step_min_2 = [None]
        self.y_step_max_2 = [None]
        self.model_marker_index = [0]
        self.model_color_index = [0]
        self.diag_marker_index = [0]
        self.diag_color_index = [0]
        self.line_marker_index = [0]
        self.line_color_index = [0]
        self.model_marker_index_2 = [0]
        self.model_color_index_2 = [0]
        self.diag_marker_index_2 = [0]
        self.diag_color_index_2 = [0]
        self.line_marker_index_2 = [0]
        self.line_color_index_2 = [0]

    def get_figures(self):
        return self.fig, self.fig_2

    def simple_plot(self, filename):
        self.setup_axes("", "simple test")
        self.axlist[0], self.y_range_list[0] = \
            self.add_plot(self.axlist[0], self.y_range_list[0], data=\
                np.array([np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]))

    def get_si_string(self, unit):
        if(plot_mode != 'Software'):
            return r'\si{' + unit + r'}'
        else:
            format_string = unit.split('\\')
            output = r'[\, \mathrm{'
            for cur_str in format_string:
                if(len(cur_str) == 0):
                    continue
                elif(cur_str == 'second'):
                    output += 's'
                else:
                    print('Unit ' + cur_str + ' not yet supported')
                    output += '?'
            return output + r'}]'

    def make_SI_string(self, value, unit):
        if(plot_mode != 'Software'):
            return r'\SI{' + value + r'}{' + unit + r'}'
        else:
            output = self.get_si_string(unit)
            return value + r'\,' + output

    def make_num_string(self, value):
        if(plot_mode != 'Software'):
            return r'\num{' + value + r'}'
        else:
            return value
#    def Imaging_data(self):
#        fig = plt.figure(figsize=(9, 9))
#        gs = gridspec.GridSpec(nPlots, 2)
#        ax = fig.add_subplot(gs[:, 0])
        # plt.title(r"$\rm{<T_{rad}>}$" ,fontsize=22)
        # plt.title(r"cold res. ECEI at $\rho_{pol}\sim$%.3f
#        (dashed)"%mData['rhop'][idxROI].mean() ,fontsize=22)
#        ax.text(0.01,
#        1.045,r'(a)',horizontalalignment='left',verticalalignment='top'
#        ,transform=ax.transAxes,bbox=dict(facecolor='w', edgecolor='black',
#        boxstyle='round'),fontsize=18)
#        plt.pcolormesh(RR.T, zz.T, gridTrad.T,shading='gouraud',vmin=250.,vmax=800.)
#        plt.plot(mData['R'][idxROI],mData['z'][idxROI], 'wo', label = 'used
#        ch.',markersize=10)
#        plt.plot(mData['R'][idxAvail],mData['z'][idxAvail],'kx',label='cold res.')
#        plt.xlim([Rmin*1.005,Rmax*0.995])
#        plt.ylim([zmin*0.98,zmax*0.98])
#        plt.xlabel(r"$R\ \rm{[m]}$",fontsize=labelsize)
#        plt.ylabel(r"$z\ \rm{[m]}$",fontsize=labelsize)
#        ax.yaxis.set_label_coords(-0.1, 0.5)

    def Show_TRad(self, folder, compare_folders, compare_eds, shotno, time, dstf, simpl=False, ne=True, diags=None, \
                  model=True, Te=True, x_axis_LFS_HFS=False, rel_diff=False, \
                  rel_res=False, Comp=False, IDA_model=False, rho_max=1.5, \
                  eq_diag=None, mode="mix", X_frac=False, ECE_freq=False, harmonic_number=2, \
                  lower_tau_boundary=0.0, show_measurements=True):
        if(eq_diag is None):
            eq_diag = Diag("EQ", "AUGD", "EQH", 0)
        shotstr = "\#" + str(shotno) + " t = " + "{0:1.4f}".format(time) + " s  "
        rel_diff_comp = False
        ne_at_ax_2 = False
        if((rel_diff and model and simpl or ((model or simpl) and X_frac))):
            twinx = '_twinx'
        elif((rel_diff or X_frac) and model and Comp):
            twinx = '_twinx'
            rel_diff_comp = True
            rel_diff = False
            ne_at_ax_2 = True
        else:
            twinx = ''
            rel_diff = False
        if(not ne and Te and (len(diags.keys()) > 0 or model or simpl)):
            self.setup_axes("Te_no_ne" + twinx, r"$T_\mathrm{rad}$, $T_\mathrm{e}$ for " + \
                    shotstr, r"Optical depth $\tau_\omega$", shot=shotno, time=time)
        elif(not ne and not Te):
            self.setup_axes("Te_no_ne" + twinx, r"$T_\mathrm{rad}$ for " + \
                    shotstr, shot=shotno, time=time)
        elif(not ne and Te and not (len(diags.keys()) > 0 and model and simpl)):
            self.setup_axes("Te_no_ne" + twinx, r"$T_\mathrm{e}$ for " + \
                    shotstr, r"Optical depth $\tau_\omega$", shot=shotno, time=time)
        elif(not ne):
            self.setup_axes("Te_no_ne" + twinx, r"$T_\mathrm{rad}$, $T_\mathrm{e}$, $n_\mathrm{e}$ for " + \
                    shotstr, r"Optical depth $\tau_\omega$", shot=shotno, time=time)
        else:
            self.setup_axes("Te" + twinx, r"$T_\mathrm{rad}$, $T_\mathrm{e}$ and $n_\mathrm{e}$ for " + \
                    shotstr, r"Optical depth $\tau_\omega$", shot=shotno, time=time)
        if(os.path.isfile(os.path.join(folder, "residue_ece.res"))):
            ECE_data = os.path.join(folder, "residue_ece.res")
        elif(os.path.isfile(os.path.join(folder, "ECE_data.res"))):
            ECE_data = os.path.join(folder, "ECE_data.res")
        else:
            print("No ECE data in", folder)
            ECE_data = os.path.join(folder, "ECE_data.res")
        if(mode == "XO" or mode == "OX"):
            diag_masks = self.overview_plot(folder, shotno, time, ECE_data, \
                           dstf=dstf, ne=ne, diags=diags, show_measurements=show_measurements, \
                           model=model, simpl=simpl, Te=Te, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                           rel_diff=rel_diff, rel_res=rel_res, \
                           Comp=False, IDA_model=IDA_model, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                           mode="mix", X_frac=X_frac, ECE_freq=ECE_freq, harmonic_number=harmonic_number, \
                           lower_tau_boundary=lower_tau_boundary)  # residue_ece.res
            diag_masks = self.overview_plot(folder, shotno, time, ECE_data, \
                           dstf=dstf, ne=ne, diags=diags, show_measurements=False, \
                           model=model, simpl=simpl, Te=False, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                           rel_diff=False, rel_res=rel_res, \
                           Comp=False, IDA_model=IDA_model, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                           mode="X", X_frac=False, harmonic_number=harmonic_number, \
                           lower_tau_boundary=lower_tau_boundary)  # residue_ece.res
            diag_masks = self.overview_plot(folder, shotno, time, ECE_data, \
                           dstf=dstf, ne=ne, diags=diags, show_measurements=False, \
                           model=model, simpl=simpl, Te=False, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                           rel_diff=False, rel_res=rel_res, \
                           Comp=False, IDA_model=IDA_model, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                           mode="O", X_frac=False, harmonic_number=harmonic_number, \
                           lower_tau_boundary=lower_tau_boundary)  # residue_ece.res
        else:
            diag_masks = self.overview_plot(folder, shotno, time, ECE_data, \
                           dstf=dstf, ne=ne, diags=diags, show_measurements=show_measurements, \
                           model=model, simpl=simpl, Te=Te, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                           rel_diff=rel_diff, rel_res=rel_res, \
                           Comp=False, IDA_model=IDA_model, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                           mode=mode, X_frac=X_frac, ECE_freq=ECE_freq, harmonic_number=harmonic_number, \
                           lower_tau_boundary=lower_tau_boundary)  # residue_ece.res
        if(Comp):
            if("IDA" in diags.keys()):
                org_ed = diags["IDA"].ed
            ECE_freq_loc = ECE_freq
            for comp_folder, compare_ed in zip(compare_folders, compare_eds):
                try:
                    if("IDA" in diags.keys()):
                        diags["IDA"].ed = compare_ed
                    if(mode == "XO" or mode == "OX"):
                        diag_masks = self.overview_plot(comp_folder, shotno, time, ECE_data, \
                                       dstf=dstf, ne=ne, diags=diags, show_measurements=False, \
                                       model=model, simpl=simpl, Te=Te, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                                       rel_diff=rel_diff, rel_diff_comp=rel_diff_comp, rel_diff_folder=folder, rel_res=rel_res, \
                                       Comp=Comp, IDA_model=False, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                                       mode="mix", X_frac=X_frac, ECE_freq=ECE_freq_loc, harmonic_number=harmonic_number, \
                                       lower_tau_boundary=lower_tau_boundary, diag_masks=diag_masks)  # residue_ece.res
                        diag_masks = self.overview_plot(comp_folder, shotno, time, ECE_data, \
                                       dstf=dstf, ne=ne, diags=diags, show_measurements=False, \
                                       model=model, simpl=simpl, Te=False, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                                       rel_diff=False, rel_diff_comp=False, rel_res=rel_res, \
                                       Comp=Comp, IDA_model=False, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                                       mode="X", X_frac=False, harmonic_number=harmonic_number, \
                                       lower_tau_boundary=lower_tau_boundary, diag_masks=diag_masks)
                        diag_masks = self.overview_plot(comp_folder, shotno, time, ECE_data, \
                                       dstf=dstf, ne=ne, diags=diags, show_measurements=False, \
                                       model=model, simpl=simpl, Te=False, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                                       rel_diff=False, rel_diff_comp=False, rel_res=rel_res, \
                                       Comp=Comp, IDA_model=False, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                                       mode="O", X_frac=False, harmonic_number=harmonic_number, \
                                       lower_tau_boundary=lower_tau_boundary, diag_masks=diag_masks)  # residue_ece.res
                    else:
                        diag_masks = self.overview_plot(comp_folder, shotno, time, ECE_data, \
                                       dstf=dstf, ne=ne, diags=diags, show_measurements=False, \
                                       model=model, simpl=simpl, Te=Te, x_axis_LFS_HFS=x_axis_LFS_HFS, \
                                       rel_diff=rel_diff, rel_diff_comp=rel_diff_comp, rel_diff_folder=folder, rel_res=rel_res, \
                                       Comp=Comp, IDA_model=False, rho_max=rho_max, ne_at_ax_2=ne_at_ax_2, \
                                       mode=mode, X_frac=X_frac, ECE_freq=ECE_freq_loc, harmonic_number=harmonic_number, \
                                       lower_tau_boundary=lower_tau_boundary, diag_masks=diag_masks)  # residue_ece.res
                    ECE_freq_loc = False
                except IndexError as e:
                    print(e)
                    print("Index Error occurred while creating comparison plot")
                    print("Most likely cause is that the routine ran out of marker or colors")
                    print("Try plotting fewer graphs")
            if("IDA" in diags.keys()):
                diags["IDA"].ed = org_ed
        if(not ne):
            self.create_legends("Te_no_ne" + twinx)
        else:
            self.create_legends("Te" + twinx)

        return self.fig, self.fig_2

    def comparison_plot(self, folder, shotno, time, mfilename, dfilename, \
        tfilename="te_ida.res", nfilename="ne_ida.res", f="na"):
        # dist = "well-Juettner distribution"
        d_abs_filename = os.path.join(folder, dfilename)
        m_abs_filename = os.path.join(folder, mfilename)
        te_abs_filename = os.path.join(folder, tfilename)
        ne_abs_filename = os.path.join(folder, nfilename)
        # n_mod_filename = os.path.join(folder,"ECRad_data/par_n_mod.dat")
        # t_mod_filename = os.path.join(folder,"ECRad_data/par_t_mod.dat")
        th_abs_filename = os.path.join(folder, "residue_ece.res")
        th_mod_filename = os.path.join(folder, "residue_ece_old.res")  # _unmod
        te_mod_filename = os.path.join(folder, "Te.dat")
        ne_mod_filename = os.path.join(folder, "ne_ida_old.res")
        # r_mod_filename = os.path.join(folder,"ECRad_data/TRadM_Relax.dat")
        d_backup_filename = os.path.join(folder, "residue_ece.res")
        try:
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], filename=d_abs_filename, \
                name=r"ECE data", marker="D", color=(0.75, 0.0, 0.75), y_range_in=self.y_range_list_2[0])
        except:
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], filename=d_backup_filename, \
                name=r"ECE data", marker="D", color=(0.75, 0.0, 0.75), y_range_in=self.y_range_list_2[0])
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], filename=th_abs_filename, \
            name=r"IDA MJ Rk4 $T_\mathrm{rad}$", marker="o", coloumn=2, color=(1.0, 0.5, 0.0), y_range_in=self.y_range_list_2[0])

        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], filename=te_abs_filename, \
                name="IDA MJ Rk4 $T_\mathrm{e}$", marker="-", coloumn=1, color=(0.3, 0.3, 0.5), y_range_in=self.y_range_list_2[0])
        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], filename=ne_abs_filename, ax_flag="ne", \
                name=r"IDA MJ Rk4 - electron density", marker="-", coloumn=1, color=(0.3, 0.5, 0.0), y_range_in=self.y_range_list_2[1])
        try:
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], filename=te_mod_filename, \
                name=r"IDA non-rel edf  $T_\mathrm{e}$", marker="-", coloumn=1, color=(0.5, 0.3, 0.3), y_range_in=self.y_range_list_2[0])
            self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], filename=ne_mod_filename, ax_flag="ne", \
                name=r"IDA non-rel edf. - electron density", marker="-", coloumn=1, color=(1.0, 0.0, 0.0), y_range_in=self.y_range_list_2[1])
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], filename=th_mod_filename, \
                name=r"IDA non-rel edf. $T_\mathrm{rad}$", marker="s", coloumn=2, color=(0.0, 0.0, 0.75), y_range_in=self.y_range_list_2[0])
        except:
            pass

    def overview_plot(self, folder, shotno, time, dfilename, \
            dstf=None, simpl=None, ne=True, diags={}, show_measurements=True, model=True, Te=True, x_axis_LFS_HFS=False, \
            rel_diff=False, rel_res=False, Comp=False, IDA_model=False, \
            rho_max=1.5, rel_diff_comp=False, rel_diff_folder=None, ne_at_ax_2=False, eq_diag=None, \
            mode="mix", X_frac=False, ECE_freq=False, error_margin=0.5, harmonic_number=2, lower_tau_boundary=0.0, diag_masks={}):
        if(eq_diag is None):
            eq_diag = Diag("EQ", "AUGD", "EQH", 0)
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        if(Comp and (dstf == "ON" or dstf == "BA")):
            data_name = simplfile
            dist = dist_simpl
        print(data_name)
        print(simplfile)
        print(tertiary_model)
        data_name_X_frac = os.path.join(folder, "ECRad_data", "X_" + data_name)
        simplfile_X_frac = os.path.join(folder, "ECRad_data", "X_" + simplfile)
        if(tertiary_model is not None):
            tertiary_model_X_frac = os.path.join(folder, "ECRad_data", "X_" + tertiary_model)
            if(mode == "X"):
                tertiary_model = "X_" + tertiary_model
            if(mode == "O"):
                tertiary_model = "O_" + tertiary_model
        mode_str = ""
        if(mode == "X"):
            data_name = "X_" + data_name
            simplfile = "X_" + simplfile
            mode_str = " $X$-mode"
        elif(mode == "O"):
            data_name = "O_" + data_name
            simplfile = "O_" + simplfile
            mode_str = " $O$-mode"
        if(simpl):
            simpl = simplfile
        else:
            simpl = None
        fall_back_diag = "CEC"
        if("ECE" in diags.keys()):
            if(diags["ECE"].diag == "CEC"):
                fall_back_diag = "RMD"
        d_abs_filename = os.path.join(folder, dfilename)
        # d_backup_filename = os.path.join(folder, "residue_ece.res")
        diag_filename = os.path.join(folder, "ECRad_data", "diag.dat")
        m_abs_filename = os.path.join(folder, "ECRad_data", data_name)
        if(not os.path.isfile(m_abs_filename)):
            if(data_name.startswith("X_")):
                data_name = data_name.split("X_")[1]
            elif(data_name.startswith("O_")):
                data_name = data_name.split("O_")[1]
            print("Could not find " + mode + " specific file: " + m_abs_filename)
            print("Trying file with X O  mode super position")
            m_abs_filename = os.path.join(folder, "ECRad_data", data_name)
        if("OERT" in folder):
            m_abs_comp_filename = os.path.join(folder.replace("/OERT", ""), "ECRad_data", data_name)
            diag_comp_filename = os.path.join(folder.replace("/OERT", ""), "ECRad_data", "diag.dat")
        else:
            m_abs_comp_filename = os.path.join(folder, "OERT", "ECRad_data", data_name)
            diag_comp_filename = os.path.join(folder, "OERT", "ECRad_data", "diag.dat")
        if(simpl is not None):
            m_simpl_filename = os.path.join(folder, "ECRad_data", simpl)
            if(not os.path.isfile(m_simpl_filename)):
                if(simpl.startswith("X_")):
                    simpl = simpl.split("X_")[1]
                elif(simpl.startswith("O_")):
                    simpl = simpl.split("O_")[1]
                print("Could not find " + mode + " specific file for secondary model: " + m_simpl_filename)
                print("Trying file with X O  mode super position")
                m_simpl_filename = os.path.join(folder, "ECRad_data", simpl)
        if(backup_name is not None):
            backup_name = os.path.join(folder, "ECRad_data", backup_name)
#            if("OERT" in folder):
#                m_simpl_comp_filename = os.path.join(folder.replace("\OERT", ""), "ECRad_data", simpl)
#            else:
#                m_simpl_comp_filename = os.path.join(folder, "OERT", "ECRad_data", simpl)
        te_reduced_filename = os.path.join(folder, "te_ida.res")
        freq = np.loadtxt(os.path.join(folder, "ECRad_data", "f_ECE.dat"))
        ch_cnt = len(freq)
        t_abs_filename = os.path.join(folder, "te_ida.res")
        n_abs_filename = os.path.join(folder, "ne_ida.res")
        if("OERT" in folder):
            try:
                try:
                    Te_data = np.loadtxt(os.path.join(folder, "ECRad_data", "Te_file.dat"), skiprows=1)
                    rhop_te_vec = Te_data.T[0]
                    te_vec = Te_data.T[1] / 1.e3
                except IOError:
                    print("No te_dat.file at", os.path.join(folder, "ECRad_data", "Te_file.dat"))
                    rhop_te_vec, te_vec = self.read_file(t_abs_filename)
            except IOError:
                print("No ida Te data -> Te turned off")
                Te = False
            try:
                try:
                    ne_data = np.loadtxt(os.path.join(folder, "ECRad_data", "ne_file.dat"), skiprows=1)
                    rhop_ne_vec = ne_data.T[0]
                    ne_vec = ne_data.T[1] / 1.e20
                except IOError:
                    print("No ne_dat.file at", os.path.join(folder, "ECRad_data", "ne_file.dat"))
                    rhop_ne_vec, ne_vec = self.read_file(n_abs_filename)
            except IOError:
                print("No ida ne data -> ne turned off")
                ne = False
        else:
            try:
                rhop_ne_vec, ne_vec = self.read_file(n_abs_filename)
            except IOError:
                print("No ida ne data -> ne turned off")
                ne = False
            try:
                try:
                    rhop_te_vec, te_vec = self.read_file(te_reduced_filename)
                except IOError:
                    rhop_te_vec, te_vec = self.read_file(t_abs_filename)
            except IOError:
                print("No ida Te data -> Te turned off")
                Te = False
        if(x_axis_LFS_HFS):
            EQ_obj = EQData(int(shotno), EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=int(eq_diag.ed))
            Channel_list = identify_LFS_channels(float(time), folder, ch_cnt, EQ_obj, rel_res)
            rhop_te_vec = np.hstack([-rhop_te_vec[::-1], rhop_te_vec])
            te_vec = np.hstack([te_vec[::-1], te_vec])
            if(ne):
                rhop_ne_vec = np.hstack([-rhop_ne_vec[::-1], rhop_ne_vec])
                ne_vec = np.hstack([ne_vec[::-1], ne_vec])
        else:
            Channel_list = np.zeros(ch_cnt)
            Channel_list[:] = 1
        model_data = np.loadtxt(m_abs_filename)
        if(rel_res):
            rhop = rel_rhop_res_all_ch(folder)
        else:
            rhop = model_data.T[0]
            if(harmonic_number != 2):
                for i in range(1, len(rhop) + 1):
                    success, dummy, dummy_2, dummy_3, rhop[i - 1] = find_cold_res(os.path.join(folder, "ECRad_data"), i, mode, harmonic_number)
                    if(rhop[i - 1] > 1.3 or success == False):
                        success, dummy, dummy_2, dummy_3, rhop[i - 1] = find_cold_res(os.path.join(folder, "ECRad_data"), i, mode, 2)
        rhop[Channel_list == 0] *= -1.0
        tau = model_data.T[2]
        try:
            diag_data = np.genfromtxt(diag_filename, dtype='str')
        except IOError:
            diag_data = np.zeros(ch_cnt, dtype='|S3')
            if("ECE" in diags.keys()):
                diag_data[:] = diags["ECE"].diag
            else:
                diag_data[:] = "CEC"
            print("Warning no diag data found")
        if("ECE" in diags.keys()):
            ECE_diag_mask = ((diag_data == diags["ECE"].diag) | (diag_data == fall_back_diag) | (diag_data == "ECE"))
            if(diags["ECE"].diag not in diag_data and fall_back_diag in diag_data):
                diags["ECE"].diag = fall_back_diag
        # is_ch_on_hfs(shot_folder, shotno, time, Ich)

        # dist_simple = "MJ"
        # if(dstf == "Mx"):
        #    dist_simple = "well"
        # n_mod_filename = os.path.join(folder,"ECRad_data/par_n_mod.dat")
        # t_mod_filename = os.path.join(folder,"ECRad_data/par_t_mod.dat")
        # th_abs_filename = os.path.join(folder,"ECRad_data/TRadM_therm.dat")
        cur_filelist = glob(folder + os.path.sep + "*")
        if(Te and not(model or len(diags.keys()) > 0 or simpl or IDA_model)):
            ax_flag = "Te"
        elif(not Te):
            ax_flag = "Te_rad"
        else:
            ax_flag = "Te_Te_rad"
        if(os.path.join(folder, "residue_ece_mod.res") in cur_filelist):
            d_abs_filename = os.path.join(folder, "residue_ece_mod.res")
            print("Warning modified ECE-data is displayed")
        if(ne):
            label_x = False  #    y_range_in = self.y_range_list[0])  \,R = 0.90 $
            if(rel_diff or rel_diff_comp or ne_at_ax_2 or X_frac):
                self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], data=[rhop_ne_vec, ne_vec], \
                    name=r"$n_\mathrm{e}$", marker=self.line_markers[self.line_marker_index[1]], color=self.line_colors[self.line_color_index[1]], y_range_in=self.y_range_list[2], ax_flag="ne")
            else:
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[rhop_ne_vec, ne_vec], \
                    name=r"$n_\mathrm{e}$", marker=self.line_markers[self.line_marker_index[1]], color=self.line_colors[self.line_color_index[1]], y_range_in=self.y_range_list[1], ax_flag="ne")
            self.line_color_index[1] += 1
        else:
            label_x = True
        if(Te):
            if("IDA" in diags.keys()):
                plasma_dict = load_IDA_data(shotno, [time], exp=diags["IDA"].exp, ed=diags["IDA"].ed)
                print("Time point for IDA profile", plasma_dict["time"])
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[plasma_dict["rhop"][0], plasma_dict["Te"][0] / 1.e3], \
                                                                     name=r"$T_\mathrm{e}$", marker=self.line_markers[self.line_marker_index[0]], color=self.line_colors[self.line_color_index[0]], \
                                                                     y_range_in=self.y_range_list[0], y_scale=1.0, ax_flag=ax_flag, label_x=label_x)  # \times 100$
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[plasma_dict["rhop"][0], plasma_dict["Te_up"][0] / 1.e3], \
                                                                     marker=":", color=self.line_colors[self.line_color_index[0]], \
                                                                     y_range_in=self.y_range_list[0], y_scale=1.0, ax_flag=ax_flag, label_x=label_x)
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[plasma_dict["rhop"][0], plasma_dict["Te_low"][0] / 1.e3], \
                                                                     marker=":", color=self.line_colors[self.line_color_index[0]], \
                                                                     y_range_in=self.y_range_list[0], y_scale=1.0, ax_flag=ax_flag, label_x=label_x)
                self.line_color_index[0] += 1
                self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], data=[plasma_dict["rhop"][0], plasma_dict["Te"][0] / 1.e3], \
                                                                         name=r"$T_\mathrm{e}$", coloumn=1, marker=self.line_markers[self.line_marker_index_2[0]], color=self.line_colors[self.line_color_index_2[0]], \
                                                                         y_range_in=self.y_range_list_2[1], y_scale=1.0, ax_flag="Te", label_x=label_x)  # \times 100$
                self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], data=[plasma_dict["rhop"][0], plasma_dict["Te_up"][0] / 1.e3], \
                                                                         marker=":", color=self.line_colors[self.line_color_index_2[0]], \
                         y_range_in=self.y_range_list_2[1], y_scale=1.0, ax_flag="Te", label_x=label_x)
                self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], data=[plasma_dict["rhop"][0], plasma_dict["Te_low"][0] / 1.e3], \
                                                                         marker=":", color=self.line_colors[self.line_color_index_2[0]], \
                                                                         y_range_in=self.y_range_list_2[1], y_scale=1.0, ax_flag="Te", label_x=label_x)
                self.line_color_index_2[0] += 1
            else:
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[rhop_te_vec, te_vec], \
                    name=r"$T_\mathrm{e}$", coloumn=1, marker=self.line_markers[self.line_marker_index[0]], color=self.line_colors[self.line_color_index[0]], \
                         y_range_in=self.y_range_list[0], y_scale=1.0, ax_flag=ax_flag, label_x=label_x)  # \times 100$
                self.line_color_index[0] += 1
                self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], data=[rhop_te_vec, te_vec], \
                    name=r"$T_\mathrm{e}$", coloumn=1, marker=self.line_markers[self.line_marker_index_2[0]], color=self.line_colors[self.line_color_index_2[0]], \
                         y_range_in=self.y_range_list_2[1], y_scale=1.0, ax_flag="Te", label_x=label_x)  # \times 100$
                self.line_color_index_2[0] += 1
        if("ECE" in diags.keys() and show_measurements):
            if(diags["ECE"].diag not in diag_data and "ECE" not in diag_data):
                diags["ECE"].diag = fall_back_diag
            y_err, ECE_data = get_data_calib(diag=diags["ECE"], shot=shotno, time=float(time), ext_resonances=rhop[np.logical_or(diag_data == diags["ECE"].diag, diag_data == "ECE")], \
                                             t_smooth=diags["ECE"].t_smooth)
            y_err = y_err[0]
            if(ECE_data is not None):
                if(len(y_err) == 0 or len(ECE_data[0]) == 0 or len(ECE_data[1]) == 0):
                    print("Warning no data was loaded", len(y_err), len(ECE_data[0]), len(ECE_data[1]))
                if(ECE_data is not None and len(ECE_data[0]) == len(ECE_data[1])):
                    diag_masks["ECE"] = np.logical_and(y_err < error_margin * ECE_data[1], ECE_data[0] < rho_max)
                    diag_masks["ECE"] = np.logical_and(diag_masks["ECE"], tau[np.logical_or(diag_data == diags["ECE"].diag, diag_data == "ECE")] > lower_tau_boundary)
                    # ECE_data[0][Channel_list[np.logical_or(diag_data == diags["ECE"].diag, diag_data == fall_back_diag)] == 0] *= -1.0
                    if(not Comp):
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[ECE_data[0][diag_masks["ECE"]], \
                                                                                   ECE_data[1][diag_masks["ECE"]]], \
                                                                             y_error=y_err[diag_masks["ECE"]], \
                                                                             name=r"$T_\mathrm{rad,ECE}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x, y_scale=1.e0)  # (0.0,0.3,0.0)
                        self.diag_color_index[0] += 1
            else:
                print("Failed to load ECE data")
#            except IOError:
#
#                self.axlist[0],  self.y_range_list[0] = self.add_plot(self.axlist[0], filename = d_backup_filename, \
#                         name = r"$T_\mathrm{rad,ECE}$", marker = r"o",color=(0,126.0/255,0),\
#                         y_range_in = self.y_range_list[0],ax_flag = ax_flag,label_x = label_x)
        if("CTC" in diags.keys() and show_measurements):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "CTC" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                if(len(Calib_file > 8)):
                    y_err, CTC_data = get_data_calib(diag=diags["CTC"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "CTC"], \
                                                 t_smooth=diags["CTC"].t_smooth)
                else:
                    y_err, CTC_data = get_data_calib(diag=diags["CTC"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "CTC"], \
                                                 t_smooth=diags["CTC"].t_smooth)
                std_dev = y_err[0]
                sys_dev = y_err[1]
                if(CTC_data is not None):
                    diag_masks["CTC"] = np.logical_and(std_dev < error_margin * CTC_data[1], CTC_data[0] < rho_max)
                    diag_masks["CTC"] = np.logical_and(diag_masks["CTC"], tau[diag_data == diags["CTC"].diag] > lower_tau_boundary)
                    # CTC_data[0][Channel_list[diag_data == "CTC"] == 0] *= -1.0
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                         data=[CTC_data[0][diag_masks["CTC"]], \
                               CTC_data[1][diag_masks["CTC"]]], \
                         y_error=sys_dev[diag_masks["CTC"]] + std_dev[diag_masks["CTC"]], \
                         marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.diag_color_index[0] += 1
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                         data=[CTC_data[0][diag_masks["CTC"]], \
                               CTC_data[1][diag_masks["CTC"]]], \
                         y_error=std_dev[diag_masks["CTC"]], \
                         name=r"$T_\mathrm{rad,OECE}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.diag_color_index[0] += 1
            except Exception as e:
                print("Could not plot CTC data")
                print("Reason", e)
        if("CTA" in diags.keys() and show_measurements):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "CTA" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                if(len(Calib_file > 8)):
                    y_err, CTA_data = get_data_calib(diag=diags["CTA"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "CTA"], \
                                                 t_smooth=diags["CTA"].t_smooth)
                else:
                    y_err, CTA_data = get_data_calib(diag=diags["CTA"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "CTA"], \
                                                 t_smooth=diags["CTA"].t_smooth)
                std_dev = y_err[0]
                sys_dev = y_err[1]
                if(CTA_data is not None):
                    # CTA_data[0][Channel_list[diag_data == "CTA"] == 0] *= -1.0
                    if(np.all(CTA_data[0] > rho_max)):
                        print("All CTA data outside rho_max")
                        print("Smallest rho", np.min(CTA_data[0]))
                    elif(np.all(std_dev[CTA_data[0] < rho_max] > error_margin * CTA_data[1][CTA_data[0] < rho_max])):
                        print("All errors larger than ", error_margin * CTA_data[1][CTA_data[0] < rho_max])
                        print("Smallest error", np.min(std_dev[CTA_data[0] < rho_max]))
                    else:
                        diag_masks["CTA"] = np.logical_and(CTA_data[0] < rho_max, np.sqrt(std_dev ** 2 + sys_dev ** 2) < error_margin * CTA_data[1])
                        diag_masks["CTA"] = np.logical_and(diag_masks["CTA"], tau[diag_data == diags["CTA"].diag] > lower_tau_boundary)
                        avg_sys_dev = np.zeros(sys_dev.shape)
                        avg_sys_dev[:] = np.mean(sys_dev[diag_masks["CTA"]] / CTA_data[1][diag_masks["CTA"]])
#                        sys_dev_mask = np.zeros(sys_dev.shape, dtype=np.bool)
#                        i_skip = 10
#                        i_lower = 0
#                        i = 0
#                        while(i < len(sys_dev_mask) and i_lower < i_skip):
#                            if(diag_masks["CTA"][i]):
#                                i_lower += 1
#                            i += 1
#                        i_upper = 0
#                        i = 0
#                        while(i < len(sys_dev_mask) and i_upper < i_skip):
#                            if(diag_masks["CTA"][-i]):
#                                i_upper += 1
#                            i += 1
#                        sys_dev_mask[i_lower - 1] = True
#                        sys_dev_mask[-i_upper + 1] = True
                        if(not Comp):
                            slicing = 5
                            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                                 data=[CTA_data[0][diag_masks["CTA"]][::slicing], CTA_data[1][diag_masks["CTA"]][::slicing]], \
                                                                                 y_error=np.sqrt(std_dev[diag_masks["CTA"]][::slicing] ** 2 + avg_sys_dev[diag_masks["CTA"]][::slicing] ** 2 * CTA_data[1][diag_masks["CTA"]][::slicing] ** 2), \
                                                                                 marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                                 y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                            self.diag_color_index[0] += 1
                            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                                 data=[CTA_data[0][diag_masks["CTA"]], CTA_data[1][diag_masks["CTA"]]], \
                                                                                 y_error=std_dev[diag_masks["CTA"]], \
                                                                                 name=r"$T_\mathrm{rad,OECE}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                                 y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                            self.diag_color_index[0] += 1
            except Exception as e:
                print("Could not plot CTA data")
                print("Reason", e)
        if("IEC" in diags.keys() and show_measurements):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "IEC" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                if(len(Calib_file > 8)):
                    y_err, IEC_data = get_data_calib(diag=diags["IEC"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "IEC"], \
                                                 t_smooth=diags["IEC"].t_smooth)
                else:
                    y_err, IEC_data = get_data_calib(diag=diags["IEC"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "IEC"], \
                                                 t_smooth=diags["IEC"].t_smooth)
                std_dev = y_err[0]
                sys_dev = y_err[1]
                if(IEC_data is not None):
                    diag_masks["IEC"] = np.logical_and(IEC_data[0] < rho_max, std_dev < error_margin * IEC_data[1])
                    diag_masks["IEC"] = np.logical_and(diag_masks["IEC"], tau[diag_data == diags["IEC"].diag] > lower_tau_boundary)
                    if(not Comp):
                    # IEC_data[0][Channel_list[diag_data == "IEC"] == 0] *= -1.0
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[IEC_data[0][diag_masks["IEC"]], IEC_data[1][diag_masks["IEC"]]], \
                                                                             y_error=np.sqrt(std_dev[diag_masks["IEC"]] ** 2 + sys_dev[diag_masks["IEC"] ** 2]), \
                                                                             marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.diag_color_index[0] += 1
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[IEC_data[0][diag_masks["IEC"]], IEC_data[1][diag_masks["IEC"]]], \
                                                                             y_error=std_dev[diag_masks["IEC"]], \
                                                                             name=r"$T_\mathrm{rad,Inline}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.diag_color_index[0] += 1
            except Exception as e:
                print("Could not plot IEC data")
                print("Reason", e)
        if("ECI" in diags.keys() and show_measurements):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "ECI" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                if(len(Calib_file > 8)):
                    y_err, ECI_data = get_data_calib(diag=diags["ECI"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "ECI"], \
                                                 t_smooth=diags["ECI"].t_smooth)
                else:
                    y_err, ECI_data = get_data_calib(diag=diags["ECI"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "ECI"], \
                                                 t_smooth=diags["ECI"].t_smooth)
            except Exception as e:
                print("Could not plot ECI data")
                print("Reason", e)
                ECI_data = None
            std_dev = y_err[0]
            sys_dev = y_err[1]
            if(ECI_data is not None):
                ECI_data[0][Channel_list[diag_data == "ECI"] == 0] *= -1.0
                diag_masks["ECI"] = np.logical_and(ECI_data[0] < rho_max, std_dev < error_margin * ECI_data[1])
                diag_masks["ECI"] = np.logical_and(diag_masks["ECI"], tau[diag_data == diags["ECI"].diag] > lower_tau_boundary)
                if(not Comp):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                         data=[ECI_data[0][diag_masks["ECI"]], \
                                                                               ECI_data[1][diag_masks["ECI"]]], \
                                                                         y_error=np.sqrt(std_dev[diag_masks["ECI"]] ** 2 + sys_dev[diag_masks["ECI"] ** 2]), \
                                                                         marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.diag_color_index[0] += 1
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                         data=[ECI_data[0][diag_masks["ECI"]], \
                                                                               ECI_data[1][diag_masks["ECI"]]], \
                                                                         y_error=std_dev[diag_masks["ECI"]], \
                                                                         name=r"$T_\mathrm{rad,ECEI}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.diag_color_index[0] += 1
        if("ECN" in diags.keys() and show_measurements):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "ECN" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                if(len(Calib_file > 8)):
                    y_err, ECI_data = get_data_calib(diag=diags["ECN"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "ECN"], \
                                                 t_smooth=diags["ECN"].t_smooth)
                else:
                    y_err, ECI_data = get_data_calib(diag=diags["ECN"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "ECN"], \
                                                 t_smooth=diags["ECN"].t_smooth)
                std_dev = y_err[0]
                sys_dev = y_err[1]
            except Exception as e:
                    print("Could not plot ECN data")
                    print("Reason", e)
                    ECI_data = None
            if(ECI_data is not None):
                diag_masks["ECN"] = np.logical_and(ECI_data[0] < rho_max, std_dev < error_margin * ECI_data[1])
                diag_masks["ECN"] = np.logical_and(diag_masks["ECN"], tau[diag_data == diags["ECN"].diag] > lower_tau_boundary)
                # ECI_data[0][Channel_list[diag_data == "ECN"] == 0] *= -1.0
                try:
                    if(not Comp):
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[ECI_data[0][diag_masks["ECN"]], \
                                                                                   ECI_data[1][diag_masks["ECN"]]], \
                                                                             y_error=np.sqrt(std_dev[diag_masks["ECN"] ** 2 + sys_dev[diag_masks["ECN"] ** 2]]), \
                                                                             marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.diag_color_index[0] += 1
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[ECI_data[0][diag_masks["ECN"]], \
                                                                                   ECI_data[1][diag_masks["ECN"]]], \
                                                                             y_error=std_dev[diag_masks["ECN"]], \
                                                                             name=r"$T_\mathrm{rad,ECEI}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.diag_color_index[0] += 1
                except:
                    print((ECI_data[1][ECI_data[0] < rho_max] * error_margin > y_err[ECI_data[0] < rho_max]).shape, y_err.shape, ECI_data[1][ECI_data[0] < rho_max].shape)
        if("ECO" in diags.keys() and show_measurements):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "ECO" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                if(len(Calib_file > 8)):
                    y_err, ECI_data = get_data_calib(diag=diags["ECO"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 sys_dev_calib=np.abs(Calib_file.T[3] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "ECO"], \
                                                 t_smooth=diags["ECO"].t_smooth)
                else:
                    y_err, ECI_data = get_data_calib(diag=diags["ECO"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=np.abs(Calib_file.T[2] / 100.0 * Calib_file.T[1]), \
                                                 ext_resonances=rhop[diag_data == "ECO"], \
                                                 t_smooth=diags["ECO"].t_smooth)
                std_dev = y_err[0]
                sys_dev = y_err[1]
            except Exception as e:
                    print("Could not plot ECO data")
                    print("Reason", e)
                    ECI_data = None
            if(ECI_data is not None):
                diag_masks["ECO"] = np.logical_and(ECI_data[0] < rho_max, std_dev < error_margin * ECI_data[1])
                diag_masks["ECO"] = np.logical_and(diag_masks["ECO"], tau[diag_data == diags["ECO"].diag] > lower_tau_boundary)
                # ECI_data[0][Channel_list[diag_data == "ECO"] == 0] *= -1.0
                if(not Comp):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                         data=[ECI_data[0][diag_masks["ECO"]], \
                                                                               ECI_data[1][diag_masks["ECO"]]], \
                                                                         y_error=np.sqrt(std_dev[diag_masks["ECO"]] ** 2 + sys_dev[diag_masks["ECO"] ** 2]), \
                                                                         marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.diag_color_index[0] += 1
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                         data=[ECI_data[0][diag_masks["ECO"]], \
                                                                               ECI_data[1][diag_masks["ECO"]]], \
                                                                         y_error=std_dev[diag_masks["ECO"]], \
                                                                         name=r"$T_\mathrm{rad,ECEI}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                                                                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.diag_color_index[0] += 1
        if("TS_c" in diags.keys() and show_measurements and not Comp):
            if(Te):
                y_err, TS_c_data = get_Thomson_data(shotno, time, diags["TS_c"], Te=True, ne=False, edge=False, core=True, \
                                                    EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
                # print("Error margin and errors for core Thomson Te")
                # print(y_err, error_margin * TS_c_data[1])
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                         data=[TS_c_data[0][y_err < error_margin * TS_c_data[1]], \
                               TS_c_data[1][y_err < error_margin * TS_c_data[1]] * 1.e-3], \
                         y_error=y_err[y_err < error_margin * TS_c_data[1]] * 1.e-3, \
                         name=r"$T_\mathrm{e,TS\,core}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                         y_range_in=self.y_range_list[0], ax_flag="Te", label_x=label_x)
                self.diag_color_index[0] += 1
            if(ne):
                y_err, TS_c_data = get_Thomson_data(shotno, time, diags["TS_c"], Te=False, ne=True, edge=False, core=True, \
                                                    EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                         data=[TS_c_data[0][y_err < error_margin * TS_c_data[1]], \
                               TS_c_data[1][y_err < error_margin * TS_c_data[1]] * 1.e-20], \
                         y_error=y_err[y_err < error_margin * TS_c_data[1]] * 1.e-20, \
                         name=r"$n_\mathrm{e,TS\,core}$", marker=self.diag_markers[self.diag_marker_index[1]], color=self.diag_colors[self.diag_color_index[1]], \
                         y_range_in=self.y_range_list[1], ax_flag="ne", label_x=label_x)
                self.diag_color_index[1] += 1
        if("TS_e" in diags.keys() and show_measurements and not Comp):
            if(Te):
                y_err, TS_c_data = get_Thomson_data(shotno, time, diags["TS_e"], Te=True, ne=False, edge=True, core=False, \
                                                    EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                         data=[TS_c_data[0][y_err < error_margin * TS_c_data[1]], \
                               TS_c_data[1][y_err < error_margin * TS_c_data[1]] * 1.e-3], \
                         y_error=y_err[y_err < error_margin * TS_c_data[1]] * 1.e-3, \
                         name=r"$T_\mathrm{e,TS\,edge}$", marker=self.diag_markers[self.diag_marker_index[0]], color=self.diag_colors[self.diag_color_index[0]], \
                         y_range_in=self.y_range_list[0], ax_flag="Te", label_x=label_x)
                self.diag_color_index[0] += 1
            if(ne):
                y_err, TS_c_data = get_Thomson_data(shotno, time, diags["TS_e"], Te=False, ne=True, edge=True, core=False, \
                                                    EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                         data=[TS_c_data[0][y_err < error_margin * TS_c_data[1]], \
                               TS_c_data[1][y_err < error_margin * TS_c_data[1]] * 1.e-20], \
                         y_error=y_err[y_err < error_margin * TS_c_data[1]] * 1.e-20, \
                         name=r"$n_\mathrm{e,TS\,edge}$", marker=self.diag_markers[self.diag_marker_index[1]], color=self.diag_colors[self.diag_color_index[1]], \
                         y_range_in=self.y_range_list[1], ax_flag="ne", label_x=label_x)
                self.diag_color_index[1] += 1
        if(len(diags.keys()) > 0 and show_measurements and not Comp):
            self.diag_marker_index[0] += 1
        if(IDA_model and "ECE" in diags.keys()):
            try:
                ECE_residue = np.loadtxt(d_abs_filename)
                # ECE_residue.T[0][Channel_list[(diag_data == diags["ECE"].diag) | (diag_data == fall_back_diag)] == 0] *= -1.0
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                         data=[ECE_residue.T[0][0:len((diag_data == diags["ECE"].diag) | \
                                (diag_data == fall_back_diag))][ECE_residue.T[0][0:len((diag_data == diags["ECE"].diag) | \
                                (diag_data == fall_back_diag))] < rho_max], ECE_residue.T[2][0:len((diag_data == diags["ECE"].diag) | \
                                (diag_data == fall_back_diag))][ECE_residue.T[0][0:len((diag_data == diags["ECE"].diag) | \
                                (diag_data == fall_back_diag))] < rho_max]], \
                         name=r"$T_\mathrm{rad,IDA}$", marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                         y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)  # (0.0,0.3,0.0)
                self.model_marker_index[0] += 1
                self.model_color_index[0] += 1
            except IOError:
                print("Could not find IDA model data at:", d_abs_filename)

        # print(dstf)
        # if(dstf != "Th"):
        #    self.axlist[0],  self.y_range_list[0] = self.add_plot(self.axlist[0], filename = t_th_filname, \
        #                                                      name = r"$T_\mathrm{rad}$ direct", coloumn = 1, marker = "d",color=(0.5,0.0,0.5),\
        #

        # try:
        #    #pass
        #    self.axlist[0],  self.y_range_list[0] = self.add_plot(self.axlist[0], filename = m_abs_filename, \
        #        name = r"$T_\mathrm{rad}$ New ECRad using Rk4", marker = "D",color=(1.0,0.25,0.0), y_range_in = self.y_range_list[0])
        # except:
        #    pass
        # try:
            # pass
        #     self.axlist[0],  self.y_range_list[0] = self.add_plot(self.axlist[0], filename =te_reduced_filename, \
        #         name = r"$T_\mathrm{e}$ interpolated to 80 scaled by 2.0", marker = "-",color=(0.0,0.25,1.0), y_range_in = self.y_range_list[0], y_scale = 2.0)
        #     self.axlist[1],  self.y_range_list[1] = self.add_plot(self.axlist[1], filename = ne_reduced_filename, \
        #     name = "electron density interpolated to 80 scaled by 2.0", marker = "-",color=(0.0,0.0,1.0), y_range_in = self.y_range_list[1], ax_flag = "ne", y_scale = 0.2)
        # except:
        #     pass
        if(model):
            try:
                model_TRad = model_data.T[1]
                model_tau = model_data.T[2]
                if(simpl is None):
                    dist = ""
                for diag_key in diags.keys():
                    if(diag_key == "ECE" and diag_key not in diag_masks.keys()):
                        diag_masks[diag_key] = np.logical_and(rhop[ECE_diag_mask] < rho_max, model_tau[ECE_diag_mask] > lower_tau_boundary)
                    elif(diag_key not in diag_masks.keys()):
                        diag_masks[diag_key] = np.logical_and(rhop[diag_key == diag_data] < rho_max, model_tau[diag_key == diag_data] > lower_tau_boundary)
                    if(dist != "f_\mathrm{MJ}" and dist != "ECRad" and diag_key == "ECE"):
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]],
                                                                                   model_TRad[ECE_diag_mask][diag_masks[diag_key]]], \
                                                                             name=r"$T_\mathrm{rad,mod}" + dist + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)  # (0.0,0.0,1.0)
                        self.model_color_index[0] += 1
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]],
                                                                                   np.exp(-model_tau[ECE_diag_mask][diag_masks[diag_key]])], \
                                                                             name=r"$T_{\omega,\mathrm{mod}}" + dist + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)  # (0.0,0.0,1.0)
                        self.model_color_index_2[0] += 1
                    elif(diag_key == "ECE"):
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]], model_TRad[ECE_diag_mask][diag_masks[diag_key]]], \
                                                                             name=r"$T_\mathrm{rad,mod}" + dist + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.model_color_index[0] += 1
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]], np.exp(-model_tau[ECE_diag_mask][diag_masks[diag_key]])], \
                                                                             name=r"$T_{\omega,\mathrm{mod}}" + dist + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                        self.model_color_index_2_[0] += 1
                    elif(len(rhop[diag_data == diag_key] > 0) and show_measurements):
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                            data=[rhop[diag_data == diag_key][diag_masks[diag_key]], \
                                    model_TRad[diag_data == diag_key][diag_masks[diag_key]]], \
                            name=r"$T_\mathrm{rad,mod} \,\mathrm{" + diag_key + r"} " + dist + r"$" + mode_str, \
                            marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                            y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.model_color_index[0] += 1
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                            data=[rhop[diag_data == diag_key][diag_masks[diag_key]], \
                                  np.exp(-model_tau[diag_data == diag_key][diag_masks[diag_key]])], \
                            name=r"$T_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + dist + r"$" + mode_str, \
                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                            y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                        self.model_color_index_2[0] += 1
                    elif(len(rhop[diag_data == diag_key]) > 0):
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                            data=[rhop[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)], \
                                  model_TRad[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)]], \
                            name=r"$T_\mathrm{rad,mod}\,\mathrm{" + diag_key + r"}" + dist + r"$" + mode_str, \
                            marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                            y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.model_color_index[0] += 1
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                data=[rhop[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)], \
                                      np.exp(-model_tau[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)])], \
                                name=r"$T_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + dist + r"$" + mode_str, \
                                marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                                y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                        self.model_color_index_2[0] += 1
#                    elif(len(rhop[diag_data == diag_key] > 0)):
#                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                            data=[rhop[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_TRad[diag_data == diag_key] > lower_tau_boundary)], \
#                                    model_TRad[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)]], \
#                            name=r"$T_\mathrm{rad,mod} \,\mathrm{" + diag_key + r"} " + dist + r"$", \
#                            marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
#                            y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
#                        self.model_color_index[0] += 1
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                            data=[rhop[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)], \
#                                    model_TRad[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)]], \
#                            name=r"$\tau_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + dist + r"$", \
#                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
#                            y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
#                        self.model_color_index_2[0] += 1
                self.model_marker_index[0] += 1
            except Exception as e:
                print(e, " occured when reading nth radiation profile")
                print("Filename: " + m_abs_filename)
        if(simpl is not None):
            # try:

            simpl_model_data = np.loadtxt(m_simpl_filename)
            simpl_rhop = simpl_model_data.T[0]
            if(not rel_res and harmonic_number != 2):
                simpl_rhop = rhop
            simpl_rhop[Channel_list == 0] *= -1.0
            simpl_model_TRad = simpl_model_data.T[1]
            simpl_model_tau = simpl_model_data.T[2]
            for diag_key in diags.keys():
                if(diag_key == "ECE"):
                    diag_masks[diag_key] = np.logical_and(rhop[ECE_diag_mask] < rho_max, model_tau[ECE_diag_mask] > lower_tau_boundary)
                elif(diag_key not in diag_masks.keys()):
                    diag_masks[diag_key] = np.logical_and(rhop[diag_key == diag_data] < rho_max, model_tau[diag_key == diag_data] > lower_tau_boundary)
                if((dist == "f_\mathrm{MJ}" or dist == "ECRad") and diag_key == "ECE"):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        data=[simpl_rhop[ECE_diag_mask][diag_masks[diag_key]], \
                        simpl_model_TRad[ECE_diag_mask][diag_masks[diag_key]]], \
                        name=r"$T_\mathrm{rad,mod}" + dist_simpl + r"$" + mode_str, \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.model_color_index[0] += 1
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]],
                                                                                   np.exp(-simpl_model_tau[ECE_diag_mask][diag_masks[diag_key]])], \
                                                                             name=r"$T_{\omega,\mathrm{mod}}" + dist_simpl + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)  # (0.0,0.0,1.0)
                    self.model_color_index_2[0] += 1
                elif(diag_key == "ECE"):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        data=[simpl_rhop[ECE_diag_mask][diag_masks[diag_key]], \
                              simpl_model_TRad[ECE_diag_mask][diag_masks[diag_key]]], \
                        name=r"$T_\mathrm{rad,mod}" + dist_simpl + r"$" + mode_str, \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)  # color=(1.0,0.0,0.0)
                    self.model_color_index[0] += 1
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]], \
                                                                                   np.exp(-simpl_model_tau[ECE_diag_mask][diag_masks[diag_key]])], \
                                                                             name=r"$T_{\omega,\mathrm{mod}}" + dist_simpl + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                    self.model_color_index_2[0] += 1
                elif(len(rhop[diag_data == diag_key] > 0) and show_measurements):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        data=[simpl_rhop[diag_data == diag_key][diag_masks[diag_key]], \
                        simpl_model_TRad[diag_data == diag_key][diag_masks[diag_key]]], \
                        name=r"$T_\mathrm{rad,mod}\,\mathrm{" + diag_key + r"}" + dist_simpl + r"$" + mode_str, \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.model_color_index[0] += 1
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                            data=[rhop[diag_data == diag_key][diag_masks[diag_key]], \
                                  np.exp(-simpl_model_tau[diag_data == diag_key][diag_masks[diag_key]])], \
                            name=r"$T_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + dist_simpl + r"$" + mode_str, \
                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                            y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                    self.model_color_index_2[0] += 1
                elif(len(rhop[diag_data == diag_key]) > 0):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        data=[simpl_rhop[diag_data == diag_key][np.logical_and(simpl_rhop[diag_data == diag_key] < rho_max, simpl_model_data.T[2][diag_data == diag_key] > lower_tau_boundary)], \
                        simpl_model_TRad[diag_data == diag_key][np.logical_and(simpl_rhop[diag_data == diag_key] < rho_max, simpl_model_data.T[2][diag_data == diag_key] > lower_tau_boundary)]], \
                        name=r"$T_\mathrm{rad,mod}\,\mathrm{" + diag_key + r"}" + dist_simpl + r"$" + mode_str, \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                    self.model_color_index[0] += 1
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                            data=[rhop[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)], \
                                  np.exp(-simpl_model_tau[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)])], \
                            name=r"$T_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + dist_simpl + r"$" + mode_str, \
                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                            y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                    self.model_color_index_2[0] += 1
                if(dstf == "TB" and diag_key == "ECE"):
                    try:
                        model_data_TO = np.loadtxt(os.path.join(folder, "ECRad_data", "TRadM_thrms.dat"))
                        rhop_TO = model_data.T[0]
                        rhop_TO[Channel_list == 0] *= -1.0
                        model_TRad_TO = model_data_TO.T[1]
                        model_tau_TO = model_data_TO.T[1]
                        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                            data=[rhop_TO[ECE_diag_mask][diag_masks[diag_key]], \
                                  model_TRad_TO[ECE_diag_mask][diag_masks[diag_key]]], \
                            name=r"$T_\mathrm{rad,mod}\left[\mathrm{Imp.\,Hutchinson}\right]  $", \
                            marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                            y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                        self.model_color_index[0] += 1
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                            data=[rhop_TO[ECE_diag_mask][diag_masks[diag_key]], \
                                  np.exp(-model_tau_TO[ECE_diag_mask][diag_masks[diag_key]])], \
                            name=r"$T_{\omega,\mathrm{mod}}\left[\mathrm{Imp.\,Hutchinson}\right]  $", \
                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                            y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                        self.model_color_index_2[0] += 1
                    except IOError as e:
                        print("Could not find imp. Hutchinson - proceeding without it")
                        print(e)
            self.model_marker_index[0] += 1
        if(use_tertiary_model):
            try:
                tertiary_model = os.path.join(folder, "ECRad_data", tertiary_model)
                tertiary_model_data = np.loadtxt(tertiary_model)
                tertiary_rhop = tertiary_model_data.T[0]
                tertiary_model_TRad = tertiary_model_data.T[1]
                tertiary_model_tau = tertiary_model_data.T[2]
                if(not rel_res and harmonic_number != 2):
                    tertiary_rhop = rhop
                tertiary_rhop[Channel_list == 0] *= -1.0
                if(len(tertiary_rhop) == len(rhop)):
                    for diag_key in diags.keys():
                        if(diag_key == "ECE"):
                            diag_masks[diag_key] = np.logical_and(rhop[ECE_diag_mask] < rho_max, model_tau[ECE_diag_mask] > lower_tau_boundary)
                        elif(diag_key not in diag_masks.keys()):
                            diag_masks[diag_key] = np.logical_and(rhop[diag_key == diag_data] < rho_max, model_tau[diag_key == diag_data] > lower_tau_boundary)
                        if(tertiary_dist != "f_\mathrm{MJ}" and tertiary_dist != "ECRad" and diag_key == "ECE"):
                            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                data=[tertiary_rhop[ECE_diag_mask][diag_masks[diag_key]], \
                                      tertiary_model_TRad[ECE_diag_mask][diag_masks[diag_key]]], \
                                name=r"$T_\mathrm{rad,mod}" + tertiary_dist + r"$",
                                marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)  # (0.0,0.0,1.0)
                            self.model_color_index[0] += 1
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]],
                                                                                   tertiary_model_tau[ECE_diag_mask][diag_masks[diag_key]]], \
                                                                             name=r"$\tau_{\omega,\mathrm{mod}}" + tertiary_dist + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)  # (0.0,0.0,1.0)
                            self.model_color_index_2[0] += 1
                        elif(diag_key == "ECE"):
                            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                data=[tertiary_rhop[ECE_diag_mask][diag_masks[diag_key]], \
                                      tertiary_model_TRad[ECE_diag_mask][diag_masks[diag_key]]], \
                                name=r"$T_\mathrm{rad,mod}" + tertiary_dist + +r"$" + mode_str, \
                                markers=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                            self.model_color_index[0] += 1
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                                                             data=[rhop[ECE_diag_mask][diag_masks[diag_key]], \
                                                                                   tertiary_model_tau[ECE_diag_mask][diag_masks[diag_key]]], \
                                                                             name=r"$\tau_{\omega,\mathrm{mod}}" + tertiary_dist + r"$" + mode_str, \
                                                                             marker=self.model_markers[self.model_marker_index_2[0]], \
                                                                             color=self.model_colors[self.model_color_index_2[0]], \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
                            self.model_color_index_2_[0] += 1
                        elif(len(tertiary_rhop[diag_data == diag_key]) > 0):
                            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                data=[tertiary_rhop[diag_data == diag_key][diag_masks[diag_key]], \
                                      tertiary_model_TRad[diag_data == diag_key][diag_masks[diag_key]]], \
                                name=r"$T_\mathrm{rad,mod} \,\mathrm{" + diag_key + r"} " + tertiary_dist + r"$" + mode_str, \
                                marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
                            self.model_color_index[0] += 1
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                            data=[rhop[diag_data == diag_key][diag_masks[diag_key]], \
                                  tertiary_model_tau[diag_data == diag_key][diag_masks[diag_key]]], \
                            name=r"$T_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + tertiary_dist + r"$" + mode_str, \
                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
                            y_range_in=self.y_range_list_2[0], ax_flag="T_rho", label_x=label_x)
                            self.model_color_index_2[0] += 1
#                        elif(len(tertiary_rhop[diag_data == diag_key] > 0)):
#                            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                                data=[tertiary_rhop[diag_data == diag_key][tertiary_rhop[diag_data == diag_key] < rho_max], \
#                                      tertiary_model_TRad [diag_data == diag_key][np.logical_and(tertiary_rhop[diag_data == diag_key] < rho_max, tertiary_model_data.T[2][diag_data == diag_key] > lower_tau_boundary)]], \
#                                name=r"$T_\mathrm{rad,mod} \,\mathrm{" + diag_key + r"} " + tertiary_dist + r"$", \
#                                marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
#                                y_range_in=self.y_range_list[0], ax_flag=ax_flag, label_x=label_x)
#                            self.model_color_index[0] += 1
#                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                            data=[rhop[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)], \
#                                  tertiary_model_tau[diag_data == diag_key][np.logical_and(rhop[diag_data == diag_key] < rho_max, model_tau[diag_data == diag_key] > lower_tau_boundary)]], \
#                            name=r"$\tau_{\omega,\mathrm{mod}} \,\mathrm{" + diag_key + r"} " + tertiary_dist + r"$", \
#                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
#                            y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
#                            self.model_color_index_2[0] += 1
            except IOError as e:
                print(e, " occured when reading tertiary_model radiation profile")
                print("Filename: " + tertiary_model)
        if(rel_diff or rel_diff_comp):
            if(rel_diff_comp):
                comp_data = np.loadtxt(os.path.join(rel_diff_folder, "ECRad_data", data_name))
                comp_Trad = comp_data.T[1]
                if(comp_Trad.shape != model_TRad.shape):
                    print("Cannot calculate relative deviation size not same amount of channels")
                    rel_diff_data = []
                else:
                    rel_diff_data = 200.0 * np.abs(model_TRad - comp_Trad) / (model_TRad + comp_Trad)
            elif(simpl is not None):
                rel_diff_data = 200.0 * np.abs(model_TRad - simpl_model_TRad) / (model_TRad + simpl_model_TRad)
#            for i in range(len(model_TRad)):
#                print(rhop[i], model_TRad[i], simpl_model_TRad[i], rel_diff_data[i])
            ax_flag = "rel_diff_Trad"
            for diag_key in diags.keys():
                if((dist == "f_\mathrm{MJ}" or dist == "ECRad") and diag_key == "ECE"):
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                        data=[rhop[(diag_data == diags[diag_key].diags["ECE"].diag) | (diag_data == fall_back_diag)][rhop < rho_max], \
                        rel_diff_data[(diag_data == diags[diag_key].diag) | (diag_data == fall_back_diag)][rhop < rho_max]], \
                        name=r"$\Delta T_\mathrm{rad,mod}$", \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[1], ax_flag=ax_flag, label_x=label_x)  # maxlines = 45,
                    self.model_color_index[0] += 1
                elif(diag_key == "ECE"):
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                        data=[rhop[(diag_data == diags[diag_key].diag) | (diag_data == fall_back_diag)][rhop[(diag_data == diags["ECE"].diag) | \
                                                                                                         (diag_data == fall_back_diag)] < rho_max], \
                              rel_diff_data[(diag_data == diags[diag_key].diag) | (diag_data == fall_back_diag)][rhop[ECE_diag_mask] < rho_max]], \
                        name=r"$\Delta T_\mathrm{rad,mod}$", \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[1], ax_flag=ax_flag, label_x=label_x)  # color=(1.0,0.0,0.0)
                    self.model_color_index[0] += 1
                elif(len(rhop[diag_data == diag_key] > 0)):
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                        data=[rhop[diag_data == diag_key][rhop[diag_data == diag_key] < rho_max], \
                        rel_diff_data[diag_data == diag_key][rhop[diag_data == diag_key] < rho_max]], \
                        name=r"$\Delta T_\mathrm{rad,mod}\,\mathrm{" + diag_key + r"}$", \
                        marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                        y_range_in=self.y_range_list[1], ax_flag=ax_flag, label_x=label_x)
                    self.model_color_index[0] += 1
            self.model_marker_index[0] += 1
        elif(X_frac):
            if(model):
                ax_flag = "X_frac"
                X_frac_data = np.loadtxt(data_name_X_frac)
                # rhop_X_frac = X_frac_data.T[0] # rhop_X_frac often zero- rather use rhop
                rhop_X_frac = rhop
                X_frac_model = X_frac_data.T[3] * 100.e0
                for diag_key in diags.keys():
                    if(diag_key == "ECE"):
                        if(len(rhop_X_frac[diag_data == diags[diag_key].diag] > 0)):
                            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                                data=[rhop_X_frac[diag_data == diags[diag_key].diag][rhop_X_frac[diag_data == diags[diag_key].diag] < rho_max], \
                                X_frac_model[diag_data == diags[diag_key].diag][rhop_X_frac[diag_data == diags[diag_key].diag] < rho_max]], \
                                name=r"$X_\mathrm{frac}\,\mathrm{" + diag_key + r"}$", \
                                marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                y_range_in=self.y_range_list[1], ax_flag=ax_flag, label_x=label_x)
                            self.model_color_index[0] += 1
                    else:
                        if(len(rhop[diag_data == diag_key] > 0)):
                            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                                data=[rhop_X_frac[diag_data == diag_key][rhop_X_frac[diag_data == diag_key] < rho_max], \
                                X_frac_model[diag_data == diag_key][rhop_X_frac[diag_data == diag_key] < rho_max]], \
                                name=r"$X_\mathrm{frac}\,\mathrm{" + diag_key + r"}$", \
                                marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                                y_range_in=self.y_range_list[1], ax_flag=ax_flag, label_x=label_x)
                            self.model_color_index[0] += 1
                self.model_marker_index[0] += 1
            if(simpl):
                ax_flag = "X_frac"
                X_frac_data = np.loadtxt(simplfile_X_frac)
                rhop_X_frac_simpl = X_frac_data.T[0]
                X_frac_simpl = X_frac_data.T[3] * 100.e0
                for diag_key in diags.keys():
                    if(len(rhop[diag_data == diag_key] > 0)):
                        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                            data=[rhop_X_frac_simpl[diag_data == diag_key][rhop_X_frac_simpl[diag_data == diag_key] < rho_max], \
                            X_frac_simpl[diag_data == diag_key][rhop_X_frac_simpl[diag_data == diag_key] < rho_max]], \
                            name=r"$X_\mathrm{frac}\,\mathrm{" + diag_key + r"}$", \
                            marker=self.model_markers[self.model_marker_index[0]], color=self.model_colors[self.model_color_index[0]], \
                            y_range_in=self.y_range_list[1], ax_flag=ax_flag, label_x=label_x)
                        self.model_color_index[0] += 1
                self.model_marker_index[0] += 1
        if(ne):
            plt.setp(self.axlist[0].get_xticklabels(), visible=False)
        self.axlist_2[0].set_ylim(0.0, 1.0)
        return diag_masks
#        try:
#            if(model):
#                Trad_data = np.loadtxt(m_abs_filename)
#                for diag_key in diags.keys():
#                    if(diag_key == "ECE"):
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[Trad_data.T[0][(diag_data == diags[diag_key].diag) | \
#                            (diag_data == fall_back_diag)], Trad_data.T[2][(diag_data == diags[diag_key].diag) | (diag_data == fall_back_diag)]], \
#                            name=r"$\tau_{\omega,\mathrm{mod}}" + dist + r"$ " + diag_key, \
#                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
#                            y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
#                        self.model_color_index_2[0] += 1
#                    if(len(rhop[diag_data == diag_key] > 0)):
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                            data=[rhop[diag_data == diag_key][rhop[diag_data == diag_key] < rho_max], \
#                                    Trad_data.T[2][diag_data == diag_key][rhop[diag_data == diag_key] < rho_max]], \
#                            name=r"$\tau_{\omega,\mathrm{mod}}" + dist + r"$ " + diag_key, \
#                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
#                            y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
#                        self.model_color_index_2[0] += 1
#                self.model_marker_index_2[0] += 1
#            if(simpl):
#                if(os.path.isfile(m_simpl_filename) or backup_name is None):
#                    Trad_simpl_data = np.loadtxt(m_simpl_filename)
#                else:
#                    Trad_simpl_data = np.loadtxt(backup_name)
#                    dist_simpl = backup_dist
#                for diag_key in diags.keys():
#                    if(diag_key == "ECE"):
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[Trad_simpl_data.T[0][(diag_data == diags[diag_key].diag) | \
#                            (diag_data == fall_back_diag)], Trad_simpl_data.T[2][(diag_data == diags[diag_key].diag) | (diag_data == fall_back_diag)]], \
#                            name=r"$\tau_{\omega,\mathrm{mod}}" + dist_simpl + r"$ " + diag_key, \
#                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
#                            y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
#                        self.model_color_index_2[0] += 1
#                    elif(len(rhop[diag_data == diag_key] > 0)):
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                            data=[rhop[diag_data == diag_key][rhop[diag_data == diag_key] < rho_max], \
#                                    Trad_simpl_data.T[2][diag_data == diag_key][rhop[diag_data == diag_key] < rho_max]], \
#                            name=r"$\tau_{\omega,\mathrm{mod}}" + dist_simpl + r"$ " + diag_key, \
#                            marker=self.model_markers[self.model_marker_index_2[0]], color=self.model_colors[self.model_color_index_2[0]], \
#                            y_range_in=self.y_range_list_2[0], ax_flag="tau", label_x=label_x)
#                        self.model_color_index_2[0] += 1
#                self.model_marker_index_2[0] += 1
#        except Exception as e:
#            print("Plotting failed")
#            print("Reason: ", e)
#            pass
#        self.axlist[0].set_ylim(0, 10.0)

    def plot_Trad(self, time, rhop, Trad, Trad_comp, rhop_Te, Te, \
                  diags, diag_names, dstf, model_2=True, \
                  X_mode_fraction=None, X_mode_fraction_comp=None):
        if(X_mode_fraction is not None):
            twinx = "_twinx"
        else:
            twinx = ""
        if(plot_mode == "Software"):
            self.setup_axes("Te_no_ne" + twinx, r"$T_{rad}$, $T_{e}$ ", r"Optical depth $\tau_\omega$")
        else:
            self.setup_axes("Te_no_ne" + twinx, r"$T_\mathrm{rad}$, $T_\mathrm{e}$", \
                            r"Optical depth $\tau_\omega$")
        mathrm = r"\mathrm"
        if(dstf == "Mx"):
            dist_simpl = r"M"
            dist = r"MJ"
            dist = r"[" + dist + r"]"
        elif(dstf == "TB"):
            dist_simpl = r"Fa"
            dist = r"Alb"
            dist = r"[" + dist + r"]"
        elif(dstf == "Th"):
            dist_simpl = r"Hu"
            dist = r"Alb"
            dist = r"[" + dist + r"]"
        elif(dstf == "Re"):
            dist_simpl = r"Fa"
            dist = r"RELAX"
            dist = r"[" + dist + r"]"
        elif(dstf == "Ge"):
            dist_simpl = r"Background"
            dist = r"GENE"
            dist = r"[" + dist + r"]"
        elif(dstf == "GB"):
            dist_simpl = r"Background"
            dist = r"BiMax"
            dist = r"[" + dist + r"]"
        if(len(dist_simpl) > 0):
            dist_simpl = r"[" + dist_simpl + r"]"
        ax_flag = "Te_Te_Trad"
        # No ne as of yet                                       #    y_range_in = self.y_range_list[0])  \,R = 0.90 $
#        self.axlist[1],  self.y_range_list[1] = self.add_plot(self.axlist[1], data = [rhop_ne_vec, ne_vec], \
#            name = r"IDA $n_\mathrm{e}$", marker = "-",color=(0.0,0.0,0.0), y_range_in = self.y_range_list[1], ax_flag = "ne")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[rhop_Te, Te], \
                                                             name=r"$T_" + mathrm + "{e}$", coloumn=1, \
                                                             marker="-", color=(0.0, 0.0, 0.0), \
                                                             y_range_in=self.y_range_list[0], y_scale=1.0, \
                                                             ax_flag=ax_flag)  # \times 100$
        mask = np.zeros(len(Trad), dtype=np.bool)
        if(len(diags) > 0):
            mask[:] = False
            for diag in diags:
                # For some reason there is no automatic cast from unicode to string here -> make it explicit
                mask[str(diag.name)==diag_names] = True
        else:
            mask[:] = True
        if(model_2 and len(Trad_comp) > 0):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[rhop[mask], Trad_comp[mask]], \
                name=r"$T_" + mathrm + "{rad,mod}" + dist_simpl + r"$", \
                marker="s", color=(126.0 / 255, 0.0, 126.0 / 255), \
                y_range_in=self.y_range_list[0], ax_flag=ax_flag)
            if(X_mode_fraction_comp is not None):
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                                                                     data=[rhop[mask], X_mode_fraction_comp[mask] * 1.e2], \
                                                                     name=r"X-mode fraction $" + dist_simpl + r"$", \
                                                                     marker="o", color=(0.0, 0.2, 0.2e0), \
                                                                     y_range_in=self.y_range_list[1], ax_flag="X_frac")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            data=[rhop[mask], Trad[mask]], \
            name=r"$T_" + mathrm + "{rad,mod}" + dist + r"$", \
            marker="v", color=(126.0 / 255, 126.0 / 255, 0.e0), \
            y_range_in=self.y_range_list[0], ax_flag=ax_flag)
        if(X_mode_fraction is not None):
            # percent
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                data=[rhop[mask], X_mode_fraction[mask] * 1.e2], \
                name=r"X-mode fraction $" + dist + r"$", \
                marker="+", color=(0.0, 0.0, 0.e0), \
                y_range_in=self.y_range_list[1], ax_flag="X_frac")
        for diag in diags:
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[diag.rhop, diag.val], y_error=diag.unc, \
                name=diag.name, marker=self.diag_markers[self.diag_marker_index[0]], \
                color=self.diag_colors[self.diag_color_index[0]], \
                y_range_in=self.y_range_list[0], ax_flag=ax_flag)
            self.diag_marker_index[0] += 1
            if(self.diag_marker_index[0] > len(self.diag_markers)):
                print("Warning too many diagnostics to plot - ran out of unique markers")
                self.diag_color_index[0] = 0
            self.diag_color_index[0] += 1
            if(self.diag_color_index[0] > len(self.diag_colors)):
                print("Warning too many diagnostics to plot - ran out of unique colors")
                self.diag_color_index[0] = 0
            
        self.create_legends("Te_no_ne" + twinx)
        return self.fig

    def plot_tau(self, time, rhop, tau, tau_comp, rhop_IDA, Te_IDA, dstf, model_2=True):
        if(plot_mode == "Software"):
            self.setup_axes("twinx", r"$\tau_{\omega}$, $T_{e}$ ", r"Optical depth $\tau_\omega$")
        else:
            self.setup_axes("twinx", r"$\tau_{\omega}$, $T_\mathrm{e}$", \
                            r"Optical depth $\tau_\omega$")
        mathrm = r"\mathrm"
        if(dstf == "Mx"):
            dist_simpl = r"M"
            dist = r"MJ"
            dist = r"[" + dist + r"]"
        elif(dstf == "TB"):
            dist_simpl = r"Fa"
            dist = r"Alb"
            dist = r"[" + dist + r"]"
        elif(dstf == "Th"):
            dist_simpl = r"Hu"
            dist = r"Alb"
            dist = r"[" + dist + r"]"
        elif(dstf == "Re"):
            dist_simpl = r"Fa"
            dist = r"RELAX"
            dist = r"[" + dist + r"]"
        if(len(dist_simpl) > 0):
            dist_simpl = r"[" + dist_simpl + r"]"
        ax_flag = "T_rho"
        if(model_2 and tau_comp is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[rhop, np.exp(-tau_comp)], \
                name=r"$T_" + mathrm + "{mod}" + dist_simpl + r"$", \
                marker="s", color=(126.0 / 255, 0.0, 126.0 / 255), \
                y_range_in=self.y_range_list[0], ax_flag=ax_flag)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            data=[rhop, np.exp(-tau)], \
            name=r"$T_" + mathrm + "{mod}" + dist + r"$", \
            marker="v", color=(126.0 / 255, 126.0 / 255, 0.e0), \
            y_range_in=self.y_range_list[0], ax_flag=ax_flag)
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ rhop_IDA, Te_IDA], \
            name=r"$T_" + mathrm + "{e}$", coloumn=1, marker="-", color=(0.0, 0.0, 0.0), \
                 y_range_in=self.y_range_list[1], y_scale=1.0, ax_flag="Te")  # \times 100$
        self.create_legends("BDP")
        if(len(rhop_IDA) > 0):
            self.axlist[0].set_xlim(0.0, 1.05 * np.max([np.max(rhop_IDA), np.max(rhop)]))
        self.axlist[0].set_ylim(0.0, 1.00)
        return self.fig, self.fig_2

    def plot_1D_cpo(self, x, y, shot, time, x_info, y_info):
        self.setup_axes("single", r"AUG data for \# {0:d}".format(shot) + r" and $t = \SI{" + r"{0:2.2f}".format(time) + "}{\second}$")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], self.y_range_list[0], \
            data=[x, y], xlabel=x_info[0], ylabel=y_info[0], x_scale=x_info[1], y_scale=y_info[1])

    def plot_2D_cpo(self, x, y, z, shot, time, x_info, y_info, z_info):
        self.setup_axes("single", r"AUG data for \# {0:d}".format(shot) + r" and $t = \SI{" + r"{0:2.2f}".format(time) + "}{\second}$")
        cmap = plt.cm.get_cmap("jet")
        z_max = np.max(z.flatten())
        z_min = np.min(z.flatten())
        cont1 = self.axlist[0].contourf(x / x_info[1], y / y_info[1], z.T / z_info[1], levels=np.linspace(z_min, z_max, 30), cmap=cmap)  # ,norm = LogNorm()
        cont2 = self.axlist[0].contour(x / x_info[1], y / y_info[1], z.T / z_info[1], levels=np.linspace(z_min, z_max, 30), colors='k',
                            hold='on', alpha=0.25, linewidths=1)
        self.axlist[0].set_xlabel(x_info[0])
        self.axlist[0].set_ylabel(y_info[0])
        for c in cont2.collections:
            c.set_linestyle('solid')
        cb = self.fig.colorbar(cont1, ax=self.axlist[0])  # ticks = levels[::6] #,
        cb.set_label(z_info[0])  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
        # cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
        cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([z_min, z_max, 5])))
        cb.ax.minorticks_on()
        if(AUG):
            fconf.plt_vessel(self.axlist[0])
#        steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
#        steps_y = steps
#        self.axlist[0].get_xaxis().set_major_locator(MaxNLocator(nbins=4, steps=steps, prune='lower'))
#        self.axlist[0].get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 4.0))
#        self.axlist[0].get_yaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps_y))
#        self.axlist[0].get_yaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps_y / 4.0))

    def ECEI_plot(self, folder, shotno, time, dstf=None, simpl=False, ne=True, diags=None, model=True, \
                  rel_res=False, Img_diag="ECN", plot_rays=False, eq_diag=None):
        dist = r""
        EQ_obj = EQData(shotno, EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
        print("using ..", folder)
        if(diags is None):
            print("No diags for ECEI_plot - returning!")
            return self.fig, self.fig_2
        self.setup_axes("single", r"$T_\mathrm{rad,ECEI}$ " + \
                    str(shotno) + ", $t = \SI{" + "{0:1.2f}".format(float(time)) + "}{\second}$", r"$T_\mathrm{mod,ECEI}$ " + \
                    str(shotno) + ", $t = \SI{" + "{0:1.2f}".format(float(time)) + "}{\second}$")
        # plot_rays = True
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        try:
            ida_log = open(os.path.join(folder, "ida.log"))
            time = float(ida_log.readline().replace("\n", ""))
            ida_log.close()
        except Exception as e:
            print(e)
            print("No ida.log found time of ECE data might be incorrect")
        if(plot_rays and AUG):
            self.plot_EQH_vessel(shotno, time, self.axlist[0])
        else:
            self.plot_sep(shotno, time, self.axlist[0])
            self.plot_sep(shotno, time, self.axlist_2[0])
        diag_filename = os.path.join(folder, "ECRad_data", "diag.dat")
        m_abs_filename = os.path.join(folder, "ECRad_data", data_name)
        freq = np.loadtxt(os.path.join(folder, "ECRad_data", "f_ECE.dat"))
        ch_cnt = len(freq)
        Channel_list = np.zeros(ch_cnt)
        Channel_list[:] = 1
        if(simpl):
            model_data = np.loadtxt(os.path.join(folder, "ECRad_data", simplfile))
        else:
            model_data = np.loadtxt(m_abs_filename)
        if(rel_res):
            rhop = rel_rhop_res_all_ch(folder)
        else:
            rhop = model_data.T[0]
        rhop[Channel_list == 0] *= -1.0
        try:
            diag_data = np.genfromtxt(diag_filename, dtype='str')
        except IOError:
            diag_data = np.zeros(ch_cnt, dtype='|S3')
            diag_data[:] = Img_diag
            print("Warning no diag data found")
        if(len(diags.keys()) > 1):
            print("Can only plot data for one imaging array at a time - priorities are ECN > ECO > ECI")
        if("ECN" in  diags.keys()):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "ECN" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                y_err, ECI_data = get_data_calib(diag=diags["ECN"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=Calib_file.T[2] / 100.0 * Calib_file.T[1], \
                                                 ext_resonances=rhop[diag_data == "ECN"])
            except Exception as e:
                ECI_data = None
                print("Could not get ECN data")
                print("Reason", e)
                # return self.fig, self.fig_2
        elif("ECO" in diags.keys()):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "ECO" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                y_err, ECI_data = get_data_calib(diag=diags["ECO"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                             std_dev_calib=Calib_file.T[2] / 100.0 * Calib_file.T[1], \
                                             ext_resonances=rhop[diag_data == "ECO"])
            except Exception as e:
                ECI_data = None
                print("Could not get ECO data")
                print("Reason", e)
                # return self.fig, self.fig_2
        elif("ECI" in diags.keys()):
            try:
                Calib_file = np.loadtxt("/afs/ipp/home/s/sdenk/Documentation/Data/" + "ECI" + "_" + str(shotno) + "_calib_fact.txt", skiprows=2)
                y_err, ECI_data = get_data_calib(diag=diags["ECI"], shot=shotno, time=float(time), calib=Calib_file.T[1], \
                                                 std_dev_calib=Calib_file.T[2] / 100.0 * Calib_file.T[1], \
                                                 ext_resonances=rhop[diag_data == "ECI"])
            except Exception as e:
                ECI_data = None
                print("Could not get ECI data")
                print("Reason", e)
                # return self.fig, self.fig_2
        model_TRad = model_data.T[1][diag_data == Img_diag]
        if(rel_res):
            res = np.loadtxt(os.path.join(folder, "ECRad_data", "sres_rel.dat"))
        else:
            res = np.loadtxt(os.path.join(folder, "ECRad_data", "sres.dat"))
        if(Img_diag == "ECN"):
            shape = (20, 8)
        else:
            shape = (16, 8)
        R = res.T[1][diag_data == Img_diag]
        z = res.T[2][diag_data == Img_diag]
        rhop = res.T[3][diag_data == Img_diag]
        R_lin = np.linspace(np.min(R), np.max(R), shape[1])
        z_lin = np.linspace(np.min(z), np.max(z), shape[0])
        rhop_lin = np.linspace(np.min(z), np.max(z), shape[0])
        R_mesh, z_mesh = np.meshgrid(R, z)
        rhop_mesh, z_rhop_mesh = np.meshgrid(rhop, z)
        # weights = np.zeros(len(ECI_data[1]))
        # weights[:] = 1.e-5
        # y_err = np.array(y_err)
        # np.shape(y_err)
        # print(ECI_data[1])
        # weights[ECI_data[1] > y_err * 0.5] = 1.0 / (y_err[ECI_data[1] > y_err * 0.5] ** 2.0)
        # bad_channels = []
        # ECI_data[1][bad_channels] = -1
        # ECI_data[1] = ECI_data[1] / np.max(ECI_data[1])
        # colors = cm.jet(ECI_data[1])
        # colors[bad_channels] = np.array([1.0, 1.0, 1.0, 1.0])
        # ECI_data[1][ECI_data[1] < y_err * 0.5] = np.nan
        # print(ECI_data[1])

        # ECI_datam = np.ma.masked_invalid(ECI_data[1].reshape(shape).T)
        # weights[np.isnan(ECI_data[1])] = 0.0
        # ECI_data[1][np.isnan(ECI_data[1])] = 0.0
        # ECEISpline = SmoothBivariateSpline(R_mesh, z_mesh, ECI_data[1].reshape(8, 20), s=1.0)
        # ECEISpline = SmoothBivariateSpline(R, z, ECI_data[1], w=weights)
        R_2D = R.reshape(shape)
        z_2D = z.reshape(shape)
        rhop_2D = rhop.reshape(shape)
        if(ECI_data is not None):
            print("Making masked data")
            ECI_data[1] = ECI_data[1].reshape(shape)
            y_err = np.copy(y_err[0] + y_err[1]).reshape(shape)
            if(Img_diag == "ECN"):
                ECI_data[1] = ECI_data[1][2:20, :]
                y_err = y_err[2:20, :]
            good_data = np.zeros(len(R), dtype=np.bool).reshape(shape)
            bad_data = np.zeros(len(R), dtype=np.bool).reshape(shape)
            good_data[:] = True
            if(Img_diag == "ECN"):
                good_data = good_data[2:20, :]
                bad_data = bad_data[2:20, :]
            for iz in range(len(ECI_data[1])):
                for jR in range(len(ECI_data[1][iz])):
                    if(np.abs(ECI_data[1][iz][jR]) * 0.3 < np.abs(y_err[iz][jR])):
                        good_data[iz][jR] = False
                        bad_data[iz][jR] = True
                        print("Fixing channel", iz, jR)
                        if(iz == 0):
                            ECI_data[1][iz][jR] = ECI_data[1][iz + 1][jR]
                        else:  # (iz == len(ECI_data[1]) - 1):
                            ECI_data[1][iz][jR] = ECI_data[1][iz - 1][jR]
    #                    elif(ECI_data[1][iz + 1][jR] < y_err[iz][jR] * 0.5):
    #                        ECI_data[1][iz][jR] = ECI_data[1][iz - 1][jR]
    #                    else:
    #                        ECI_data[1][iz][jR] = ECI_data[1][iz + 1][jR] - (ECI_data[1][iz + 1][jR] - ECI_data[1][iz - 1][jR]) * \
    #                            (z_2D[iz][jR] - z_2D[iz - 1][jR]) / (z_2D[iz + 1][jR] - z_2D[iz - 1][jR])
                        print("Fixed channel TRad", ECI_data[1][iz][jR])
            max_data = np.max(ECI_data[1].flatten())
            min_data = np.min(ECI_data[1].flatten())
            np.savetxt("ECEI_data", ECI_data[1])
        else:
            min_data = np.Inf
            max_data = -np.inf
        if(Img_diag == "ECN"):
            R_2D_data = R_2D[2:20, :]
            z_2D_data = z_2D[2:20, :]
        else:
            R_2D_data = R_2D
            z_2D_data = z_2D
        model_TRad_2D = model_TRad.reshape(shape)
#        reduced_R = []
#        reduced_z = []
#        reduced_data = []
#        for i in range(len(y_err)):
#            R_temp = []
#            z_temp = []
#            temp_data = []
#            for j in range(len(y_err[i])):
#                if(ECI_data[1][i, j] > y_err[i, j] * 0.5):
#                    ECI_data[1][i, j] = 0.0
#                    R_temp.append(R[i, j])
#                    z_temp.append(z[i, j])
#                    temp_data.append(ECI_data[1][i, j])
#            if(len(R_temp) > 1):
#                reduced_R.append(np.array(R_temp))
#                reduced_z.append(np.array(z_temp))
#                reduced_data.append(np.array(temp_data))
#        reduced_R = np.array(reduced_R)
#        reduced_z = np.array(reduced_z)
#        reduced_data = np.array(reduced_data)
        print("shapes", R.shape, z.shape, model_TRad.shape)
#        print("shapes", reduced_R.shape, reduced_z.shape, reduced_data.shape)
        max_model = np.max(model_TRad_2D.flatten())
        min_model = np.min(model_TRad_2D.flatten())
        vmin = 0.0
        vmax = np.max([max_model, max_data])
        R, z = EQ_obj.get_axis(float(time))
        skip_channels = 1
        if(plot_rays):
            for i in range(len(diag_data)):
                if(diag_data[i] == Img_diag and skip_channels > 20):
                    svec, freq = read_svec_from_file(os.path.join(folder, 'ECRad_data'), i + 1)
                    self.plot_los(self.axlist[0], self.y_range_list[0], shotno, time, svec.T[1][R < svec.T[1]], \
                                  svec.T[2][R < svec.T[1]], None, None, no_names=True, eq_diag=eq_diag)
                if(diag_data[i] == Img_diag):
                    skip_channels += 1
        if(hasattr(cm, "plasma")):
            colormodel = self.axlist_2[0].pcolormesh(R_2D.T, z_2D.T, model_TRad_2D.T, vmin=vmin, vmax=vmax, cmap=cm.plasma)  # , shading='gouraud'
        else:
            colormodel = self.axlist_2[0].pcolormesh(R_2D.T, z_2D.T, model_TRad_2D.T, vmin=vmin, vmax=vmax, cmap=cm.jet)
        cb2 = self.fig_2.colorbar(colormodel)
        cb2.set_label(r"$T_\mathrm{rad}\,[\si{\kilo\electronvolt}]$")
        # print(np.shape(z_2D.T[1]), np.shape(ECEISpline(R, z, grid=False).reshape(shape).T[1]))
        # self.axlist_2[0].plot(z_2D.T[1], ECEISpline(R, z, grid=False).reshape(shape).T[1])
        # print(ECI_data[1])
        # ECI_data[1] = ECI_data[1] / np.max(ECI_data[1])
        print("Making masked cmap")
        print("Plotting")
        np.savetxt("ECEI_data_R", R_2D)
        np.savetxt("ECEI_data_z", z_2D)
        if(ECI_data is not None):
            if(hasattr(cm, "plasma")):
                colorplot = self.axlist[0].pcolormesh(R_2D_data.T, z_2D_data.T, ECI_data[1].T, vmin=vmin, vmax=vmax, cmap=cm.plasma)  # , shading='gouraud'
            else:
                colorplot = self.axlist[0].pcolormesh(R_2D_data.T, z_2D_data.T, ECI_data[1].T, vmin=vmin, vmax=vmax, cmap=cm.jet)  # , shading='gouraud'
            self.axlist[0].scatter(R_2D_data.flatten()[good_data.flatten()], z_2D_data.flatten()[good_data.flatten()])
            cb = self.fig.colorbar(colorplot)
            cb.set_label(r"$T_\mathrm{rad}\,[\si{\kilo\electronvolt}]$")
        self.axlist[0].set_xlabel("$R\, [\si{\meter}]$")
        self.axlist_2[0].set_xlabel("$R\, [\si{\meter}]$")
        self.axlist[0].set_ylabel("$z\, [\si{\meter}]$")
        self.axlist_2[0].set_ylabel("$z\, [\si{\meter}]$")
        # self.axlist_2[0].scatter(R_2D_data.flatten()[bad_data.flatten()], z_2D_data.flatten()[bad_data.flatten()], marker="^", color="red", linewidths=3)
        # m = cm.ScalarMappable(cmap=cm.jet)
        # m.set_array(val / np.max(val.flatten()))
        if(plot_rays):
            self.axlist[0].set_aspect("equal")
            self.axlist_2[0].set_aspect("equal")
        # plt.plot(mData['R'][idxROI],mData['z'][idxROI], 'wo', label = 'used
        # self.axlist_2[0].contourf(R_2D, z_2D, model_TRad_2D)
        return self.fig, self.fig_2

    def compare_driven_current(self, shot, time):
        tb = np.loadtxt("/ptmp1/work/sdenk/Relax/Maj/tools/j_tb")
        rho_LUKE = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/rho_luke")
        j_LUKE = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/j_luke")
        relax = np.loadtxt("/ptmp1/work/sdenk/Relax/Maj/tools/j_relax")
        self.setup_axes("single", "Driven current \#" + str(shot) + " $t = \SI{" + "{0:1.2f}".format(float(time)) + "}{\second}$" , "Cold vs. Warm refractive Index")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[tb.T[0], -tb.T[1]], \
                    name=r"TORBEAM", marker=":", color="black", \
                         y_range_in=self.y_range_list[0], ax_flag="j_eccd")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[relax.T[0], -relax.T[1]], \
                    name=r"RELAX", marker="--", color=(0.4, 0.4, 0.0), \
                         y_range_in=self.y_range_list[0], ax_flag="j_eccd")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[rho_LUKE, j_LUKE], \
                    name=r"LUKE", marker="-", color=(0.0, 0.4, 0.4), \
                         y_range_in=self.y_range_list[0], ax_flag="j_eccd")
        self.create_legends("single")
        return self.fig


    def B_plot(self, folder, shotno, time, comp_folder, ich, r_axis, N_filename, mode, OERT, comp=False, eq_diag=None):
        ECRad_data_folder = os.path.join(folder, "ECRad_data")
        f_R, omega_p, arr = get_omega_c_and_cutoff(ECRad_data_folder, ich, mode)
        svec, freq = read_svec_from_file(ECRad_data_folder, ich, mode)
        # svec, freq = read_svec_from_file(ECRad_data_folder, ich)
        # R,B = get_B(ECRad_data_folder,ich)
        self.setup_axes("double", "Cyclotron frequencies and measuring frequency", "Cold vs. Warm refractive Index")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], arr[1]], \
                    name=r"$f_\mathrm{c}$", marker="--", color=(0.4, 0.4, 0.0), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
#        if(comp and "OERT" in folder):
#            comp_folder = folder.replace("/OERT", "")
#            print("Comparison folder", comp_folder)
#            ECRad_data_comp_folder = os.path.join(comp_folder, "ECRad_data")
#            svec_IDA, freq = read_svec_from_file(ECRad_data_comp_folder, ich, mode)
#            r = ratio_B(svec.T[1], svec_IDA.T[1], svec.T[8], svec_IDA.T[8])
#            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[svec_IDA.T[1], r - 1.0], \
#                    name=r"$f_\mathrm{c,2,IDA}$", marker="--", color=(0.2, 0.2, 0.2), \
#                         y_range_in=self.y_range_list[0], ax_flag="rf")
#            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], arr[2]], \
#                    name=r"$f_\mathrm{c,2}$", marker="-", color=(0.2, 0.6, 0.0), \
#                         y_range_in=self.y_range_list[0], ax_flag="f", vline=r_axis)
#        else:
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], arr[2]], \
                name=r"$2 f_\mathrm{c}$", marker="-", color=(0.2, 0.6, 0.0), \
                     y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], omega_p / (2 * np.pi)], \
                    name=r"$f_\mathrm{p}$", marker=":", color=(0.2, 0.0, 0.6), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], arr[3]], \
                    name=r"$3 f_\mathrm{c}$", marker="--", color=(0.0, 0.4, 0.4), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], arr[4]], \
                    name=r"$f_\mathrm{ECE}$", marker="-", color=(0.0, 0.0, 0.0), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[arr[0], f_R], \
                    name=r"$f_\mathrm{R}$", marker=":", color=(0.6, 0.0, 0.3), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.model_color_index = 0
        for n in range(1, 4):
                success, s_res, R_res, z_res, rhop_res = find_cold_res(ECRad_data_folder, ich, mode, harmonic_number=n)
                if(success):
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[R_res, z_res], \
                                                                     name=r"", marker="o", color=self.model_colors[self.model_color_index], \
                                                                     y_range_in=self.y_range_list[1], ax_flag="Rz")
                    self.axlist[0].vlines(R_res, self.y_range_list[0][0], self.y_range_list[0][1], linestyle='dotted')
                    self.model_color_index += 1
        self.plot_los(self.axlist[1], self.y_range_list[1], shotno, time, svec.T[1][svec.T[3] != -1.0], svec.T[2][svec.T[3] != -1.0], None, None, eq_diag=eq_diag)
        if(comp):
            self.model_color_index = 0
            ECRad_data_comp_folder = os.path.join(comp_folder, "ECRad_data")
            svec_c, freq_c = read_svec_from_file(ECRad_data_comp_folder, ich, mode)
            for n in range(1, 4):
                success, s_res_c, R_res_c, z_res_c, rhop_res_c = find_cold_res(ECRad_data_comp_folder, ich, mode, harmonic_number=n)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[R_res_c, z_res_c], \
                                                                     name=r"", marker="+", color=self.model_colors[self.model_color_index], \
                                                                     y_range_in=self.y_range_list[1], ax_flag="Rz")
                self.model_color_index += 1
            self.plot_los(self.axlist[1], self.y_range_list[1], shotno, time, svec_c.T[1][svec_c.T[3] != -1.0], \
                           svec_c.T[2][svec_c.T[3] != -1.0], None, None, marker="--")
        plt.setp(self.axlist[0].get_xticklabels(), visible=False)
        self.axlist[0].set_xlabel("")
        if(N_filename is None):
            ds = np.concatenate([np.zeros(1), svec.T[0][1:len(svec.T[0])] - svec.T[0][0:len(svec.T[0]) - 1]])
            print("lenghts ", len(ds), len(svec.T[0]))
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[svec.T[0], ds], marker="--", color=(0.5, 0.0, 0.4), \
                                name=r"$N_\mathrm{OERT}$", ax_flag="N")
            self.create_legends("Te")
            if(not (comp and "OERT" in folder)):
                plt.setp(self.axlist[0].get_xticklabels(), visible=False)
                self.axlist[0].set_xlabel("")
            return self.fig, self.fig_2
        try:
            N_file = np.loadtxt(N_filename)
            if(OERT):
                if(len(N_file.T) == 5):
                    rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[1], f2=None)
                    for i in range(len(rhop_split)):
                        if(i == 0):
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="--", color=(0.5, 0.0, 0.4), \
                                    name=r"$N_\mathrm{OERT}$", ax_flag="N")
                        else:
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="--", color=(0.5, 0.0, 0.4), ax_flag="N")
                    rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[2], f2=None)
                    for i in range(len(rhop_split)):
                        if(i == 0):
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-", color="blue", \
                                    name=r"$N_\mathrm{cold}$", ax_flag="N")
                        else:
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-", color="blue", ax_flag="N")
                    rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[3], f2=None)
                    for i in range(len(rhop_split)):
                        if(i == 0):
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-.", color="red", \
                                    name=r"$N_\mathrm{cor}$", ax_flag="N")
                        else:
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-.", color="red", ax_flag="N")
                    rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[4], f2=None)
                    for i in range(len(rhop_split)):
                        if(i == 0):
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="--", color="black", \
                                    name=r"$N_\mathrm{warm \, dispersion}$", ax_flag="N")
                        else:
                            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                    self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="--", color="black", ax_flag="N")
                else:  # to support older files
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[N_file.T[0], N_file.T[1]], marker="-", color="blue", \
                                name=r"$N_\mathrm{cold}$", ax_flag="N")
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[N_file.T[0], N_file.T[2]], marker="-", color="red", \
                                name=r"$N_\mathrm{cor}$", ax_flag="N")
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[N_file.T[0], N_file.T[3]], marker="--", color="black", \
                                name=r"$N_\mathrm{warm \, dispersion}$", ax_flag="N")  #  / np.sin(svec.T[6][1:len(svec.T[6])])
            else:
                rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[1], f2=None)
                for i in range(len(rhop_split)):
                    if(i == 0):
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-", color="blue", \
                                name=r"$N_\mathrm{cold}$", ax_flag="N")
                    else:
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-", color="blue", ax_flag="N")
                rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[2], f2=None)
                for i in range(len(rhop_split)):
                    if(i == 0):
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-.", color="red", \
                                name=r"$N_\mathrm{cor}$", ax_flag="N")
                    else:
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="-.", color="red", ax_flag="N")
                rhop_split, N , R_axis = make_rhop_signed_axis(int(shotno), float(time), svec.T[1], svec.T[3], N_file.T[3], f2=None)
                for i in range(len(rhop_split)):
                    if(i == 0):
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="--", color="black", \
                                name=r"$N_\mathrm{warm \, dispersion}$", ax_flag="N")
                    else:
                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                                self.y_range_list_2[0] , data=[rhop_split[i], N[i]], marker="--", color="black", ax_flag="N")  #  / np.sin(svec.T[6][1:len(svec.T[6])])
        except IOError:
            print("No refractive index infomation found in: ", N_filename)
        self.create_legends("single")
        if(not (comp and "OERT" in folder)):
            plt.setp(self.axlist[0].get_xticklabels(), visible=False)
            self.axlist[0].set_xlabel("")
        self.axlist[1].set_aspect("equal")
        return self.fig, self.fig_2
        # try:
        #    if(dstf == "Th" or dstf == "Mx"):
        #        self.axlist[0],  self.y_range_list[0] = self.add_plot(self.axlist[0], filename = m_simpl_filename, \
        #          name = r"$T_\mathrm{rad}$ " + dist_simple + "-edf. using Euler", marker = "1",color=(0.2,0.3,0.5), y_range_in = self.y_range_list[0])
        # except:
        #    pass

    def diag_calib_summary(self, diag, file_list):
        self.setup_axes("single", "Calibration factors for diagn. " + diag, r"Rel. mean scatter for diagn. " + diag)
        shot_list = []
        data_list = []
        print("Found " + str(len(shot_list)) + " files")
        print("Shots:", shot_list)
        for file_nom in file_list:
            shot_list.append(file_nom.split("_", 2)[1])
            data_list.append(np.loadtxt(file_nom, skiprows=2))
        for i in range(len(shot_list)):
            # if(int(shot_list[i]) == 31873):
            #    data_list[i].T[1] *= 0.4222222222222222
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[data_list[i].T[0][data_list[i].T[2] < 10], data_list[i].T[1][data_list[i].T[2] < 10]], \
                y_error=data_list[i].T[1][data_list[i].T[2] < 10] * data_list[i].T[2][data_list[i].T[2] < 10] / 100.e3, \
                name=r"$c$ for \# " + str(shot_list[i]), marker="+", \
                     y_range_in=self.y_range_list[0], y_scale=1.e-3, ax_flag="Calib")
            # self.axlist_2[0],  self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
            #    data = [data_list[i].T[0][data_list[i].T[2] < 10],data_list[i].T[2][data_list[i].T[2] < 10]], \
            #    name = r"rel. mean scatter of c \# " + str(shot_list[i]), marker = "+",\
            #         y_range_in = self.y_range_list_2[0], ax_flag = "Calib_std_dev")
        self.create_legends("errorbar")
        return self.fig, self.fig_2

    def diag_calib_avg(self, diag, freq, calib, rel_dev, title):
        self.setup_axes("single", title, r"Rel. mean scatter for diagn. " + diag.name)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            data=[freq * 1.e-9, calib], \
            y_error=calib * rel_dev / 100.e0, \
            name=title, marker="+", \
                 y_range_in=self.y_range_list[0], ax_flag="Calib")
        self.create_legends("errorbar")
        return self.fig, self.fig_2

    def diag_calib_slice(self, diag, freq, calib, std_dev, title):
        self.setup_axes("single", title, r"Rel. mean scatter for diagn. " + diag.name)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            data=[freq * 1.e-9, calib], \
            y_error=std_dev, \
            name=title, marker="+", \
                 y_range_in=self.y_range_list[0], ax_flag="Calib")
        self.create_legends("errorbar")
        return self.fig, self.fig_2


    def calib_evolution(self, diag, ch, ECRad_result_list):
        self.title = False
        self.setup_axes("twinx", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
        # \"--\" $= T_\mathrm{e}$
        i = 0
        for result in ECRad_result_list:
            if(len(ECRad_result_list) - 1 == 0):
                color = (0.0, 0.0, 1.0)
            else:
                color = self.diag_cmap.to_rgba(float(i) / float(len(ECRad_result_list) - 1))
            Trad = []
            for itime in range(len(result.time)):
                if(result.masked_time_points[diag][itime]):
                    Trad.append(result.Trad[itime][result.Scenario.ray_launch[itime]["diag_name"]  == diag])
            Trad = np.array(Trad)
#            self.axlist[0].plot([], [], color=color, label=r"$T_\mathrm{rad, mod}$" + r" \# {0:d} ed {1:d} ch {2:d}".format(result.Config.shot, result.edition, ch + 1))
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[result.time[result.masked_time_points[diag]], np.abs(result.calib_mat[diag].T[ch])], \
                y_error=result.std_dev_mat[diag].T[ch], \
                name=r"$c$" + r" \# {0:d} ed {1:d} ch {2:d}".format(result.Scenario.shot, result.edition, ch + 1), marker="+", \
                     color=color, y_range_in=self.y_range_list[0], ax_flag="Calib_trace")
#            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                data=[result.Config.time, result.T.T[ch]], \
#                name=r"$T_\mathrm{e}$ \# {0:n} ch {1:n}".format(result.Config.shot, ch + 1), marker="+", \
#                     y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
#           Cheat in the ax in order to fix the legend ("legend a la Tomas")
#            if(result.Config.dstf == "TB" and result.Config.considered_modes > 1):
#                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                    data=[result.Config.time, result.Trad_comp.T[ch]], \
#                    marker="-", \
#                         y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
#                # name=r"$T_\mathrm{rad}$" + r" \# {0:n} ed {1:n} ch {2:n}".format(result.Config.shot, result.edition, ch + 1), \
#            else:
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                data=[result.time[result.masked_time_points[diag]], Trad.T[ch]], \
                color=color, marker="-", \
                     y_range_in=self.y_range_list[1], ax_flag="Trad_trace")
#            Te_spline = InterpolatedUnivariateSpline(result.Config.IDA_dict["rhop"], result.Config.IDA_dict["Te"])
#            rhop_Te = result.resonance["rhop_cold"].T[ch]
#            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                    data=[rhop_Te, Te_spline(rhop_Te)], \
#                    marker="--", \
#                         y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
            i += 1
#        self.axlist[1].set_ylim(0.0, self.y_range_list[1][1])
#        self.axlist[0].set_ylim(1.2 * self.y_range_list[0][0], 0.0)
        self.axlist[0].legend(loc="best")
        return self.fig

    def calib_evolution_Trad(self, diag, ch, ECRad_result_list, diag_data, std_dev_data, popt_list, pol_angle_list=None):
        self.title = False
        if(pol_angle_list is None):
            self.setup_axes("single", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
        else:
            self.setup_axes("twinx", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
        if(diag == "TDI"):
            if(len(ECRad_result_list[0].Trad.T[ECRad_result_list[0].diag == "ECN"]) > 0):
                actual_diag = "ECN"
            else:
                actual_diag = "ECO"
        else:
            actual_diag = diag
        # \"--\" $= T_\mathrm{e}$
        i = 0
        for result in ECRad_result_list:
            if(len(ECRad_result_list) - 1 == 0):
                color = (0.0, 0.0, 1.0)
            else:
                color = self.diag_cmap.to_rgba(float(i) / float(len(ECRad_result_list) - 1))
            Trad = []
            for itime in range(len(result.time)):
                if(result.masked_time_points[diag][itime]):
                    Trad.append(result.Trad[itime][result.Scenario.ray_launch[itime]["diag_name"]  == diag])
            Trad = np.array(Trad)
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    data=[Trad.T[ch], \
                          diag_data[i]], \
                    y_error=std_dev_data[i], name=r"Signal ed {0: d}  ch. no. {1: d}".format(result.edition, ch + 1), \
                    marker="+", color=color, \
                    y_range_in=self.y_range_list[0], ax_flag="Sig_vs_Trad_small", y_scale=1.e3)
            if(pol_angle_list is not None):
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    data=[Trad.T, pol_angle_list[i][result.masked_time_points[actual_diag]]], \
                    marker="*", color="red", name=r"$\theta_\mathrm{pol}$", \
                    y_range_in=self.y_range_list[1], ax_flag="Ang_vs_Trad")
            Trad_ax = np.linspace(0.0, np.max(Trad.T) * 1.2, 100)
            art_data = Trad_ax * popt_list[i][1] + popt_list[i][0]
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[Trad_ax, art_data], \
                y_range_in=self.y_range_list[0], marker="-", color=color , ax_flag="Sig_vs_Trad_small", y_scale=1.e3)
            # name=r"Linear fit for ch. no. {0: 2d}".format(ch + 1), \
            i += 1
#            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                data=[result.Config.time, result.T.T[ch]], \
#                name=r"$T_\mathrm{e}$ \# {0:n} ch {1:n}".format(result.Config.shot, ch + 1), marker="+", \
#                     y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
#            if(result.Config.dstf == "TB" and result.Config.considered_modes > 1):
#                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                    data=[result.Config.time, result.Trad_comp.T[ch]], \
#                    marker="-", \
#                         y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
#            else:
#                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                    data=[result.Config.time, result.Trad.T[ch]], \
#                    marker="-", \
#                         y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
#            Te_spline = InterpolatedUnivariateSpline(result.Config.IDA_dict["rhop"], result.Config.IDA_dict["Te"])
#            rhop_Te = result.resonance["rhop_cold"].T[ch]
#            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
#                    data=[rhop_Te, Te_spline(rhop_Te)], \
#                    marker="--", \
#                         y_range_in=self.y_range_list[1], ax_flag="Te_Te_Trad")
        if(pol_angle_list is not None):
            self.create_legends("errorbar_twinx")
        else:
            self.create_legends("errorbar")
        return self.fig

    def calib_vs_launch(self, diag, ch, ECRad_result_list, pol_ang_list):
        self.title = False
        self.setup_axes("single", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
        # \"--\" $= T_\mathrm{e}$
        i = 0
        for result in ECRad_result_list:
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[pol_ang_list[i][result.masked_time_points[diag]], result.calib_mat[diag][result.masked_time_points[diag]].T[ch]], \
                y_error=result.std_dev_mat[diag].T[ch], \
                name=r"\# {0:n} ch {1:n}".format(result.Config.shot, ch + 1), marker="+", \
                     y_range_in=self.y_range_list[0], ax_flag="calib_vs_launch")  # ed {1:n}
            i += 1
        self.create_legends("errorbar")
        return self.fig

    def Trad_vs_diag(self, diagnostic, ch, time_list, calib_diag_trace, Trad_trace, ECE_diag_trace):
        self.title = False
        self.setup_axes("single", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diagnostic, r"Rel. mean scatter for diagn. " + diagnostic)
        # \"--\" $= T_\mathrm{e}$
        for i in range(len(time_list)):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[time_list[i], calib_diag_trace[i][0]], \
                y_error=calib_diag_trace[i][1], \
                name=r"Calibrated signal for {0:s} channel {1:d}".format(diagnostic, ch + 1), marker="+", \
                     y_range_in=self.y_range_list[0], ax_flag="Trad_trace")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[time_list[i], Trad_trace[i]], \
                name=r"$T_\mathrm{rad, mod}$" + " for {0:s} channel {1:d}".format(diagnostic, ch + 1), marker="-", \
                     y_range_in=self.y_range_list[0], ax_flag="Trad_trace")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[time_list[i], ECE_diag_trace[i][0]], \
                y_error=ECE_diag_trace[i][1], \
                name=r"$T_\mathrm{rad, ECE}$", marker="*", \
                     y_range_in=self.y_range_list[0], ax_flag="Trad_trace")
        self.create_legends("errorbar")
        return self.fig

    def Intensity_plot(self, folder, folder_2, shotno, time, OERT, ch, ch2, dstf, rel_res, chHf, ch2Hf, R_ax, O_mode=False):
        ibb_flag = True
        # data = True
        scale_w_Trad = False
        x_axis = "rhop"  # "R"  #  "s"  #
        if(x_axis == "s"):
            x_coloumn = 0
        elif(x_axis == "rhop"):
            x_coloumn = 3
        else:
            x_coloumn = 1  # R
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        Trad = np.loadtxt(os.path.join(folder, "ECRad_data", data_name)).T[1]
        Trad_simpl = np.loadtxt(os.path.join(folder, "ECRad_data", simplfile)).T[1]
#        print(Trad)
        # folder = os.path.join(folder,shotno,time)
        # rhop, Trad = self.read_file(os.path.join(folder,"ECRad_data",data_name))
        sres = np.loadtxt(os.path.join(folder, "ECRad_data", "sres.dat"))
        rhop = sres.T[3]
        if(dstf == "ON"):
            Ich1 = "Ich" + "TB"
            Ich2 = "Ich" + "OM"
        else:
            Ich1 = "Ich" + dstf
            Ich2 = "Ich" + dstf
        if(chHf):
            ch_Hf_str = "HFS"
        else:
            ch_Hf_str = "LFS"
        if(ch2Hf):
            ch_Hf_str_2 = "HFS"
        else:
            ch_Hf_str_2 = "LFS"
        shotstr = "\#" + str(shotno) + " t = " + "{0:1.4f}".format(float(time)) + " s  "
        filelist = glob(os.path.join(folder, "ECRad_data", Ich1, "*.dat"))
        if(O_mode):
            svec_1, ece_freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch, mode='O')
        else:
            svec_1, ece_freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch)
        # omega_bar_1 = svec_1.T[7]/ ece_freq
        if(O_mode):
            print("Intensity plot for 'O' mode")
            filename_n = os.path.join(folder, "ECRad_data", Ich1, "IrhoOch" + "{0:0>3}.dat".format(ch))
            filename_n_2 = os.path.join(folder_2, "ECRad_data", Ich2, "IrhoOch" + "{0:0>3}.dat".format(ch2))
            filename_transparency = os.path.join(folder, "ECRad_data", Ich1, "TrhoOch" + "{0:0>3}.dat".format(ch))
            filename_transparency_2 = os.path.join(folder_2, "ECRad_data", Ich2, "TrhoOch" + "{0:0>3}.dat".format(ch2))
            filename_BPD = os.path.join(folder, "ECRad_data", Ich1, "BPDO" + "{0:0>3}.dat".format(ch))
            filename_BPD_2 = os.path.join(folder_2, "ECRad_data", Ich2, "BPDO" + "{0:0>3}.dat".format(ch2))
        else:
            print("Intensity plot for 'X' mode")
            filename_n = os.path.join(folder, "ECRad_data", Ich1, "Irhopch" + "{0:0>3}.dat".format(ch))
            filename_n_2 = os.path.join(folder_2, "ECRad_data", Ich2, "Irhopch" + "{0:0>3}.dat".format(ch2))
            filename_transparency = os.path.join(folder, "ECRad_data", Ich1, "Trhopch" + "{0:0>3}.dat".format(ch))
            filename_transparency_2 = os.path.join(folder_2, "ECRad_data", Ich2, "Trhopch" + "{0:0>3}.dat".format(ch2))
            filename_BPD = os.path.join(folder, "ECRad_data", Ich1, "BPDX" + "{0:0>3}.dat".format(ch))
            filename_BPD_2 = os.path.join(folder_2, "ECRad_data", Ich2, "BPDX" + "{0:0>3}.dat".format(ch2))
        filelist_2 = glob(os.path.join(folder_2, "ECRad_data", Ich2, "*.dat"))
        if(O_mode):
            svec_2, ece_freq_2 = read_svec_from_file(os.path.join(folder_2, "ECRad_data"), ch2, mode="O")
        else:
            svec_2, ece_freq_2 = read_svec_from_file(os.path.join(folder_2, "ECRad_data"), ch2)
        # filename_n_3 = os.path.join(folder,"ECRad_data",Ich,"Irhopchs" + "{0:0>3}.dat".format(ch))
        # filename_th = os.path.join(folder,"ECRad_data","IchTh","Irhopch" + "{0:0>3}.dat".format(ch))
        # filename_IDA_backw = os.path.join(folder,"ECRad_data",Ich,"IRad_backwIDAch{0:0>3}.dat".format(ch))
        # filename_IDA_backw_2 = os.path.join(folder,"ECRad_data",Ich,"IRad_backwIDAch{0:0>3}.dat".format(ch2))
        # filename_ECRad_backw = os.path.join(folder,"ECRad_data",Ich,"IRad_backwch{0:0>3}.dat".format(ch))
        # filename_ECRad_backw_2 = os.path.join(folder,"ECRad_data",Ich,"IRad_backwch{0:0>3}.dat".format(ch2))
        # print(rhop_final_arr)
        # filename_transparency_th = os.path.join(folder,"ECRad_data","IchTh","Trhopch" + "{0:0>3}.dat".format(ch))
        # filename_ne = os.path.join(folder,"ne_ida.res")
        filename_te = os.path.join(folder, "te_ida.res")
        BDP_plot = True
        try:
            if(not os.path.exists(filename_te) and OERT):
                Te_data = np.loadtxt(os.path.join(folder, "ECRad_data", "Te_file.dat"), skiprows=1)
                rhop_te = Te_data.T[0]
                te = Te_data.T[1] / 1.e3
            else:
                rhop_te, te = self.read_file(filename_te)
        except IOError as e:
            print(e)
            print("No Te file - turning off BDP plot")
            BDP_plot = False
        # rhop_ne, ne = self.read_file(filename_ne)
        # rhop_te, te = self.read_file(filename_te)
        # optical_thickness = 3.9*10**-19 * te * ne *1.e3
        # r_res_ch,z_res_ch = cold_res(os.path.join(folder,"ECRad_data"), ch)
        # r_res_ch_2,z_res_ch_2 = cold_res(os.path.join(folder,"ECRad_data"), ch2)
        if(rel_res):
            x_temp = np.loadtxt(os.path.join(folder, "ECRad_data", "sres_rel.dat"))
            x_temp_2 = np.loadtxt(os.path.join(folder_2, "ECRad_data", "sres_rel.dat"))
        else:
            x_temp = np.loadtxt(os.path.join(folder, "ECRad_data", "sres.dat"))
            x_temp_2 = np.loadtxt(os.path.join(folder_2, "ECRad_data", "sres.dat"))
        x_res_ch = x_temp.T[x_coloumn][ch - 1]
        x_res_ch_2 = x_temp_2.T[x_coloumn][ch2 - 1]
        if(x_axis == "rhop"):
            if(chHf):
                x_res_ch *= -1
            if(ch2Hf):
                x_res_ch_2 *= -1
        rhop_res = x_temp.T[3][ch - 1]
        rhop_res_2 = x_temp_2.T[3][ch2 - 1]
        Ibb = ece_freq ** 2 * svec_1.T[5] * cnst.e / cnst.c ** 2  # * 1.e3
        conv_fac = (ece_freq ** 2 * cnst.e / cnst.c ** 2)
        Ibb_2 = ece_freq_2 ** 2 * svec_2.T[5] * cnst.e / cnst.c ** 2  # 1.e3
        conv_fac_2 = (ece_freq_2 ** 2 * cnst.e / cnst.c ** 2)
        # omega_bar_2 = svec_2.T[7]/ ece_freq_2
        if(filename_n in filelist and filename_n_2 in filelist_2):
            ich_data = np.loadtxt(filename_n)
            T_data = np.loadtxt(filename_transparency)
            BPD_data = np.loadtxt(filename_BPD)
            x = svec_1.T[x_coloumn]
            if(x_coloumn == 3):
                x[R_ax > svec_1.T[1]] *= -1.0
            if(ch == ch2 and (filename_n_2 == filename_n)):
                self.setup_axes("Ich", shotstr + r"Radiation Transport cold resonance: $\rho_\mathrm{pol,res} = " + \
                    r"{0:1.2f}$".format(rhop[ch - 1]) + " on " + ch_Hf_str, r"BDO: $\rho_\mathrm{pol,res} = " + \
                    r"{0:1.2f}$".format(rhop[ch - 1]) + " on " + ch_Hf_str)
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[x[svec_1.T[3] != -1.0], ich_data.T[1][svec_1.T[3] != -1.0] * conv_fac],
                    name=r"$I_{\omega}" + dist + r"$", \
                    vline=x_res_ch, ax_flag="I" + "_" + x_axis, y_scale=1.e12, marker="-", color="black")  # vline = magn_axis
                if(ibb_flag):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=np.array([svec_1.T[x_coloumn][svec_1.T[3] != -1.0], Ibb[svec_1.T[3] != -1.0]]), \
                        name=r"$I_\mathrm{BB,\omega}$", \
                        coloumn=1, ax_flag="I", y_scale=1.e12, color="black", marker="--")  # vline = magn_axis,
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=[x[svec_1.T[3] != -1.0], ich_data.T[3][svec_1.T[3] != -1.0]], color="blue", \
                    name=r"$j_{\omega}" + dist + r"$", \
                    ax_flag="j" + "_" + x_axis, y_scale=1.e9)
                    # , vline = magn_axis
                # print(self.y_range_list[0])
                # self.axlist[3], self.y_range_list[1] = self.add_plot(self.axlist[3], \
                #    self.y_range_list[1] ,filename = filename_n_3, maxlines = 0, color = "blue", \
                #    name = r"$j_\mathrm{" + dist_simpl + "}$",  \
                #    coloumn = 2, vline = magn_axis,  ax_flag = "j",y_scale = 1.e9)


                # self.axlist[5], self.y_range_list[5] = self.add_plot(self.axlist[5], \
                #    self.y_range_list[5] ,filename = filename_transparency, maxlines = 0, color = "black", \
                #    name = r"$T_\mathrm{e}$",  \
                #    coloumn = 3,  ax_flag = "Te_I",y_scale = 1.e-3) #vline = magn_axis,
                self.axlist[4], self.y_range_list[4] = self.add_plot(self.axlist[4], \
                    self.y_range_list[4] , data=[x[svec_1.T[3] != -1.0], T_data.T[1][svec_1.T[3] != -1.0]], marker="--", color=(0.0, 0.0, 0.0), \
                    name=r"$T_{\omega}" + dist + r"$", vline=x_res_ch, ax_flag="T" + "_" + x_axis)
                self.axlist[5], self.y_range_list[5] = self.add_plot(self.axlist[5], \
                    self.y_range_list[5] , data=[x[svec_1.T[3] != -1.0], ich_data.T[5][svec_1.T[3] != -1.0]], marker="-", color=(0.6, 0.0, 0.0), \
                    name=r"$\alpha_{\omega}" + dist + r"$", \
                    ax_flag="alpha" + "_" + x_axis, y_scale=1.e-3)  # ,y_scale = 1.e-3
                self.axlist[8], self.y_range_list[8] = self.add_plot(self.axlist[8], \
                    self.y_range_list[8] , data=[x[svec_1.T[3] != -1.0], ich_data.T[7][svec_1.T[3] != -1.0]], color="blue", \
                    name=r"$j_{\omega}" + dist + r" - j" + dist_simpl + r"$", \
                    ax_flag="j" + "_" + x_axis, y_scale=1.e9)
                self.axlist[9], self.y_range_list[5] = self.add_plot(self.axlist[9], \
                    self.y_range_list[5] , data=[x[svec_1.T[3] != -1.0], ich_data.T[8][svec_1.T[3] != -1.0]], marker="-", color=(0.6, 0.0, 0.0), \
                    name=r"$\alpha_{\omega}" + dist + r" - \alpha" + dist_simpl + r"$", \
                    ax_flag="alpha" + "_" + x_axis, y_scale=1.e-3)
                # self.axlist[6], self.y_range_list[4] = self.add_plot(self.axlist[6], \
                #    self.y_range_list[4] ,filename = filename_n_3, maxlines = 0, color = "orange", \
                #    name = r"$\tau_\mathrm{Kirchoff}$", \
                #    coloumn = 4, vline = magn_axis,  ax_flag = "tau")
                # self.axlist[7], self.y_range_list[5] = self.add_plot(self.axlist[7], \
                #    self.y_range_list[5] ,filename = filename_n_3, maxlines = 0, color = "green",\
                #    name = r"$\alpha$", \
                #    coloumn = 3, vline = magn_axis, ax_flag = "alpha")

#                if(dstf in ["BM","MS","Re","DM","Pd","SH"])
                # if(dstf != "Th"):
                self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
                    self.y_range_list[0] , data=[x[svec_1.T[3] != -1.0], ich_data.T[2][svec_1.T[3] != -1.0] * conv_fac], \
                    name=r"$I_{\omega}" + dist_simpl + r"$", \
                    vline=x_res_ch, ax_flag="I" + "_" + x_axis, y_scale=1.e12, color="black")  # vline = magn_axis
#                else:
#                    self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
#                        self.y_range_list[0] ,filename = filename_n_3, maxlines = 0, marker = "-", color = "black",\
#                        name = r"$I_{\omega}" + dist_simpl + r"$",  \
#                        coloumn = 1, vline = x_res_ch, ax_flag = "I", y_scale = 1.e12)
                if(ibb_flag):
                    self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
                        self.y_range_list[0] , data=np.array([svec_1.T[x_coloumn][svec_1.T[3] != -1.0], Ibb[svec_1.T[3] != -1.0]]), maxlines=0,
                        name=r"$I_\mathrm{BB,\omega}$", \
                        ax_flag="I" + "_" + x_axis, y_scale=1.e12, color="black", marker="--")  # vline = magn_axis,
                self.axlist[3], self.y_range_list[1] = self.add_plot(self.axlist[3], \
                    self.y_range_list[1] , data=[x[svec_1.T[3] != -1.0], ich_data.T[4][svec_1.T[3] != -1.0]], color="blue", \
                    name=r"$j_{\omega}" + dist_simpl + r"$", \
                    ax_flag="j" + "_" + x_axis, y_scale=1.e9)
                self.axlist[6], self.y_range_list[4] = self.add_plot(self.axlist[6], \
                        self.y_range_list[4] , data=[x[svec_1.T[3] != -1.0], T_data.T[2][svec_1.T[3] != -1.0]], marker="--", color=(0.0, 0.0, 0.0), \
                        name=r"$T_{\omega}" + dist_simpl + r"$", \
                        vline=x_res_ch, ax_flag="T" + "_" + x_axis)
                self.axlist[7], self.y_range_list[5] = self.add_plot(self.axlist[7], \
                    self.y_range_list[5] , data=[x[svec_1.T[3] != -1.0], ich_data.T[6][svec_1.T[3] != -1.0]], marker="-", color=(0.6, 0.0, 0.0), \
                    name=r"$\alpha_{\omega}" + dist_simpl + r"$", \
                    ax_flag="alpha" + "_" + x_axis, y_scale=1.e-3)
#                else:
#                    self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
#                        self.y_range_list[0] ,filename = filename_n_3, maxlines = 0, color = "black",\
#                        name = r"$I_{\omega}" + dist_simpl + r"$",  \
#                        coloumn = 1, vline = r_res_ch, ax_flag = "I", y_scale = 1.e12)
#                    if(ibb_flag):
#                        self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
#                        self.y_range_list[0] ,data = np.array([svec_1.T[0],Ibb]), maxlines = 0,
#                        name = r"$I_\mathrm{BB,\omega}$", \
#                        coloumn = 1,  ax_flag = "I", y_scale = 1.e12, color = "black", marker = "--")
#                    self.axlist[3], self.y_range_list[1] = self.add_plot(self.axlist[3], \
#                        self.y_range_list[1] ,filename = filename_n, maxlines = 0, color = "blue", \
#                        name = r"$j_{\omega}" + dist_simpl + r"$",  \
#                        coloumn = 4,  ax_flag = "j",y_scale = 1.e9)
#                    self.axlist[6], self.y_range_list[4] = self.add_plot(self.axlist[6], \
#                    self.y_range_list[4] ,filename = filename_transparency, maxlines = 0, color = "red", \
#                    name = r"$T_{\omega}" + dist_simpl + r"$", \
#                    coloumn = 2, vline = r_res_ch, ax_flag = "T")
#                    self.axlist[7], self.y_range_list[5] = self.add_plot(self.axlist[7], \
#                        self.y_range_list[5] ,filename = filename_n, maxlines = 0, color = "green",\
#                        name = r"$\alpha_{\omega}" + dist_simpl + r"$", \
#                        coloumn = 6, ax_flag = "alpha") #, vline = magn_axis
                # vline = magn_axis
                # self.axlist[7], self.y_range_list[5] = self.add_plot(self.axlist[7], \
                #    self.y_range_list[5] ,filename = filename_transparency, maxlines = 0, color = "black",\
                #    name = r"$T_\mathrm{e}$", \
                #    coloumn = 3, ax_flag = "Te_I",y_scale = 1.e-3)#, vline = magn_axis
                # print(self.y_range_list[0])
                # , vline = magn_axis
                # , vline = magn_axis
#                Refractive Index
#                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                    self.y_range_list_2[0] ,filename = filename_n, maxlines = 0, color = "black",marker = "--", \
#                    name = r"$N_\mathrm{\omega,Bornatici}$", \
#                    coloumn = 9, vline = r_res_ch,  ax_flag = "N") #, vline = magn_axis
#                rhop_birth, D = weighted_emissivity(folder, int(shotno), float(time), ch, dstf, O_mode=O_mode)
                ax_flag = "j_weighted"
                if(self.title):
                    name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
                        r"{0:1.2f}".format(x_res_ch) + "$ on " + ch_Hf_str
                else:
                    name = r"$D_\omega" + dist_simpl + "$"
                if(x_axis == "rhop"):
                    x_BPD = BPD_data.T[0]
                    D = BPD_data.T[1]
                    D_2 = BPD_data.T[2]
                    if(scale_w_Trad):
                        D *= Trad[ch - 1] * 1.e3
                        D_2 *= Trad_simpl[ch - 1] * 1.e3
                        ax_flag += "_Trad"
                else:
                    ax_flag += "_" + x_axis
                    x_BPD = x[svec_1.T[3] != -1.0]
                    D = ich_data.T[3][svec_1.T[3] != -1.0] * T_data.T[1][svec_1.T[3] != -1.0] / (Trad[ch - 1] * conv_fac * 1.e3)
                    D_2 = ich_data.T[4][svec_1.T[3] != -1.0] * T_data.T[2][svec_1.T[3] != -1.0] / (Trad_simpl[ch - 1] * conv_fac_2 * 1.e3)
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[x_BPD, D], color=(0.6, 0.0, 0.0), marker="-", \
                        name=name, \
                        vline=x_res_ch, ax_flag=ax_flag)
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[x_BPD, D_2], color=(0.0, 0.0, 0.6), marker="-", \
                        name=name, \
                        vline=x_res_ch, ax_flag=ax_flag)
#                    else:
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] , data=[rhop_birth[i], D[i]], color=(0.6, 0.0, 0.0), marker="-", \
#                        ax_flag=ax_flag)
#                if(folder_2 != folder):
#                    rhop_birth, D = weighted_emissivity(folder_2, int(shotno), float(time), ch, dstf, O_mode=O_mode)
#                else:
#                    rhop_birth, D = weighted_emissivity(folder_2, int(shotno), float(time), ch, dstf, True, O_mode=O_mode)
#                if(scale_w_Trad):
#                    for i in range(len(D)):
#                        D[i] *= Trad_simpl[ch - 1]
#                fact = 1
#                if(ch2Hf):
#                    fact = -1
#                if(self.title):
#                    name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
#                        r"{0:1.2f}".format(rhop[ch - 1]) + "$ on " + ch_Hf_str_2
#                else:
#                    name = r"$D_\omega" + dist_simpl + "$"
#                for i in range(len(rhop_birth)):
#                    if(i == 0):
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                                self.y_range_list_2[0] , data=[rhop_birth[i], D[i] ], color=(0.0, 0.0, 0.6), marker="--", \
#                                name=name, \
#                                vline=rhop[ch - 1] * fact, ax_flag=ax_flag)
#                    else:
#                        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] , data=[rhop_birth[i], D[i]], color=(0.0, 0.0, 0.6), marker="--", \
#                        ax_flag=ax_flag)
                if(BDP_plot):
                    if(x_axis == "rhop"):
                        rhop_te_weighted = np.hstack([-rhop_te[::-1], rhop_te])
                        te_weighted = np.hstack([te[::-1], te])
                        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], \
                            self.y_range_list_2[1] , data=[rhop_te_weighted, te_weighted], color="black", \
                            name=r"$T_\mathrm{e}$", ax_flag="Te")
                    else:
                        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], \
                            self.y_range_list_2[1] , data=[x[svec_1.T[3] != -1.0], svec_1.T[5][svec_1.T[3] != -1.0]], color="black", \
                            name=r"$T_\mathrm{e}$", ax_flag="Te", y_scale=1.e-3)
                    # data = np.array([svec.T[3],omega_bar]), color = "blue", \
                    # name = r"$\frac{\omega_\mathrm{2X}}{\omega}$",\
                    # ax_flag = "omega")
                # self.axlist[9], self.y_range_list[9] = self.add_plot(self.axlist[9], \
                #    self.y_range_list[9] ,filename = filename_n_3, maxlines = 0, color = "black",marker = "--", \
                #    name = r"$N_\mathrm{Bornatici}$", \
                #    coloumn = 5, vline = magn_axis,  ax_flag = "N")
                # if(dstf != "O1"):
                #   self.axlist[9], self.y_range_list[9] = self.add_plot(self.axlist[9], \
                #       self.y_range_list[9] ,filename = filename_n_3, maxlines = 0, color = "blue", \
                #       name = r"$N_\mathrm{Sylvia}$", \
                #       coloumn = 6, vline = magn_axis,  ax_flag = "N")
                #   self.axlist[9], self.y_range_list[9] = self.add_plot(self.axlist[9], \
                #       self.y_range_list[9] ,filename = filename_n_3, maxlines = 0, marker="--",color = "red", \
                #       name = r"$N_\mathrm{Stroth}$", \
                #       coloumn = 7, vline = magn_axis,  ax_flag = "N")
                # self.axlist[9], self.y_range_list[9] = self.add_plot(self.axlist[9], \
                #    self.y_range_list[9] ,filename = filename_n_3, maxlines = 0, color = "green", \
                #    name = r"$N_\mathrm{Stix}$", \
                #    coloumn = 8, vline = magn_axis,  ax_flag = "N")
                self.create_legends("Ich")
            else:
                self.setup_axes("Ich_compare", shotstr + r"Radiation Transport", "Birthplace distribution of observed intensity")
                x = svec_1.T[x_coloumn][svec_1.T[3] != -1.0]
                x_2 = svec_2.T[x_coloumn][svec_2.T[3] != -1.0]
                if(x_coloumn == 3):
                    x[svec_1.T[1][svec_1.T[3] != -1.0] < R_ax] *= -1.0
                    x_2[svec_2.T[1][svec_2.T[3] != -1.0] < R_ax] *= -1.0
                ich_data = np.loadtxt(filename_n)
                T_data = np.loadtxt(filename_transparency)
                BPD_data = np.loadtxt(filename_BPD)
                ich_data_2 = np.loadtxt(filename_n_2)
                T_data_2 = np.loadtxt(filename_transparency_2)
                BPD_data_2 = np.loadtxt(filename_BPD_2)
                ax_flag = "j_weighted"
                if(x_axis == "rhop"):
                    x_BPD = BPD_data.T[0]
                    x_BPD_2 = BPD_data_2.T[0]
                    D = BPD_data.T[1]
                    D_2 = BPD_data_2.T[1]
                    print("Validating normalization")
                    D_spl = InterpolatedUnivariateSpline(x_BPD, D)
                    print("Norm of primary: {0:1.3e}".format(D_spl.integral(np.min(x_BPD), np.max(x_BPD))))
                    D_spl = InterpolatedUnivariateSpline(x_BPD_2, D_2)
                    print("Norm of secondary: {0:1.3e}".format(D_spl.integral(np.min(x_BPD_2), np.max(x_BPD_2))))
                else:
                    ax_flag += "_" + x_axis
                    x_BPD = x
                    x_BPD_2 = x_2
                    D = ich_data.T[3][svec_1.T[3] != -1.0] * T_data.T[1][svec_1.T[3] != -1.0] / (Trad[ch - 1] * conv_fac * 1.e3)
                    D_2 = ich_data_2.T[3][svec_2.T[3] != -1.0] * T_data_2.T[2][svec_2.T[3] != -1.0] / (Trad[ch2 - 1] * conv_fac_2 * 1.e3)
                D_max = np.max(D)
                D2_max = np.max(D_2)
                scale = 10 ** int(np.log10(max(D_max, D2_max) / min(D_max, D2_max)))
                if(D_max > D2_max):
                    scale_D1 = 1
                    scale_D2 = scale
                else:
                    scale_D1 = scale
                    scale_D2 = 1
#                    if(self.title):
                if(scale_D1 != 1):
                    name_1 = r"$D_\omega \times" + "{0:d}".format(scale_D1) + \
                            "$ , $\rho_\mathrm{pol,res,X2} = " + \
                             r"{0:1.2f}".format(x_res_ch) + r"$"
                else:
                    name_1 = r"$D_\omega$, $\rho_\mathrm{pol,res,X2} = " + \
                             r"{0:1.2f}".format(x_res_ch) + r"$"
                if(scale_D2 != 1):
                    name_2 = r"$D_\omega \times" + "{0:d}".format(scale_D2) + \
                             r"$ , $\rho_\mathrm{pol,res,X2} = " + \
                             r"{0:1.2f}".format(x_res_ch_2) + r"$"
                else:
                    name_2 = r"$D_\omega$, $\rho_\mathrm{pol,res,X2} = " + \
                       r"{0:1.2f}".format(x_res_ch_2) + r"$"
#                    else:
#                        name_1 = r"$D_\omega$"
#                        name_2 = r"$D_\omega$"
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                    self.y_range_list_2[0] , data=[x_BPD, D * scale_D1], color=(0.6, 0.0, 0.0), marker="-", \
                    name=name_1, ax_flag=ax_flag)
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                    self.y_range_list_2[0] , data=[x_BPD_2, D_2 * scale_D2], color=(0.0, 0.0, 0.6), marker="--", \
                    name=name_2, ax_flag=ax_flag)
                if(BDP_plot):
                    if(x_axis == "rhop"):
                        rhop_te_weighted = np.hstack([-rhop_te[::-1], rhop_te])
                        te_weighted = np.hstack([te[::-1], te])
                        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], \
                            self.y_range_list_2[1] , data=[rhop_te_weighted, te_weighted], color="black", \
                            name=r"$T_\mathrm{e}$", ax_flag="Te")
                    else:
                        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], \
                            self.y_range_list_2[1] , data=[x, svec_1.T[5][svec_1.T[3] != -1.0]], color="black", \
                            name=r"$T_\mathrm{e}$", ax_flag="Te", y_scale=1.e-3)
                # print(self.y_range_list[0])
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[x, ich_data.T[1][svec_1.T[3] != -1.0] * conv_fac], \
                    name=r"$I_{\omega}$", \
                    vline=x_res_ch, ax_flag="I" + "_" + x_axis, marker="-", color="black", y_scale=1.e12)  # , vline = magn_axis
                if(ibb_flag):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                             self.y_range_list[0] , data=np.array([svec_1.T[x_coloumn][svec_1.T[3] != -1.0], Ibb[svec_1.T[3] != -1.0]]),
                                             name=r"$I_\mathrm{BB,\omega}$", \
                                             coloumn=1, ax_flag="I" + "_" + x_axis, y_scale=1.e12, color="black", marker="--")  # vline = magn_axis,
                if(self.title):
                    self.axlist[0].set_title(r"Cold resonance position: " + "\n" + r"$\rho_\mathrm{pol,res} = " + \
                                             r"{0:1.2f}$ (left)".format(rhop[ch - 1]) + " on " + ch_Hf_str)
                # print(self.y_range_list[0])
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=[x, ich_data.T[3][svec_1.T[3] != -1.0]], color="blue", \
                    name=r"$j_{\omega}$", \
                    ax_flag="j" + "_" + x_axis, y_scale=1.e9)
                self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
                    self.y_range_list[0] , data=[x_2, ich_data_2.T[1][svec_2.T[3] != -1.0] * conv_fac_2], color="black", \
                    name=r"$I_{\omega}$", \
                    vline=x_res_ch_2, ax_flag="I" + "_" + x_axis, marker="-", y_scale=1.e12)
                if(self.title):
                    self.axlist[2].set_title(r"Cold resonance position:" + "\n" + r"$\rho_\mathrm{pol,res} = " + \
                                             r"{0:1.2f}$ (right)".format(rhop[ch2 - 1]) + " on " + ch_Hf_str_2)
                if(ibb_flag):
                    self.axlist[2], self.y_range_list[0] = self.add_plot(self.axlist[2], \
                         self.y_range_list[0] , data=np.array([svec_2.T[x_coloumn][svec_2.T[3] != -1.0], Ibb_2[svec_2.T[3] != -1.0]]),
                         name=r"$I_\mathrm{BB,\omega}$", \
                         coloumn=1, ax_flag="I" + "_" + x_axis, y_scale=1.e12, color="black", marker="--")
                self.axlist[3], self.y_range_list[1] = self.add_plot(self.axlist[3], \
                    self.y_range_list[1] , data=[x_2, ich_data_2.T[3][svec_2.T[3] != -1.0]], color="blue", \
                    name=r"$j_{\omega}$", \
                    ax_flag="j" + "_" + x_axis, y_scale=1.e9)
                self.axlist[4], self.y_range_list[4] = self.add_plot(self.axlist[4], \
                    self.y_range_list[4] , data=[x, T_data.T[1][svec_1.T[3] != -1.0]], marker="--", color=(0.0, 0.0, 0.0), \
                    name=r"$T_{\omega}$", vline=x_res_ch, ax_flag="T" + "_" + x_axis)
                # self.axlist[5], self.y_range_list[5] = self.add_plot(self.axlist[5], \
                #    self.y_range_list[5] ,filename = filename_transparency, maxlines = 0, color = "black", \
                #    marker = "--", name = r"$T_\mathrm{e}$",  \
                #    coloumn = 3, ax_flag = "Te_I",y_scale = 1.e-3)
                self.axlist[5], self.y_range_list[5] = self.add_plot(self.axlist[5], \
                    self.y_range_list[5] , data=[x, ich_data.T[5][svec_1.T[3] != -1.0]], color=(0.6, 0.0, 0.0), \
                    name=r"$\alpha_{\omega}$", ax_flag="alpha" + "_" + x_axis, y_scale=1.e-3)  # ,y_scale = 1.e-3
                # self.axlist[6], self.y_range_list[4] = self.add_plot(self.axlist[6], \
                #    self.y_range_list[4] ,filename = filename_n_3, maxlines = 0, color = "orange", \
                #    name = r"$\tau_\mathrm{Kirchoff}$", \
                #    coloumn = 4, vline = magn_axis,  ax_flag = "tau")
                self.axlist[6], self.y_range_list[4] = self.add_plot(self.axlist[6], \
                    self.y_range_list[4] , data=[x_2 , T_data_2.T[1][svec_2.T[3] != -1.0]], marker="--", color=(0.0, 0.0, 0.0), \
                    name=r"$T_{\omega}$", vline=x_res_ch_2 , ax_flag="T" + "_" + x_axis)
                # self.axlist[7], self.y_range_list[5] = self.add_plot(self.axlist[7], \
                #    self.y_range_list[5] ,filename = filename_transparency_2, maxlines = 0, color = "black",\
                #    marker = "--", name = r"$T_\mathrm{e}$", \
                #    coloumn = 3 , ax_flag = "Te_I",y_scale = 1.e-3)
                self.axlist[7], self.y_range_list[5] = self.add_plot(self.axlist[7], \
                    self.y_range_list[5] , data=[x_2 , ich_data_2.T[5][svec_2.T[3] != -1.0]], color=(0.6, 0.0, 0.0), \
                    name=r"$\alpha_{\omega}$", ax_flag="alpha" + "_" + x_axis, y_scale=1.e-3)  # , vline = magn_axis
                # #print(self.y_range_list[5])
                self.create_legends("Ich_compare")
            if(O_mode):
                mode = "O"
            else:
                mode = "X"
            success, s_res, R_res, z_res, rhop_res = find_cold_res(os.path.join(folder, "ECRad_data"), ch, mode, 2)
            if(success):
                if(R_res < R_ax):
                    rhop_res *= -1.0
                self.axlist_2[0].vlines(rhop_res, self.y_range_list_2[0][0], self.y_range_list_2[0][1], linestyle='dotted', colors=(0.6, 0.0, 0.0))
            if(ch != ch2):
                success, s_res, R_res, z_res, rhop_res = find_cold_res(os.path.join(folder, "ECRad_data"), ch2, mode, 2)
                if(success):
                    if(R_res < R_ax):
                        rhop_res *= -1.0
                    self.axlist_2[0].vlines(rhop_res, self.y_range_list_2[0][0], self.y_range_list_2[0][1], linestyle='dotted', colors=(0.0, 0.0, 0.6))
            plt.setp(self.axlist[0].get_xticklabels(), visible=False)
            plt.setp(self.axlist[1].get_xticklabels(), visible=False)
            plt.setp(self.axlist[2].get_xticklabels(), visible=False)
            plt.setp(self.axlist[3].get_xticklabels(), visible=False)
            self.axlist[0].set_xlabel("")
            self.axlist[1].set_xlabel("")
            self.axlist[2].set_xlabel("")
            self.axlist[3].set_xlabel("")
            return self.fig, self.fig_2, rhop[ch - 1], rhop[ch2 - 1]
        else:
            print("Not data for one of", filename_n, filename_n_2)
            return self.fig, self.fig_2, rhop[ch - 1], rhop[ch2 - 1]

    def plot_BPD(self, time, rhop_signed, D, D_comp, rhop_IDA, Te_IDA, dstf, rhop_cold, scale_w_Trad=False):
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        if(rhop_cold < 0.0):
            ch_Hf_str = "HFS"
        else:
            ch_Hf_str = "LFS"
        self.setup_axes("BPD", r"BDO: $\rho_\mathrm{pol,res} = " + \
            r"{0:1.2f}$".format(np.abs(rhop_cold)) + " on " + ch_Hf_str)
        ax_flag = "j_weighted"
        if(scale_w_Trad):
            ax_flag += "_Trad"
        if(self.title):
            name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
                r"{0:1.2f}".format(rhop_cold) + "$ on " + ch_Hf_str
        else:
            name = r"$D_\omega" + dist + "$"
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                self.y_range_list[0] , data=[rhop_signed, D ], color=(0.6, 0.0, 0.0), marker="-", \
                name=name, \
                vline=rhop_cold, ax_flag=ax_flag)

        if(self.title):
            name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
                r"{0:1.2f}".format(rhop_cold) + "$ on " + ch_Hf_str
        else:
            name = r"$D_\omega" + dist_simpl + "$"
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                self.y_range_list[0] , data=[rhop_signed, D_comp ], color=(0.0, 0.0, 0.6), marker="--", \
                name=name, \
                vline=rhop_cold, ax_flag=ax_flag)
        rhop_te_weighted = np.hstack([-rhop_IDA[::-1], rhop_IDA])
        te_weighted = np.hstack([Te_IDA[::-1], Te_IDA])
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
            self.y_range_list[1] , data=[rhop_te_weighted, te_weighted], color="black", \
            name=r"$T_\mathrm{e}$", ax_flag="Te")
        self.create_legends("BDP")
        return self.fig

    def ManyBirthplaces(self, folder, shotno, time, OERT, ch, dstf, rel_res, chHf, R_ax, O_mode=False):
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        Trad = np.loadtxt(os.path.join(folder, "ECRad_data", data_name)).T[1]
        Te_data = np.loadtxt(os.path.join(folder, "ECRad_data", "Te_file.dat"), skiprows=1)
        ax_coloumn = 3
        Ich = "Ich" + dstf
        if(chHf):
            ch_Hf_str = "HFS"
        else:
            ch_Hf_str = "LFS"
        shotstr = "\#" + str(shotno) + " t = " + "{0:1.4f}".format(float(time)) + " s  "
        if(O_mode):
            print("Birthplace plot for 'O' mode")
            svec_1, ece_freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch, mode="O")
            filename_n = os.path.join(folder, "ECRad_data", Ich, "IrhoOch" + "{0:0>3}.dat".format(ch))
            filename_transparency = os.path.join(folder, "ECRad_data", Ich, "TrhoOch" + "{0:0>3}.dat".format(ch))
            filename_BPD = os.path.join(folder, "ECRad_data", Ich, "BPDO" + "{0:0>3}.dat".format(ch))
            mode = "O"
        else:
            print("Birthplace plot for 'X' mode")
            svec_1, ece_freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch)
            filename_n = os.path.join(folder, "ECRad_data", Ich, "Irhopch" + "{0:0>3}.dat".format(ch))
            filename_transparency = os.path.join(folder, "ECRad_data", Ich, "Trhopch" + "{0:0>3}.dat".format(ch))
            filename_BPD = os.path.join(folder, "ECRad_data", Ich, "BPDX" + "{0:0>3}.dat".format(ch))
            mode = "X"
        if(rel_res):
            x_temp = np.loadtxt(os.path.join(folder, "ECRad_data", "sres_rel.dat"))
        else:
            x_temp = np.loadtxt(os.path.join(folder, "ECRad_data", "sres.dat"))
        scale_w_Trad = False
        ich_data = np.loadtxt(filename_n)
        T_data = np.loadtxt(filename_transparency)
        BPD_data = np.loadtxt(filename_BPD)
        if(ax_coloumn == 3):
            x_axis = svec_1.T[ax_coloumn]
            x_axis[svec_1.T[1] < R_ax] *= -1.e0
            sign_change_rhop = x_axis[0:len(x_axis) - 2] * x_axis[1:len(x_axis) - 1] < 0
            j = np.split(ich_data.T[3], np.where(np.concatenate([[False], sign_change_rhop]))[0])
            T = np.split(T_data.T[1], np.where(np.concatenate([[False], sign_change_rhop]))[0])
            x_axis = np.split(x_axis, np.where(np.concatenate([[False], sign_change_rhop]))[0])
        else:
            x_axis = [svec_1.T[0]]
            j = [ich_data.T[3]]
            T = [T_data.T[1]]
        fact = 1
        if(chHf):
            fact = -1
        ax_flag = "j_weighted"
        rhop = BPD_data.T[0]
        D = BPD_data.T[1]
        self.axlist[1].set_ylim(0.0, 1.2)
        if(scale_w_Trad):
            D *= Trad[ch - 1]
            ax_flag += "_Trad"
        if(self.title):
            name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
                r"{0:1.2f}".format(x_temp.T[3][ch - 1]) + "$ on " + ch_Hf_str
        else:
            name = r"$D_\omega$"
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[rhop, D], \
                        color=self.line_colors[self.line_color_index_2[0]], marker="-", \
                        name=name, vline=x_temp.T[ax_coloumn][ch - 1] * fact, ax_flag=ax_flag)
        self.line_color_index_2[0] += 1
        self.create_legends("BPD_twix")
        self.finishing_touches()
        success, s_res, R_res, z_res, rhop_res = find_cold_res(os.path.join(folder, "ECRad_data"), ch, mode, 3)
        if(success):
            if(R_res < R_ax):
                rhop_res *= -1.0
            self.axlist_2[0].vlines(rhop_res, self.y_range_list_2[0][0], self.y_range_list_2[0][1] * 10.e0, linestyle='dashed', colors="k")
        return self.fig, self.fig_2

    def Birthplace_plot(self, folder, shotno, time, OERT, ch, dstf, rel_res, chHf, R_ax, O_mode=False):
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        Trad = np.loadtxt(os.path.join(folder, "ECRad_data", data_name)).T[1]
        Te_data = np.loadtxt(os.path.join(folder, "ECRad_data", "Te_file.dat"), skiprows=1)
        ax_coloumn = 3
        Ich = "Ich" + dstf
        if(chHf):
            ch_Hf_str = "HFS"
        else:
            ch_Hf_str = "LFS"
        shotstr = "\#" + str(shotno) + " t = " + "{0:1.4f}".format(float(time)) + " s  "
        if(O_mode):
            print("Birthplace plot for 'O' mode")
            mode = "O"
            svec_1, ece_freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch, mode="O")
            filename_n = os.path.join(folder, "ECRad_data", Ich, "IrhoOch" + "{0:0>3}.dat".format(ch))
            filename_transparency = os.path.join(folder, "ECRad_data", Ich, "TrhoOch" + "{0:0>3}.dat".format(ch))
            filename_BPD = os.path.join(folder, "ECRad_data", Ich, "BPDO" + "{0:0>3}.dat".format(ch))
        else:
            print("Birthplace plot for 'X' mode")
            mode = "X"
            svec_1, ece_freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch)
            filename_n = os.path.join(folder, "ECRad_data", Ich, "Irhopch" + "{0:0>3}.dat".format(ch))
            filename_transparency = os.path.join(folder, "ECRad_data", Ich, "Trhopch" + "{0:0>3}.dat".format(ch))
            filename_BPD = os.path.join(folder, "ECRad_data", Ich, "BPDX" + "{0:0>3}.dat".format(ch))
        if(rel_res):
            x_temp = np.loadtxt(os.path.join(folder, "ECRad_data", "sres_rel.dat"))
        else:
            x_temp = np.loadtxt(os.path.join(folder, "ECRad_data", "sres.dat"))
        scale_w_Trad = False
        ich_data = np.loadtxt(filename_n)
        T_data = np.loadtxt(filename_transparency)
        BPD_data = np.loadtxt(filename_BPD)
        if(ax_coloumn == 3):
            x_axis = svec_1.T[ax_coloumn]
            x_axis[svec_1.T[1] < R_ax] *= -1.e0
            sign_change_rhop = x_axis[0:len(x_axis) - 2] * x_axis[1:len(x_axis) - 1] < 0
            j = np.split(ich_data.T[3], np.where(np.concatenate([[False], sign_change_rhop]))[0])
            T = np.split(T_data.T[1], np.where(np.concatenate([[False], sign_change_rhop]))[0])
            x_axis = np.split(x_axis, np.where(np.concatenate([[False], sign_change_rhop]))[0])
        else:
            x_axis = [svec_1.T[0]]
            j = [ich_data.T[3]]
            T = [T_data.T[1]]
        fact = 1
        if(chHf):
            fact = -1
        self.setup_axes("Ich_BD", shotstr + r"Radiation Transport cold resonance: $\rho_\mathrm{pol,res} = " + \
                    r"{0:1.2f}$".format(x_temp.T[3][ch - 1]), r"BDO: $\rho_\mathrm{pol,res} = " + \
                    r"{0:1.2f}$".format(x_temp.T[3][ch - 1]) + " on " + ch_Hf_str)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            self.y_range_list[0] , data=[x_axis[0], j[0]], marker="-", color="blue", \
            name=r"$j_{\omega}$", \
            ax_flag="j", y_scale=1.e9)
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
            self.y_range_list[1] , data=[x_axis[0], T[0]], marker="--", color=(0.0, 0.0, 0.0), \
            name=r"$T_{\omega}$", \
            vline=x_temp.T[ax_coloumn][ch - 1] * fact, ax_flag="T")
        for i in range(1, len(x_axis)):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                self.y_range_list[0] , data=[x_axis[i], j[i]], marker="-", color="blue", ax_flag="j", y_scale=1.e9)
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                self.y_range_list[1] , data=[x_axis[i], T[i]], marker="--", color=(0.0, 0.0, 0.0), \
                vline=x_temp.T[ax_coloumn][ch - 1] * fact, ax_flag="T")
        ax_flag = "j_weighted"
        D = BPD_data.T[1]
        rhop = BPD_data.T[0]
        self.axlist[1].set_ylim(0.0, 1.2)
        if(scale_w_Trad):
            D *= Trad[ch - 1]
            ax_flag += "_Trad"
        if(self.title):
            name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
                r"{0:1.2f}".format(x_temp.T[3][ch - 1]) + "$ on " + ch_Hf_str
        else:
            name = r"$D_\omega$"
        rhop_Te = np.concatenate([-1.0 * Te_data.T[0, ::-1], Te_data.T[0]])
        Te = np.concatenate([Te_data.T[1, ::-1], Te_data.T[1]])
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                        self.y_range_list[2] , data=[x_axis[0], j[0] * T[0] * cnst.c ** 2 / (ece_freq ** 2 * cnst.e)], \
                        color=(225.0 / 255.0, 0.0, 1.0), marker="-", \
                        name=name, vline=x_temp.T[ax_coloumn][ch - 1] * fact, ax_flag=ax_flag)
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[rhop, D], \
                        color=self.line_colors[self.line_color_index_2[0]], marker="-", \
                        name=name, vline=x_temp.T[ax_coloumn][ch - 1] * fact, ax_flag=ax_flag)
        for i in range(1, len(x_axis)):
            self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                        self.y_range_list[2] , data=[x_axis[i], j[i] * T[i] * cnst.c ** 2 / (ece_freq ** 2 * cnst.e)], \
                        color=(225.0 / 255.0, 0.0, 1.0), marker="-", \
                        ax_flag=ax_flag)
        self.line_color_index_2[0] += 1
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                self.y_range_list[3] , data=[rhop_Te, Te], marker=":", color="black", \
                name=r"$T_\mathrm{e}$", ax_flag="Te", y_scale=1.e-3)
        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], \
                self.y_range_list_2[1] , data=[rhop_Te, Te], marker=":", color="black", \
                name=r"$T_\mathrm{e}$", ax_flag="Te", y_scale=1.e-3)
        success, s_res, R_res, z_res, rhop_res = find_cold_res(os.path.join(folder, "ECRad_data"), ch, mode, 3)
        if(success):
            if(R_res < R_ax):
                rhop_res *= -1.0
            self.axlist_2[0].vlines(rhop_res, self.y_range_list_2[0][0], self.y_range_list_2[0][1] * 10.e0, linestyle='dashed', colors="k")
        self.create_legends("BPD_twix")
        self.finishing_touches()
        plt.setp(self.axlist[0].get_xticklabels(), visible=False)
        plt.setp(self.axlist[1].get_xticklabels(), visible=False)
        self.axlist[0].set_xlabel("")
        self.axlist[1].set_xlabel("")
        return self.fig, self.fig_2, x_temp.T[3][ch - 1]

    def PlotECESens(self, folder, shotno, time, dstf, Ch_list, O_mode):
        Te_data = np.loadtxt(os.path.join(folder, "ECRad_data", "Te_file.dat"), skiprows=1)
        Ich = "Ich" + dstf
        shotstr = "\#" + str(shotno) + " t = " + "{0:1.4f}".format(float(time)) + " s  "
        rhop_map = np.linspace(-1.2, 1.2, 1000)
        BPD = np.zeros(len(rhop_map))
        for ch in Ch_list:
            if(O_mode):
                print("Birthplace plot for 'O' mode")
                filename_BPD = os.path.join(folder, "ECRad_data", Ich, "BPDO" + "{0:0>3}.dat".format(ch))
            else:
                print("Birthplace plot for 'X' mode")
                filename_BPD = os.path.join(folder, "ECRad_data", Ich, "BPDX" + "{0:0>3}.dat".format(ch))
            BPD_data = np.loadtxt(filename_BPD)
            BPD_spl = InterpolatedUnivariateSpline(BPD_data.T[0], BPD_data.T[1], ext=1)
            BPD[:] += BPD_spl(rhop_map)
        self.setup_axes("twinx", shotstr + r"ECE sensitivity function", r"")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            self.y_range_list[0] , data=[rhop_map, BPD], marker="-", color="blue", \
            name=r"$j_{\omega}$", \
            ax_flag="j_weighted", y_scale=1.e9)
        rhop_Te = np.concatenate([-1.0 * Te_data.T[0, ::-1], Te_data.T[0]])
        Te = np.concatenate([Te_data.T[1, ::-1], Te_data.T[1]])
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                self.y_range_list[1] , data=[rhop_Te, Te], marker=":", color="black", \
                name=r"$T_\mathrm{e}$", ax_flag="Te", y_scale=1.e-3)
        self.create_legends("twinx")
        self.finishing_touches()
        return self.fig, self.fig_2

    def BenchmarkingPlot(self, folder, shotno, time, OERT, ch):
        # prof_file = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model/high_ne_abs_prof.dat")
        # prof_file = np.loadtxt("/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Model/low_ne_abs_prof.dat")
        filename_n_TB = os.path.join(folder, "ECRad_data", "IchTB", "Irhopch" + "{0:0>3}.dat".format(ch))
        TB_data = np.loadtxt(filename_n_TB)
        filename_n_Th = os.path.join(folder, "ECRad_data", "IchTh", "Irhopch" + "{0:0>3}.dat".format(ch))
        Th_data = np.loadtxt(filename_n_Th)
        filename_N = os.path.join(folder, "ECRad_data", "IchTB", "Nch{0:03n}.dat".format(ch))
        N_data = np.loadtxt(filename_N)
        # irhop_min = np.where(np.min(N_data.T[0]) == N_data.T[0])[0][0]
        irhop_min = len(N_data.T[0])
        svec, freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch)
        # TODO - this is broken for raytracing
        rhop_1 = svec.T[-1][0:irhop_min] / 2.e0 / freq  # -N_data.T[0][0:irhop_min]
        # rhop_2 = svec.T[-1][0:len(N_data.T[0])] / 2.e0 / freq #N_data.T[0][irhop_min:len(N_data.T[0])]
        N_cold_1 = N_data.T[1][0:irhop_min]
        N_cold_1[N_cold_1 > 1.0] = 0.e0
        # N_cold_2 = N_data.T[1][irhop_min:len(N_data.T[0])]
        N_cor_1 = N_data.T[2][0:irhop_min]
        N_cor_1[N_cor_1 > 1.0] = 0.e0
        # N_cor_2 = N_data.T[2][irhop_min:len(N_data.T[0])]
        N_gray_1 = N_data.T[3][0:irhop_min]
        N_gray_1[N_gray_1 > 1.0] = 0.e0
        # N_gray_2 = N_data.T[3][irhop_min:len(N_data.T[0])]
        self.setup_axes("double", r"Absorption coefficient", \
            title2="Refractive index")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[rhop_1, Th_data.T[5][0:irhop_min]], marker="-", color="blue", \
                    name=r"$\alpha_\omega$ Albajar model", ax_flag="alpha_rhop", y_scale=1.e-3)
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=[rhop_1, svec.T[5][0:irhop_min] / 1.e3], marker=":", color=(126.0 / 255, 126.0 / 255, 126.0 / 255), \
                    name=r"$T_\mathrm{e}$", ax_flag="Te")
#        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                        self.y_range_list[0] ,data = [rhop_2,Th_data.T[5][irhop_min:len(N_data.T[0])]], marker = "-", color = "blue",\
#                        ax_flag = "alpha_rhop")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=[rhop_1, Th_data.T[6][0:irhop_min]], marker="--", color=(0.e0, 126.0 / 255, 126.0 / 255), \
                        name=r"$\alpha_\omega$ imp. Hutchinson model", ax_flag="alpha_rhop", y_scale=1.e-3)
#        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                        self.y_range_list[0] ,data = [rhop_2,Th_data.T[6][irhop_min:len(N_data.T[0])]], marker = "--", color = (0.e0,126.0/255,126.0/255),\
#                        ax_flag = "alpha_rhop")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=[rhop_1, TB_data.T[6][0:irhop_min]], marker="-.", color="black", \
                        name=r"$\alpha_\omega$ warm dispersion", ax_flag="alpha_rhop", y_scale=1.e-3)
#        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                        self.y_range_list[0] ,data = [rhop_2,TB_data.T[6][irhop_min:len(N_data.T[0])]], marker = "-.", color = "black",\
#                        ax_flag = "alpha_rhop")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[rhop_1, N_cold_1], marker="-", color="blue", \
                        name=r"$N_\omega$ cold plasma", ax_flag="N_omega")
        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], \
                    self.y_range_list_2[1], data=[rhop_1, svec.T[5][0:irhop_min] / 1.e3], marker=":", color=(126.0 / 255, 126.0 / 255, 126.0 / 255), \
                    name=r"$T_\mathrm{e}$", ax_flag="Te")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] ,data = [rhop_2,N_cold_2], marker = "-", color = "blue",\
#                        ax_flag = "N")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[rhop_1, N_cor_1], marker="--", color=(0.e0, 126.0 / 255, 126.0 / 255), \
                        name=r"$N_\omega$ weakly rel. approx.", ax_flag="N_omega")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] ,data = [rhop_2,N_cor_2], marker = "--", color = (0.e0,126.0/255,126.0/255),\
#                        ax_flag = "N")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[rhop_1, N_gray_1], marker="-.", color="black", \
                        name=r"$N_\omega$ warm dispersion", ax_flag="N_omega")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] ,data = [rhop_2,N_gray_2], marker = "-.", color = "black",\
#                        ax_flag = "N")
        self.axlist_2[0].set_ylim([0.0, 1.1])
        self.create_legends("double")
        return self.fig, self.fig_2

    def dist_plot(self, Te, beta1=None, beta2=None):
        self.setup_axes("single", r"$T_\mathrm{e} =$" + r" {:2.1f}".format(Te / 1000.0) + \
            r" keV", title2="Spectral distribution for $T_\mathrm{e} =$" + r" {:2.1f}".format(Te / 1000.0) + \
            r" keV" + r", $\theta = \ang{80}$")
        in_04 = int((0.45) / (1.5) / 100.0)
        # print("beta",x[i],"u",x[i]/np.sqrt(1- x[i]**2))
        # f = np.zeros(len(x))
        # mu = cnst.physical_constants["electron mass"][0] * \
        # cnst.c**2/(abs(Te) * cnst.e)
        # f_int = np.zeros(len(x))
        # y  = np.arange(0.0, 4.0, 0.005)
        # for i in range(len(x)):
        #    f = Maxwell2D_beta(x,0,Te)
            # np.sqrt(mu/(2*np.pi))**3 * np.exp(- mu / 2.0 * ((x[i]/np.sqrt(1.0 + x[i]**2+ y**2))**2 + (y/np.sqrt(1.0 + x[i]**2 + y**2))**2))
            # f_int[i] = simps(f * 2.0 * np.pi * y/np.sqrt(1.0 + x[i]**2 + y**2),y/np.sqrt(1.0 + x[i]**2 + y**2))# * np.sqrt(1.0 + x[i]**2 + y**2)**5
        # print("Normalization constant Maxwell distribution: {0:1.5f}".format(simps(f_int, x/np.sqrt(1.0 + x**2))))
        if(beta1 == beta2 and beta2 is None):
            u = np.linspace(-2.e0, 2.e0, 401)
            f_M = Maxwell2D(u, 0.0, Te)
            # f = np.log10(make_f_beta(Te,0.0,x/np.sqrt(1.0 + x**2)))
            # f_approx = MJ_approx( Te,u)

            f_MJ = make_f(Te, 0.0, u)  # _1D
            i = np.argmin(np.abs(u[u >= 0] - 0.75)) + len(u[u < 0])
            print(u[u > 0])
            print(i)
            print("ratio", u[i], f_M[i], f_MJ[i], f_M[i] / f_MJ[i])
            # int1 = simps(f_M* u**2 * 4 * np.pi,u)
            # $int2 = simps(f_MJ*u**2 * 4 * np.pi,u)
#            E = (nap.sqrt(1 + u**2) - 1) * cnst.c**2 * cnst.m_e / cnst.e
            # print("int",int1,int2)
            u_par, ratio_mass, ratio_doppl, ratio_both, limit = cyc_momentum(np.cos(80.0 / 180.0 * np.pi))
#                print("Limit", limit)
#            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                    self.y_range_list[0] , data=[u_par, ratio_mass], color="red", \
#                    name=r"Rel. mass-increase", ax_flag="ratio_u_par")  # _E
#            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                    self.y_range_list[0] , data=[u_par, ratio_doppl], color="blue", \
#                    name=r"Doppler Effect", ax_flag="ratio_u_par")
#            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                    self.y_range_list[0] , data=[u_par, ratio_both], color="black", \
#                    name=r"Combined effect", ax_flag="ratio_u_par", vline=limit)
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=[u, np.log10(f_M)], color="red", \
                        name=r"Maxwellian", ax_flag="dist_par")  # _E
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[u, np.log10(f_MJ)], color=(0.e0, 0.e0, 0.5e0), marker="--", \
                    name=r"Maxwell-J\"uttner", ax_flag="dist_par")
#            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                    self.y_range_list[0] , data=[u, np.log10(f_approx)], color=(0.e0, 0.e0, 1.e0), marker="--", \
#                    name=r"$f_\mathrm{MJ}$ app. Maxwell-J\"uttner EEDF", ax_flag="dist_par")
#             self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                        self.y_range_list[0] ,data = [E/1.e3,np.log10(f_M)], color = "red",\
#                        name = r"$f_\mathrm{M}$ Maxwellian EMDF", ax_flag = "dist_E")
#             self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                    self.y_range_list[0] ,data = [E/1.e3,np.log10(f_MJ)], color = (0.e0,126.0/255,0.e0),marker = "-",\
#                    name = r"$f_\mathrm{MJ}$ Maxwell-J\"uttner EMDF",  ax_flag = "dist_E")
#             self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
#                    self.y_range_list[0] ,data = [E/1.e3,f_M/ f_MJ], color = "black",marker = "--",\
#                    name = r"$f_\mathrm{M}/ f_\mathrm{MJ}$",  ax_flag = "dist_E")
        # for i in range(len(x)):
        #    f = make_f(Te,y,x[i])
            # in_04 = int((0.45 + 0.99) / (0.99 /100)  )

        #    f_int[i] = simps(f*2.0*np.pi * y, y)
            # print("Normalization constant Maxwell-Juettner distribution: {0:1.5f}".format(simps(f_int, x)))
        else:
            x = np.linspace(-1.5, 1.5, 200)
            f_multi = multi_slope(x, 0.0, Te, beta2[1], (1.0 - beta2[0]) * Te)
            beta1[0] = Gauss_norm(0.0, [beta1[0], beta1[1], beta1[2]])
            f_BiJuett = BiJuettner(x, 0.0, Te, beta1)
            f_multi = np.log10(f_multi)
            f_BiJuett = np.log10(f_BiJuett)
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=[x, f_BiJuett], color="red", \
                        name=r"$f_\mathrm{BiMj}$", ax_flag="dist_par")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=[x, f_multi], color=(0.e0, 126.0 / 255, 0.e0), \
                        name=r"$f_\mathrm{TwoSlope}$", ax_flag="dist_par")
            f_MJ = make_f(Te, 0.0, x)  # _beta
            # if(beta1 == beta2 and beta2 is None):
            #    ratio = ratio / f_MJ[in_04]
            #    print("Ratio",ratio)
            #    print("M/ MJ beta =",x[in_04], (10.0**f_M[in_04] / f_MJ[in_04]) )
            f_MJ = np.log10(f_MJ)
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                        self.y_range_list[0] , data=[x, f_MJ], color="black", marker="--", \
                        name=r"$f_\mathrm{MJ}$", ax_flag="dist_par")
#        omega_mass, omega_doppler, omega_total, f_omega_mass, f_omega_doppler, f_total, limit = cycl_distribution(Te, np.cos(80.0 / 180.0 * np.pi))
#        omega_c = 140.0
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] , data=[omega_mass * omega_c, f_omega_mass / omega_c], marker="--", color="blue", \
#                        name=r"Rel. mass-shift", ax_flag="shift_freq")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] , data=[omega_doppler * omega_c, f_omega_doppler / omega_c], marker="-.", color=(0.e0, 126.0 / 255, 126.0 / 255), \
#                        name=r"Doppler-shift", ax_flag="shift_freq")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0] , data=[omega_total * omega_c, f_total / omega_c], color="black", \
#                        name=r"Combined", ax_flag="shift_freq", vline=omega_c)
#        E = np.linspace(np.sqrt(20.e-3), np.sqrt(300.e3), 10000)
#        E = E ** 2
#        u = np.sqrt((E / (cnst.m_e * cnst.c ** 2 / cnst.e) + 1.e0) ** 2 - 1.e0)
        beta = np.linspace(0.001, 1.0 - 1.e-9, 4000)
        beta = beta ** 2
        self.model_color_index_2 = 0
        self.line_marker_index = 0
        if(beta2 is not None):
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[beta, multi_slope_cyl_beta(beta, Te, beta2[1], (1.0 - beta2[0]) * Te)], \
                        marker=self.line_markers[self.line_marker_index], color=self.model_colors[self.model_color_index_2], \
                        name=r"Two Slope", ax_flag="dist_beta", log_flag=True)
            self.model_color_index_2 += 1
            self.line_marker_index += 1
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[beta, Juettner1D_beta(beta, Te)], marker=self.line_markers[self.line_marker_index], color=self.model_colors[self.model_color_index_2], \
                        name=r"Maxwell-J\"uttner", ax_flag="dist_beta", log_flag=True)
        self.model_color_index_2 += 1
        self.line_marker_index += 1
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                        self.y_range_list_2[0] , data=[beta, Maxwell1D_beta(beta, Te)], marker=self.line_markers[self.line_marker_index], color=self.model_colors[self.model_color_index_2], \
                        name=r"Maxwellian", ax_flag="dist_beta", log_flag=True)
        self.model_color_index_2 += 1
        self.line_marker_index += 1
        self.create_legends("single")
        beta_single = 1.0 / np.sqrt(2.0)
        print("Ratio for u = 1:", Maxwell1D_beta(beta_single, Te) / Juettner1D_beta(beta_single, Te))
#        tau = np.linspace(0.01, 3.0, 200)
#        kappa = 1.0e0 / (1.e0 - 0.92 * np.exp(-tau))
#        print("min kappa", "max kappa", np.min(kappa), np.max(kappa))
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
#                        self.y_range_list_2[0], data=[tau, kappa], color="black", \
#                        name=r"_", ax_flag="kappa")
#        #self.create_legends("single")
        return self.fig, self.fig_2

    def Calib_plot(self, fig, freq, scale_mat, relative_dev, shotno, timepoints, label, ECI=False):
        self.setup_axes("single", label)
        color_list = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, 0.0, 0.6), (0.3, 0.0, 0.3), ((0.0, 0.3, 0.3))]
        for it in range(len(timepoints)):
            if(ECI):
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[np.array(range(len(scale_mat[it])))[relative_dev <= 15.0], scale_mat[it][relative_dev <= 15.0]], marker="+",
                    color=color_list[it], \
                    name=r"$c$ for $t$ = " "{0:1.2f}".format(timepoints[it]) + " s", \
                    ax_flag="Calib_ch", y_scale=1.e3)
            else:
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[freq[relative_dev <= 15.0], scale_mat[it][relative_dev <= 15.0]], marker="+",
                    color=color_list[it], \
                    name=r"$c$ for $t$ = " "{0:1.2f}".format(timepoints[it]) + " s", \
                    ax_flag="Calib", y_scale=1.e-3, x_scale=1.e-9)
        if(ECI):
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                    self.y_range_list_2[0] , data=[np.array(range(len(relative_dev)))[relative_dev <= 15.0], relative_dev[relative_dev <= 15.0]], marker="+",
                    color=color_list[0], \
                    name=r"rel. mean scatter of $c$", \
                    ax_flag="Calib_dev_ch")
        else:
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], \
                    self.y_range_list_2[0] , data=[freq[relative_dev <= 15.0], relative_dev[relative_dev <= 15.0]], marker="+",
                    color=color_list[0], \
                    name=r"rel. mean scatter of $c$", \
                    ax_flag="Calib_dev", x_scale=1.e-9)
        self.create_legends("single")
        return fig

    def stacked_plot(self, fig, filename, shot, channelposition):
        self.setup_axes("single", r"Effects of a sawtooth crash on various ECE channels for discharge #" + shot)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename, maxlines=0, color="blue", \
                    name=r"Cold resonance position:" + "\n" + r"$\rho_\mathrm{pol,res} \approx " + \
                    r"{0:1.2f}$".format(channelposition[0]), \
                    coloumn=1, ax_flag="Te_trace")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename, maxlines=0, color="green", \
                    name=r"Cold resonance position: " + "\n" + r"$\rho_\mathrm{pol,res} \approx " + \
                    r"{0:1.2f}$".format(channelposition[1]), \
                    coloumn=2, ax_flag="Te_trace")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename, maxlines=0, color="magenta", \
                    name=r"Cold resonance position: " + "\n" + r"$\rho_\mathrm{pol,res} \approx " + \
                    r"{0:1.2f}$".format(channelposition[2]), \
                    coloumn=3, ax_flag="Te_trace")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename, maxlines=0, color="red", \
                    name=r"Cold resonance position: " + "\n" + r"$\rho_\mathrm{pol,res} \approx " + \
                    r"{0:1.2f}$".format(channelposition[3]), \
                    coloumn=4, ax_flag="Te_trace")
        self.create_legends("single")
        return fig

    def stacked_plot_2_0(self, fig, data1, data2, data3, data4, NPA=False):
        self.title = False
        if(NPA):
            self.setup_axes("stacked", 'bla')
        else:
            self.setup_axes("stacked_small", 'bla')
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=data1, maxlines=0, color="blue", \
                    name=r"core $n_\mathrm{e}$", \
                    ax_flag="ne_trace")
        self.axlist[0].set_xlabel("")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=data2[0], maxlines=0, color="blue", \
                    name=r"$P_\mathrm{ECRH}$", \
                    ax_flag="P_trace")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=data2[1], color="red", \
                    name=r"$P_\mathrm{NBI}$", \
                    ax_flag="P_trace")
        self.axlist[1].set_xlabel("")
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                    self.y_range_list[2] , data=data3[0], color="red", \
                    name=r"Core $T_\mathrm{e}$", \
                    ax_flag="Te_trace")
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                    self.y_range_list[2] , data=data3[1], \
                    name=r"$T_\mathrm{rad, ob.\, ECE}$ cross. calib.", \
                    ax_flag="Te_trace")
        if(NPA):
            self.axlist[2].set_xlabel("")
            self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                    self.y_range_list[3] , data=data4, \
                    name=r"NPA channel @ $\approx$\SI{30}{kev}", \
                    ax_flag="NPA_trace")
        self.create_legends("stacked")
        self.finishing_touches()
        return fig

    def stacked_plot_2_0_easy_trace(self, fig, data):
        self.title = False
        self.setup_axes("stacked_small", 'bla')
        xmin = np.Inf
        xmax = -np.inf
        for data_set in data:
            if(np.min(data_set["x"]) < xmin):
                xmin = np.min(data_set["x"])
            if(np.max(data_set["x"]) > xmax):
                xmax = np.max(data_set["x"])
            self.axlist[data_set["ax_index"]], self.y_range_list[data_set["ax_index"]] = self.add_plot(self.axlist[data_set["ax_index"]], \
                        self.y_range_list[data_set["ax_index"]] , data=[data_set["x"], data_set["y"]], \
                        name=data_set["name"], \
                        ax_flag=data_set["ax_flag"])
        self.axlist[0].set_xlabel("")
        self.axlist[1].set_xlabel("")
        self.axlist[0].set_xlim(xmin, xmax)
        self.create_legends("stacked")
        # self.finishing_touches()
        return fig

    def single_plot(self, fig, filename, shot, channelposition, ECRH_stop):
        self.setup_axes("single", r"Decay of the edge radiation temperature" + shot)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename, maxlines=0, color="black", \
                    name=r"ECE data: Cold resonance position at" + "\n" + r"$\rho_\mathrm{poloidal,res} = " + \
                    r"{0:1.2f}$".format(channelposition[0]), \
                    coloumn=1, vline=ECRH_stop, ax_flag="Te_trace")
        self.create_legends("single")
        return fig

    def plot_freq(self, fig, folder, shotno, time, diag, Te=False):
        shotstr = "\#" + str(shotno) + " t = " + "{0:1.4f}".format(float(time)) + " s  "
        data = get_ECE_spectrum(shotno, time, diag, Te)
        if(Te):
            self.setup_axes("Te_spec", r"ECE $T_\mathrm{rad}$ profile for " + shotstr)
            ax_flag = "Te_spec"
            label = r"$T_\mathrm{e}(\rho_\mathrm{pol})$"
            scale = 1.e0
        else:
            self.setup_axes("freq", r"ECE spectrum for " + shotstr)
            ax_flag = "freq"
            label = r"$I(\omega)$"
            scale = 1.e12
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0], data=[data[0][data[1] > data[2]] / 1.e9, data[1][data[1] > data[2]] * scale ], y_error=data[2][data[1] > data[2]] * scale  , marker=r"o", color=(0.0, 126.0 / 255, 0.0), \
                    name=label, \
                    ax_flag=ax_flag)
        # self.create_legends("single")
        return fig

    def plot_los(self, ax, y_range, shot, time, R_ray, z_ray, R_res, z_res, no_names=False, eq_diag=None, marker=r"-"):
        if(not no_names):
            base_str = "ECRad"
            if(AUG and eq_diag is not None):
                self.plot_EQ_vessel(shot, time, ax, eq_diag)
            ax, y_range = self.add_plot(ax, \
                          y_range, data=[R_ray , z_ray ], marker=marker, color=(0.0, 0.0, 0.0), \
                          name=r"Ray" + " " + base_str, \
                          ax_flag="Rz")
            if(R_res is not None and z_res is not None):
                ax, y_range = self.add_plot(ax, \
                          y_range, data=[np.array([R_res]) , np.array([z_res]) ], marker=r"*", color=(1.0, 0.0, 0.0), \
                          name=r"", \
                          ax_flag="Rz")
        else:
            ax, y_range = self.add_plot(ax, \
                          y_range, data=[R_ray, z_ray ], marker=marker, color=(0.0, 0.0, 0.0), \
                          ax_flag="Rz")
    @staticmethod
    def plot_EQH_vessel(shot, time, ax):
        if(not AUG):
            raise(ValueError("The routine plot_EQH_vessel is AUG specific and AUG = False"))
        EQ_obj = EQData(shot, EQ_exp='AUGD', EQ_diag='EQH', EQ_ed=0)
        EQ_Slice = EQ_obj.GetSlice(time)
        fconf.plt_vessel(ax)
        CS = ax.contour(EQ_Slice.R, EQ_Slice.z, EQ_Slice.rhop.T, \
            levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = ax.contour(EQ_Slice.R, EQ_Slice.z, EQ_Slice.rhop.T, \
                levels=[1.0], linewidths=3, colors="b", linestyles="-")
        plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS2, inline=1, fontsize=14)
        ax.set_xlim(np.min(EQ_Slice.R), np.max(EQ_Slice.R))
        ax.set_ylim(np.min(EQ_Slice.z), np.max(EQ_Slice.z))

    def plot_EQ_vessel(self, shot, time, ax, eq_diag):
        if(not AUG):
            raise(ValueError("The routine plot_EQH_vessel is AUG specific and AUG = False"))
        EQ_obj = EQData(shot, EQ_exp=eq_diag.exp, EQ_diag=eq_diag.diag, EQ_ed=eq_diag.ed)
        EQ_Slice = EQ_obj.GetSlice(time)
        fconf.plt_vessel(ax)
        CS = ax.contour(EQ_Slice.R, EQ_Slice.z, EQ_Slice.rhop.T, \
            levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = ax.contour(EQ_Slice.R, EQ_Slice.z, EQ_Slice.rhop.T, \
                levels=[1.0], linewidths=3, colors="b", linestyles="-")
        plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS2, inline=1, fontsize=14)
        ax.set_xlim(np.min(EQ_Slice.R), np.max(EQ_Slice.R))
        ax.set_ylim(np.min(EQ_Slice.z), np.max(EQ_Slice.z))

    def plot_sep(self, shot, time, ax):
        EQ_obj = EQData(shot, EQ_exp='AUGD', EQ_diag='EQH', EQ_ed=0)
        EQ_Slice = EQ_obj.GetSlice(time)
        CS = ax.contour(EQ_Slice.R, EQ_Slice.z, EQ_Slice.rhop.T, \
            levels=[1.0], linewidths=2, colors="k", linestyles="--")
        plt.clabel(CS, inline=1, fontsize=10)
        ax.set_xlim(np.min(EQ_Slice.R), np.max(EQ_Slice.R))
        ax.set_ylim(np.min(EQ_Slice.z), np.max(EQ_Slice.z))


    def show_ECE_Res(self, folder, alt_folder, shot, time, diag_ch, N_ch, ch, rel_res, mode, OERT, Comp, eq_diag=None):
        shotstr = "\#" + str(shot) + " t = " + "{0:1.3f}".format(time) + " s  "
#        eqi = eqi_map()
        ECRad_folder = os.path.join(folder, "ECRad_data")
#        coords, NPFM = eqi.norm_PFM(int(shot),float(time),np.linspace(0.1,1.2,12), diag = "EQI")
        alt_str = "straight"
        self.setup_axes("single", r"ECE cold res. and ray " + shotstr, "Toroidal angle")
        tb = False
        try:
            tb_data = np.loadtxt(os.path.join(ECRad_folder, "ray", "ray_ch_R{0:04d}tb.dat".format(int(ch))))
            tb = True
        except:
            print("No tb data at", os.path.join(ECRad_folder, "ray", "ray_ch_R{0:04d}tb.dat".format(int(ch))))
        # print(len(contours))
        channel_index = int(np.argmin(np.abs(int(ch) - diag_ch)))
        R_res = np.zeros(len(diag_ch))
        z_res = np.zeros(len(diag_ch))
        if(rel_res):
            R_res_warm = np.zeros(len(diag_ch))
            z_res_warm = np.zeros(len(diag_ch))
        try:
            for ich in range(len(diag_ch)):
                ich_diag = diag_ch[ich]
                if(rel_res):
                    sucess, dummy, R_res[ich], z_res[ich ], dummy_2 = find_cold_res(ECRad_folder, ich_diag)
                    R_res_warm[ich], z_res_warm[ich] = find_rel_res(ECRad_folder, ich_diag)
                else:
                    sucess, dummy, R_res[ich], z_res[ich], dummy_2 = find_cold_res(ECRad_folder, ich_diag)
        except IndexError:
            print("Index error occured when reading svec for channel " + str(ich_diag))
        svec, freq = read_svec_from_file(ECRad_folder, int(ch), mode=mode)
        base_str = ""
        if(Comp):
            R_res_2 = np.zeros(len(diag_ch))
            z_res_2 = np.zeros(len(diag_ch))
            ECRad_alt_folder = os.path.join(alt_folder, "ECRad_data")
#            if(OERT):
#                ECRad_alt_folder = ECRad_folder.replace("OERT/", "")
#                alt_str = "Org"
#                base_str = "OERT"
#            else:
#                ECRad_alt_folder = os.path.join(folder, "OERT", "ECRad_data")
#                alt_str = "OERT"
#                base_str = "Org"
            svec_2, freq = read_svec_from_file(ECRad_alt_folder, int(ch), mode)
            try:
                for ich in range(len(diag_ch)):
                    ich_diag = diag_ch[ich]
                    if(rel_res):
                        R_res_2[ich], z_res_2[ich] = find_rel_res(ECRad_alt_folder, ich_diag)
                    else:
                        sucess, s_res, R_res_2[ich], z_res_2[ich], dummy = find_cold_res(ECRad_alt_folder, ich_diag)
            except IndexError:
                print("Index error occured when reading svec for channel " + diag_ch)
        # Central los
        self.plot_los(self.axlist[0], self.y_range_list[0], shot, time, svec.T[1][svec.T[3] != -1], svec.T[2][svec.T[3] != -1], \
                      R_res[channel_index], z_res[channel_index], eq_diag=eq_diag)
        ray_list = glob(os.path.join(ECRad_folder, "ray", "Ray*ch{0:03d}_{1:s}.dat".format(int(ch), mode)))
        if(not Comp):
            for ray_file in ray_list:
                if(not "Ray001" in ray_file):
                    Raydata = np.loadtxt(ray_file)
                    R = np.sqrt(Raydata.T[1] ** 2 + Raydata.T[2] ** 2)
                    z = Raydata.T[3]
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                          self.y_range_list[0], data=[ R[R > 0], z[R > 0]], marker=r"--", color=(0.0, 0.0, 0.0), \
                          ax_flag="Rz")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[R_res , z_res ], marker=r"+", color=(1.0, 0.0, 0.0), \
                      name=r"Cold res. pos." + " " + base_str, \
                      ax_flag="Rz")
        if(rel_res):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[R_res_warm , z_res_warm ], marker=r"^", color=(0.0, 126.0 / 255.0, 0.0), \
                      name=r"Warm res. pos." + " " + base_str, \
                      ax_flag="Rz")
        if(tb):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[tb_data.T[0] / 100.0, tb_data.T[1] / 100.0], \
                    name=r"TORBEAM ray for channel " + str(ch) + " " + base_str, marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[svec.T[1], svec.T[6] / np.pi * 180.e0], \
                    name=r"$\theta$  for ch. " + str(ch) + " " + base_str, marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                         y_range_in=self.y_range_list_2[0], ax_flag="ang", vline=R_res[channel_index])
        if(Comp):
            # self.axlist[0],  self.y_range_list_2[0] = self.add_plot(self.axlist[0], \
            #        data= [np.array([R_res_2[int(ch) - 1]]), np.array([z_res_2[int(ch) - 1]])], \
            #        name = r"Ray for channel " + str(ch) + " " + alt_str, marker = "+",color=(0.5e0,0.e0,0.e0),\
            #             y_range_in = self.y_range_list[0],ax_flag = "Rz")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[svec_2.T[1][svec_2.T[3] != -1], svec_2.T[2][svec_2.T[3] != -1]], \
                    name=r"Ray for channel " + str(ch) + " " + alt_str, marker="--", color=(126.0 / 255, 0.e0, 0.e0), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    data=[np.array([R_res_2[int(ch) - 1]]), np.array([z_res_2[int(ch) - 1]])], \
                    marker="*", color=(1.0e0, 0.e0, 0.e0), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
            self.axlist_2[0], self.y_range_list[0] = self.add_plot(self.axlist_2[0], data=[svec_2.T[1], svec_2.T[6] / np.pi * 180.e0], \
                    name=r"$\theta$ for ch. " + str(ch) + " " + alt_str, marker="--", color=(126.0 / 255, 0.e0, 0.e0), \
                         y_range_in=self.y_range_list_2[0], ax_flag="ang", vline=R_res[int(ch) - 1])
        # print NPFM[0]
        # cmap = plt.cm.get_cmap("autumn")
        # CS = self.axlist[0].contour(coords[0][0],coords[0][1],NPFM[0],\
        #     levels = np.linspace(0.1,1.2,12), linewidths=1, colors = "k", linestyles = "--")
        self.axlist[0].set_aspect("equal")
        self.axlist[0].set_ylim(-0.5, 0.5)
        self.axlist[0].set_xlim(1.2, 2.2)
        # self.axlist_2[0].set_aspect("equal")
        self.create_legends("vessel")
        return self.fig, self.fig_2

    def QL_calc_check_plot(self, args):
        Callee = args[0]
        shot = args[1]
        time = args[2]
        linear_beam = args[3]
        quasi_linear_beam = args[4]
        dist_obj = args[5]
        f_ind = np.argmax(quasi_linear_beam.PW)
        Te_spl = InterpolatedUnivariateSpline(dist_obj.rhop_1D_profs, dist_obj.Te_init)
        ne_spl = InterpolatedUnivariateSpline(dist_obj.rhop_1D_profs, dist_obj.ne_init)
        if(np.isnan(dist_obj.rhop[f_ind]) and linear_beam is not None):
            rho_max_lin = linear_beam.rhop[np.argmax(linear_beam.PW)]
            f_ind = np.argmin(np.abs(rho_max_lin - quasi_linear_beam.rhop))
        Te_cur = Te_spl(quasi_linear_beam.rhop[f_ind])
        ne_cur = ne_spl(quasi_linear_beam.rhop[f_ind])
        print("Te_cur", Te_cur)
        self.setup_axes(r"QL_calc", "Power depostion and driven current \#" + str(shot) + \
                        r" $t = " + self.make_SI_string("{0:1.2f}".format(float(time)), r"\second") + r'$', \
                        r"Distribution function slice at " + r"$\rho_\mathrm{pol} =" + \
                        self.make_num_string("{0:1.2f}".format(dist_obj.rhop[f_ind])) + r"$")
        if(quasi_linear_beam.PW_tot < 1.e-1):
            scale = 1.e3
            ax_flag_pw = "P_ecrh_kw"
            ax_flag_j = "j_eccd_kA"
        else:
            scale = 1.0
            ax_flag_pw = "P_ecrh"
            ax_flag_j = "j_eccd"
        if(linear_beam is not None):
            renorm = quasi_linear_beam.PW_tot / linear_beam.PW_tot
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                             data=[linear_beam.rhop, \
                                                                   linear_beam.PW * renorm * scale], \
                                                             name=r"Gray", marker=":", color="black", \
                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag_pw)
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[linear_beam.rhop, \
                                                                   linear_beam.j], \
                    name=r"Gray", marker=":", color="black", \
                         y_range_in=self.y_range_list[1], ax_flag=ax_flag_j)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[quasi_linear_beam.rhop, \
                                                                                   quasi_linear_beam.PW * scale], \
                                                             name=r"RELAX", marker="+", color=(0.4, 0.4, 0.0), \
                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag_pw)
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[quasi_linear_beam.rhop, \
                                                                                   quasi_linear_beam.j * scale], \
                                                             name=r"quasi-linear (RELAX)", marker="+", color=(0.4, 0.4, 0.0), \
                                                             y_range_in=self.y_range_list[1], ax_flag=ax_flag_j)
        if(linear_beam is not None):
            pw_torbeams = []
            j_torbeams = []
            rho_torbeams = []
            PW_tot_tb = 0.0
            path = "/pfs/work/g2sdenk/TB/{0:5d}_{1:1.3f}_rays/".format(shot, time)
            # print(path)
            if(os.path.isdir(path)):
                i = 1
                filename = os.path.join(path, "pw_j_beam_{0:1d}.dat".format(i))
                while(os.path.isfile(filename)):
                    tb_file = np.loadtxt(filename)
                    rho_torbeams.append(tb_file.T[0])
                    pw_torbeams.append(tb_file.T[1])
                    j_torbeams.append(tb_file.T[2])
                    inbeam_file = open(os.path.join(path, "inbeam{0:d}.dat".format(i)))
                    inbeam_lines = inbeam_file.readlines()
                    inbeam_file.close()
                    for line in inbeam_lines:
                        if("xpw0" in line):
                            PW_tot_tb += float(line.rsplit("=", 1)[1].replace(r"\n", "").replace(r",", ""))
                    i += 1
                    filename = os.path.join(path, "pw_j_beam_{0:1d}.dat".format(i))
            Pw_beams = np.zeros(200)
            j_beams = np.zeros(200)
            rhop_tb_profs = np.linspace(0.0, 1.0, 200)
            for ibeam in range(len(linear_beam.PW_beam)):
                if(ibeam < len(rho_torbeams)):
                    if(np.sum(pw_torbeams[ibeam]) > 1.e-6 and np.all(np.isfinite(pw_torbeams[ibeam]))):
                        pw_spl = InterpolatedUnivariateSpline(rho_torbeams[ibeam], pw_torbeams[ibeam])
                        j_spl = InterpolatedUnivariateSpline(rho_torbeams[ibeam], j_torbeams[ibeam])
                        Pw_beams[:] += pw_spl(rhop_tb_profs)
                        j_beams[:] += j_spl(rhop_tb_profs)
                    else:
                        print("TORBEAM ray skipped because of NAN or less than 1 W total power deposited")
                if(linear_beam.rays is not None):
                    for iray in range(len(linear_beam.rays[ibeam])):
                        rhop_spline = InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["rhop"] - dist_obj.rhop[f_ind])
                        roots = rhop_spline.roots()
                        if(np.isscalar(roots)):
                            s_res_LFS = roots
                        elif(len(roots) == 0):
                            continue
                        else:
                            s_res_LFS = roots[0]
                        omega_c = InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["omega_c"])(s_res_LFS)
                        N_par = InterpolatedUnivariateSpline(linear_beam.rays[ibeam][iray]["s"], linear_beam.rays[ibeam][iray]["Npar"])(s_res_LFS)
                        omega = 2.0 * np.pi * 140.e9
                        omega_bar = omega / omega_c
                        if(N_par ** 2 >= 1.0):
                            continue
                        m_0 = np.sqrt(1.e0 - N_par ** 2) * omega_bar
                        t = np.linspace(-1.0, 1.0, 60)
                        for m in [2, 3]:
                            u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                                               np.sqrt((float(m) / m_0) ** 2 - 1.e0) * t)
                            u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - t ** 2)
                            u_perp_sq[u_perp_sq < 0] += 1.e-7
                            if(np.all(u_perp_sq >= 0)):
                                u_perp = np.sqrt(u_perp_sq)
                                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[u_perp, \
                                                                                                                 u_par], \
                                                                                         marker="--", y_range_in=self.y_range_list_2[0])
            if(np.sum(Pw_beams) > 1.e-6):
                renorm = quasi_linear_beam.PW_tot / PW_tot_tb
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                             data=[rhop_tb_profs , Pw_beams * renorm * scale], \
                                                             name=r"TORBEAM", marker="--", \
                                                             y_range_in=self.y_range_list[0], ax_flag=ax_flag_pw)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[rhop_tb_profs , j_beams * renorm * scale], \
                    name=r"TORBEAM".format(ibeam + 1), marker="--", \
                         y_range_in=self.y_range_list[1], ax_flag=ax_flag_j)
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], data=[dist_obj.rhop_1D_profs, \
                                                                                   dist_obj.Te_init], \
                    name=r"initial $T_\mathrm{e}$", marker="-", color="black", \
                         y_range_in=self.y_range_list[2], y_scale=1.e-3, ax_flag="Te")
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], data=[dist_obj.rhop, \
                                                                                   dist_obj.Te], \
                    name=r"$T_\mathrm{e}\mathrm{\, from\, distribution}$", marker="--", color="blue", \
                         y_range_in=self.y_range_list[2], y_scale=1.e-3, ax_flag="Te")
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], data=[dist_obj.rhop_1D_profs, \
                                                                                   dist_obj.ne_init], \
                    name=r"initiall $n_\mathrm{e}$", marker="-", color="green", \
                         y_range_in=self.y_range_list[3], y_scale=1.e-20, ax_flag="ne")
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], data=[dist_obj.rhop, \
                                                                                   dist_obj.ne], \
                    name=r"$n_\mathrm{e}\mathrm{\, from\, distribution}$", marker="--", color=[0.5, 0.0, 0.5], \
                         y_range_in=self.y_range_list[3], y_scale=1.e-20, ax_flag="ne")
        plt.setp(self.axlist[0].get_xticklabels(), visible=False)
        self.axlist[0].set_xlabel("")
        plt.setp(self.axlist[1].get_xticklabels(), visible=False)
        self.axlist[1].set_xlabel("")
        self.axlist[0].set_xlim(0.0, 1.0)
        levels = np.linspace(-13, 5, 100)
        try:
            cmap = plt.cm.get_cmap("plasma")
        except ValueError:
            cmap = plt.cm.get_cmap("jet")
        cont1 = self.axlist_2[0].contourf(dist_obj.uxx, dist_obj.ull, dist_obj.f_cycl_log10[f_ind], levels=levels, cmap=cmap)  # ,norm = LogNorm()
        cont2 = self.axlist_2[0].contour(dist_obj.uxx, dist_obj.ull, dist_obj.f_cycl_log10[f_ind], levels=levels, colors='k',
                            hold='on', alpha=0.25, linewidths=1)
        self.axlist_2[0].set_ylabel(r"$u_{\Vert}$")
        self.axlist_2[0].set_xlabel(r"$u_{\perp}$")
        for c in cont2.collections:
            c.set_linestyle('solid')
        cb = self.fig_2.colorbar(cont1, ax=self.axlist_2[0], ticks=[-10, -5, 0, 5])  # ticks = levels[::6] #,
        cb.set_label(r"$\log_\mathrm{10}\left(f\right)$")  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
        # cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
        cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([1.0, 5.0, 10.0])))
        cb.ax.minorticks_on()
        steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
        steps_y = steps
        self.axlist_2[0].get_xaxis().set_major_locator(MaxNLocator(nbins=4, steps=steps, prune='lower'))
        self.axlist_2[0].get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 4.0))
        self.axlist_2[0].get_yaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps_y))
        self.axlist_2[0].get_yaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps_y / 4.0))
        self.axlist_2[0].set_ylim(dist_obj.ull[0], dist_obj.ull[-1])
        f_spl = RectBivariateSpline(dist_obj.ull, dist_obj.uxx, dist_obj.f_cycl_log10[f_ind])
        ull = np.linspace(-np.sqrt(dist_obj.uxx[-1]), np.sqrt(dist_obj.uxx[-1]), 200)
        uxx = np.abs(ull)
        f = np.zeros(len(ull))
        for j in range(len(ull)):
            f[j] = f_spl(ull[j], uxx[j])
#            print("u < 0", ull[j], uxx[j], f[j])
        f_therm = np.log10(Juettner2D(ull, uxx, Te_cur))
        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], data=[ull, \
                                                                   f], \
                    name=r"Non-thermal $\vert u_\Vert \vert=  u_\perp $", marker="-", \
                         y_range_in=self.y_range_list_2[1], ax_flag="f_cut")
        self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], data=[ull, \
                                                                   f_therm], \
                    name=r"Thermal $\vert u_\Vert \vert=  u_\perp $", marker="--", \
                         y_range_in=self.y_range_list_2[1], ax_flag="f_cut")
        self.axlist_2[1].set_ylabel("")
        self.create_legends("only_axis_1")
        self.axlist_2[0].set_aspect("equal")
        evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
        wx.PostEvent(Callee, evt_out)

    def Power_depo_plot(self, args):
        Callee = args[0]
        shot = args[1]
        time = args[2]
        linear_beam = args[3]

        if(plot_mode == "Software"):
            self.setup_axes(r"stacked_small", "Power depostion and driven current #" + str(shot) + \
                            r" $t = " + "{0:1.2f}".format(float(time)) + " s$")
        else:
            self.setup_axes(r"stacked_small", "Power depostion and driven current \#" + str(shot) + \
                            r" $t = \SI{" + "{0:1.2f}".format(float(time)) + "}{\second}$")
        pw_torbeams = []
        j_torbeams = []
        rho_torbeams = []
        path = "/marconi_work/eufus_gw/work/g2sdenk/TB/{0:5d}_{1:1.3f}_rays/".format(shot, time)
        if(os.path.isdir(path)):
            i = 1
            filename = os.path.join(path, "pw_j_beam_{0:1n}.dat".format(i))
            while(os.path.isfile(filename)):
                tb_file = np.loadtxt(filename)
                rho_torbeams.append(tb_file.T[0])
                pw_torbeams.append(tb_file.T[1])
                j_torbeams.append(tb_file.T[2])
                i += 1
                filename = os.path.join(path, "pw_j_beam_{0:1n}.dat".format(i))
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                             data=[linear_beam.rhop, \
                                                                   linear_beam.PW], \
                                                             name=r"total power (Gray)", marker="-", color="black", \
                                                             y_range_in=self.y_range_list[0], ax_flag="P_ecrh")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[linear_beam.rhop, \
                                                                   linear_beam.j], \
                    name=r"total driven current (Gray)", marker="-", color="black", \
                         y_range_in=self.y_range_list[1], ax_flag="j_eccd")
        for ibeam in range(len(linear_beam.PW_beam)):
            if(ibeam < len(rho_torbeams)):
                if(np.any(pw_torbeams[ibeam] > 0.2)):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                 data=[rho_torbeams[ibeam], \
                                                                       pw_torbeams[ibeam]], \
                                                                 name=r"power beam {0:d} (TORBEAM)".format(ibeam + 1), marker="--", \
                                                                 y_range_in=self.y_range_list[0], ax_flag="P_ecrh")
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[rho_torbeams[ibeam], \
                                                                       j_torbeams[ibeam]], \
                        name=r"driven current beam {0:d} (TORBEAM)".format(ibeam + 1), marker="--", \
                             y_range_in=self.y_range_list[1], ax_flag="j_eccd")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                             data=[linear_beam.rhop, \
                                                                   linear_beam.PW_beam[ibeam]], \
                                                             name=r"power beam {0:d} (Gray)".format(ibeam + 1), marker="--", \
                                                             y_range_in=self.y_range_list[0], ax_flag="P_ecrh")
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[linear_beam.rhop, \
                                                                   linear_beam.j_beam[ibeam]], \
                    name=r"driven current beam {0:d} (Gray)".format(ibeam + 1), marker="--", \
                         y_range_in=self.y_range_list[1], ax_flag="j_eccd")
        for beam in linear_beam.rays:
            n_ray = int(len(beam) / 10)
            for ray in np.array(beam)[::n_ray]:
                self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                                                                     data=[ray["rhop"], \
                                                                           ray["T"]], \
                                                                     marker="-", \
                                                                     y_range_in=self.y_range_list[2], ax_flag="T_rho")
        self.axlist[0].set_xlabel("")
        self.axlist[1].set_xlabel("")
        self.axlist[0].set_xlim(0.0, 1.0)
        self.axlist[2].set_ylim(0.0, 1.2)
        self.create_legends("stacked_small")
        evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
        wx.PostEvent(Callee, evt_out)

    def overlap_plot(self, path, shot, time, EQSlice, BPD_rhop, BPD, BPD_labels, diag_rays, diag_resonances, linear_beam, quasi_linear_beam, \
                     noTB=False, ray_deop_plot=True):
        slicing = 5
        self.setup_axes("overlap_plot", "Beams poloidal view")
        tb_in_path = os.path.join(path, "{0:5d}_{1:1.3f}_rays/".format(shot, time))
#        x_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(xl)), xl, k=1)
#        y_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(yl)), yl, k=1)
#        xl = x_spl(np.linspace(0.0, 1.0, 4000))
#        yl = y_spl(np.linspace(0.0, 1.0, 4000))
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                         data=[linear_beam.rhop, \
                                                               linear_beam.PW], \
                                                         name=r"Gray", marker=":", color="black", \
                                                         y_range_in=self.y_range_list[0], ax_flag="P_ecrh")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[quasi_linear_beam.rhop, \
                                                                                   quasi_linear_beam.PW], \
                                                             name=r"RELAX", marker="+", color=(0.4, 0.4, 0.0), \
                                                             y_range_in=self.y_range_list[0], ax_flag="P_ecrh")
        for i in range(len(BPD)):
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[BPD_rhop[i], \
                                                                                   BPD[i]], \
                                                             name=BPD_labels[i], marker="-", \
                                                             y_range_in=self.y_range_list[1], ax_flag="j_weighted", vline=diag_resonances["rhop"][i])
        pw_torbeams = []
        j_torbeams = []
        rho_torbeams = []
        # print(path)
        if(os.path.isdir(path) and not noTB):
            i = 1
            filename = os.path.join(tb_in_path, "pw_j_beam_{0:1d}.dat".format(i))
            while(os.path.isfile(filename)):
                tb_file = np.loadtxt(filename)
                rho_torbeams.append(tb_file.T[0])
                pw_torbeams.append(tb_file.T[1])
                j_torbeams.append(tb_file.T[2])
                i += 1
                filename = os.path.join(tb_in_path, "pw_j_beam_{0:1d}.dat".format(i))
        Pw_beams = np.zeros(200)
        rhop_tb_profs = np.linspace(0.0, 1.0, 200)
        for ibeam in range(len(linear_beam.PW_beam)):
            if(ibeam < len(rho_torbeams)):
                if(np.sum(pw_torbeams[ibeam]) > 1.e-6 and np.all(np.isfinite(pw_torbeams[ibeam]))):
                    pw_spl = InterpolatedUnivariateSpline(rho_torbeams[ibeam], pw_torbeams[ibeam])
                    Pw_beams[:] += pw_spl(rhop_tb_profs)
                else:
                    print("TORBEAM ray skipped because of NAN or less than 1 W total power deposited")
        if(np.sum(Pw_beams) > 1.e-6):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                         data=[rhop_tb_profs , Pw_beams], \
                                                         name=r"TORBEAM", marker="--", \
                                                         y_range_in=self.y_range_list[0], ax_flag="P_ecrh")
        self.axlist[0].set_xlim(0.0, 1.0)
        self.create_legends("Te_no_ne_twinx")
        CS = self.axlist[2].contour(EQSlice.R, EQSlice.z, EQSlice.rhop.T, \
             levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[2].contour(EQSlice.R, EQSlice.z, EQSlice.rhop.T, \
             levels=[1.0], linewidths=3, colors="b", linestyles="-")
        fconf.plt_vessel(self.axlist[2])
        beam_count = 1
        R_torbeams = []
        z_torbeams = []
        x_torbeams = []
        y_torbeams = []
        if(os.path.isdir(tb_in_path)):
            i = 1
            filename = os.path.join(tb_in_path, "Rz_beam_{0:1d}.dat".format(i))
            filename_tor = os.path.join(tb_in_path, "xy_beam_{0:1d}.dat".format(i))
            while(os.path.isfile(filename) and os.path.isfile(filename_tor)):
                Rz_file = np.loadtxt(filename)
                Rz_file *= 1.e-2
                R_torbeams.append([Rz_file.T[0], Rz_file.T[2], Rz_file.T[4]])
                z_torbeams.append([Rz_file.T[1], Rz_file.T[3], Rz_file.T[5]])
                tor_file = np.loadtxt(filename_tor)
                tor_file *= 1.e-2
                x_torbeams.append([tor_file.T[0], tor_file.T[2], tor_file.T[4]])
                y_torbeams.append([tor_file.T[1], tor_file.T[3], tor_file.T[5]])
                i += 1
                filename = os.path.join(tb_in_path, "Rz_beam_{0:1d}.dat".format(i))
                filename_tor = os.path.join(tb_in_path, "xy_beam_{0:1d}.dat".format(i))
        if(ray_deop_plot):
            for beam in linear_beam.rays:
                ray_count = 1
                for ray in beam[::slicing]:
                    print("Plotting beam {0:d} ray {1:d}".format(beam_count, (ray_count - 1) * slicing + 1))
                    i = 0
                    pw_max = np.max(ray["PW"])
                    multicolor = True
                    if(not np.isfinite(pw_max)):
                        print("Nans or infs in ray no. {0:d}".format(ray_count))
                        multicolor = False
                    elif(pw_max == 0.0):
                        print("Encountered zero max deposited power in ray no. {0:d}".format(ray_count))
                        multicolor = False
                    else:
                        cmap = plt.cm.ScalarMappable(plt.Normalize(0, pw_max), "autumn")
                    if(not (np.all(np.isfinite(ray["R"])) and np.all (np.isfinite(ray["z"])))):
                        print("Ray no. {0:d} has non-finite coordinates".format(ray_count))
                        print("Ray no. {0:d} skipped".format(ray_count))
                        continue
                    i_start = 0
                    i_end = 1
                    x = ray["R"] * np.cos(ray["phi"])
                    y = ray["R"] * np.sin(ray["phi"])
                    if(multicolor):
                        while(i_end + 1 < len(ray["R"])):
                            while(np.abs((ray["PW"][i_start] - ray["PW"][i_end]) / (pw_max)) < 0.01 and i_end + 1 < len(ray["R"])):
                                i_end += 1
                            color = cmap.to_rgba(ray["PW"][i_start])[0:3]
                            self.axlist[2].plot(ray["R"][i_start:i_end + 1], ray["z"][i_start:i_end + 1], color=color)
                            i_start = i_end
                    else:
                        self.axlist[0].plot(ray["R"], ray["R"], color=color)
                        self.axlist_2[0].plot(x, y, color=color)
                    ray_count += 1
                beam_count += 1
            total_rays = (beam_count - 1) * (ray_count - 1)
            print("Now rendering " + str(total_rays) + " rays - hold on a second!")
        else:
            pw_spl = InterpolatedUnivariateSpline(quasi_linear_beam.rhop, quasi_linear_beam.PW / np.max(quasi_linear_beam.PW), ext=1)
            Depo2D = pw_spl(EQSlice.rhop)
            Depo2D[Depo2D < 1.e-3] = 0.0
            CS2 = self.axlist[2].contourf(EQSlice.R, EQSlice.z, Depo2D.T, cmap=plt.cm.get_cmap("Reds"), levels=np.linspace(0.05, 1.0, 11))
        for beam_count in range(len(linear_beam.rays)):
            if(beam_count < len(R_torbeams)):
                for i_tb in range(3):
                    self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                                                                 data=[R_torbeams[beam_count][i_tb], \
                                                                       z_torbeams[beam_count][i_tb]], \
                                                                         color=(0.0, 0.0, 0.0), marker="--", \
                                                                         y_range_in=self.y_range_list[2], ax_flag="Rz")
        if(len(diag_rays["R"]) > 0):
            for iray in range(len(diag_rays["R"][0])):
                self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                                                                     data=[diag_rays["R"][0][iray], \
                                                                           diag_rays["z"][0][iray]], \
                                                                     color=(0.0, 0.0, 0.0), marker="--", \
                                                                     y_range_in=self.y_range_list[2], ax_flag="Rz")
        for ich in range(len(diag_resonances["R"])):
            self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                                                                 data=[diag_resonances["R"][ich], \
                                                                       diag_resonances["z"][ich]], \
                                                                         color=(0.0, 0.0, 1.0), marker="+", \
                                                                         y_range_in=self.y_range_list[2], ax_flag="Rz")
        self.axlist[2].set_aspect("equal")
        self.axlist[2].set_xlim(0.7, 2.5)
        self.axlist[2].set_ylim(-1.75, 1.75)
        for ax in self.axlist:
            ax.get_xaxis().set_major_locator(NLocator(nbins=4, prune='lower'))
            ax.get_xaxis().set_minor_locator(NLocator(nbins=8))
            ax.get_yaxis().set_major_locator(NLocator(nbins=7))
            ax.get_yaxis().set_minor_locator(NLocator(nbins=14))
        return self.fig

    def Plot_Rz_Res(self, shot, time, R_cold, z_cold, R_warm, z_warm, EQ_obj=False, \
                    Rays=None, tb_Rays=None, straight_Rays=None, \
                    R_warm_comp=None, z_warm_comp=None, \
                    EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, eq_aspect_ratio=True):
        shotstr = "\#" + str(shot) + " t = " + "{0:2.3f}".format(time) + " s  "
        self.setup_axes("single", r"ECE cold res. and ray " + shotstr, "Toroidal angle")
        stop_at_axis = False
        if(stop_at_axis):
            if(EQ_obj == False):
                EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
            R, z = EQ_obj.get_axis(float(time))
        if(Rays is not None):
            if(not np.isscalar(Rays[0][0][0])):  # multiple rays stored
                for i in range(len(Rays)):
                    Rays[i] = Rays[i][0]  # Select the central ray
        if(Rays is not None):
            if(tb_Rays is not None or straight_Rays is not None):
                i_axis = len(Rays[0][1])
                if(stop_at_axis):
                    i_axis = 0
                    for i in range(len(Rays[0][0])):
                        i_axis = i
                        if(Rays[0][0][::-1][i] < R):
                            break
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ Rays[0][0][:i_axis] , Rays[0][1][:i_axis] ], marker="-", color="blue", \
                      name=r"Rays according to ECRad", ax_flag="Rz")
            else:
                i_axis = len(Rays[0][1])
                if(stop_at_axis):
                    i_axis = 0
                    for i in range(len(Rays[0][0])):
                        i_axis = i
                        if(Rays[0][0][::-1][i] < R):
                            break
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ Rays[0][0][:i_axis] , Rays[0][1][:i_axis] ], marker="-", color="blue", \
                      ax_flag="Rz")
            for ray in Rays[1:len(Rays) - 1]:
                try:
                    i_axis = len(Rays[0][::-1])
                    if(stop_at_axis):
                        i_axis = 0
                        for i in range(len(ray[0])):
                            i_axis = i
                            if(ray[0][::-1][i] < R):
                                break
                    else:
                        i_axis = len(ray[0])
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ray[0][::-1][:i_axis] , ray[1][::-1][:i_axis] ], marker="-", color="blue", \
                      ax_flag="Rz")
                except ValueError:
                    print(ray[0], ray[1])
        if(straight_Rays is not None):
            print(straight_Rays[0][0] , straight_Rays[0][1])
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[straight_Rays[0][0] , straight_Rays[0][1] ], marker="--", color="black", \
                      name=r"Straight rays", ax_flag="Rz")
            for ray in straight_Rays[1:len(straight_Rays) - 1]:
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ray[0] , ray[1] ], marker="--", color="black", \
                      ax_flag="Rz")
        if(tb_Rays is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[tb_Rays[0][0] , tb_Rays[0][1] ], marker=":", color=(126.0 / 255.0, 0.0, 126.0 / 255.0), \
                      name=r"Rays according to TORBEAM", ax_flag="Rz")
            for ray in tb_Rays[1:len(tb_Rays) - 1]:
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ray[0] , ray[1] ], marker=":", color=(126.0 / 255.0, 0.0, 126.0 / 255.0), \
                      ax_flag="Rz")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                  self.y_range_list[0], data=[R_warm , z_warm ], marker=r"o", color=(0.0, 126.0 / 255.0, 0.0), \
                  name=r"Warm res. pos. proposed model", \
                  ax_flag="Rz")
        if(R_warm_comp is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                  self.y_range_list[0], data=[R_warm_comp , z_warm_comp ], marker=r"s", color=(0.0, 0.0, 126.0 / 255.0), \
                  name=r"Warm res. pos. fully rel. dispersion", \
                  ax_flag="Rz")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[R_cold , z_cold ], marker=r"*", color=(1.0, 0.0, 0.0), \
                      name=r"Cold res. pos.", \
                      ax_flag="Rz")
        equilibrium = True
        if(equilibrium and EQ_obj == False):
            if(EQ_exp and EQ_ed is not None and EQ_diag is not None):
                EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
            else:
                EQ_obj = EQData(shot, EQ_exp='AUGD', EQ_diag='EQH', EQ_ed=0)
            EQ_Slice = EQ_obj.GetSlice(time)
            R = EQ_Slice.R
            z = EQ_Slice.z
            rhop = EQ_Slice.rhop
        else:
            R = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].R
            z = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].z
            rhop = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].rhop
        CS = self.axlist[0].contour(R, z, rhop.T, \
            levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[0].contour(R, z, rhop.T, \
            levels=[1.0], linewidths=3, colors="b", linestyles="-")
        plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS2, inline=1, fontsize=12)
        if(AUG):
            fconf.plt_vessel(self.axlist[0])
        if(eq_aspect_ratio):
            self.axlist[0].set_aspect("equal")
        self.axlist[0].set_xlim((0.8, 2.4))
        self.axlist[0].set_ylim((-0.75, 1.0))
        self.create_legends("vessel")
        return self.fig


#    def plt_vessel(self, ax):
#        vessel_file = open(os.path.join(home, "Documentation", "Data", "vessel.coords.txt"), 'r')
#        struct_dict = {}
#        struct_name = None
#        for line in vessel_file:
#            if len(line) == 1:
#                continue
#            elif line[0] == '#':
#                struct_name = line[1:].strip()
#                struct_dict[struct_name] = []
#            else:
#                struct_dict[struct_name].append([float(n) for n in line.split()])
#
#        vessel_file.close()
#        for k, items in struct_dict.iteritems():
#            struct_dict[k] = np.array(items).T
#        struct_dict = struct_dict
#        for _, struct in struct_dict.iteritems():
#                if np.size(struct) > 1:
#                    ax.plot(struct[0], struct[1], 'k', lw=0.5)
        # cmap = plt.cm.get_cmap("autumn")

    def time_trace_for_calib(self, fig, shot, time, diag_time, IDA, IDA_labels, ECE, ECE_labels, ECRad, ECRad_labels, diag=None, diag_labels=None, divertor_cur=None):
        if(divertor_cur is not None and diag is not None):
            self.setup_axes("stacked_large", r"Timetraces for \#" + str(shot))
        elif(diag is not None or divertor_cur is not None):
            self.setup_axes("stacked_small", r"Timetraces for \#" + str(shot))
        elif(len(ECE) > 0):
            self.setup_axes("stacked_very_small", r"Timetraces for \#" + str(shot))
        else:
            self.setup_axes("single", r"Timetraces for \#" + str(shot))
        for i in range(len(IDA)):
            if(len(IDA[i]) == 1):
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[time, IDA[i]], marker="+", \
                    name=IDA_labels[i], \
                    coloumn=1, ax_flag="Te_trace")
            else:
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[time, IDA[i]], \
                    name=IDA_labels[i], \
                    coloumn=1, ax_flag="Te_trace")
        for i in range(len(ECE)):
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=[time, ECE[i]], marker="+", \
                    name=ECE_labels[i], coloumn=1, ax_flag="Trad_trace")  # \
        for i in range(len(ECRad)):
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=[time, ECRad[i]], \
                     marker="-", \
                    coloumn=1, ax_flag="Trad_trace")  # name = ECRad_labels[i],
        if(diag is not None):
            for i in range(len(diag)):
                self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                        self.y_range_list[2] , data=[diag_time, diag[i]], \
                        name=diag_labels[i], \
                        coloumn=1, ax_flag="diag_trace")
        if(divertor_cur is not None and diag is not None):
            self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                    self.y_range_list[3] , data=[divertor_cur[0], divertor_cur[1]], \
                    name="Divertor current", marker="-", \
                    coloumn=1, ax_flag="diag_trace")
        elif(divertor_cur is not None):
            self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                    self.y_range_list[2] , data=[divertor_cur[0], divertor_cur[1]], \
                    name="Divertor current", marker="-", \
                    coloumn=1, ax_flag="diag_trace")
        self.axlist[0].set_xlim((np.min(time), np.max(time)))
        self.create_legends("single")
        # self.finishing_touches()
        return fig

    def stacked_plot_time_trace(self, fig, filename_list, shot, labels):
        self.setup_axes("stacked_2_twinx", r"Timetrace of \#" + str(shot))
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename_list[0], maxlines=0, color="blue", \
                    name=labels[0], \
                    coloumn=1, ax_flag="ne_trace")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename_list[0], maxlines=0, color="black", \
                    name=labels[1], \
                    coloumn=2, ax_flag="ne_trace")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , filename=filename_list[1], maxlines=0, color="blue", \
                    name=labels[2], \
                    coloumn=1, ax_flag="P_trace")
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                    self.y_range_list[2] , filename=filename_list[2], maxlines=0, color="red", \
                    name=labels[3], \
                    coloumn=1, ax_flag="P_trace")
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                    self.y_range_list[0] , filename=filename_list[3], maxlines=0, color="blue", \
                    name=r"TRad $\rho_\mathrm{pol} = " + \
                    r"{:}$".format(labels[4]), \
                    coloumn=1, ax_flag="Te_trace")
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                    self.y_range_list[3] , filename=filename_list[3], maxlines=0, color="red", \
                    name=r"TRad $\rho_\mathrm{pol} = " + \
                    r"{:}$".format(labels[5]), \
                    coloumn=2, ax_flag="Te_trace")
        self.axlist[4], self.y_range_list[4] = self.add_plot(self.axlist[4], \
                    self.y_range_list[4] , filename=filename_list[4], maxlines=0, color="blue", \
                    name=labels[6], \
                    coloumn=1, ax_flag="cnt_trace", log_flag=True, y_scale=1.e-3)
        self.axlist[5], self.y_range_list[5] = self.add_plot(self.axlist[5], \
                    self.y_range_list[5] , filename=filename_list[4], maxlines=0, color="black", \
                    name=labels[7], \
                    coloumn=2, ax_flag="n_trace", log_flag=True, y_scale=1.e-9)
        self.create_legends("stacked_2_twinx")
        self.axlist[0].set_xlim(0.0, 6.0)
        return fig

    def stacked_plot_time_trace_3(self, fig, filename_list, shot, labels):
        self.setup_axes("stacked_1_twinx", r"Timetrace of #" + str(shot))
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename_list[0], maxlines=0, color="blue", \
                    name=labels[0], \
                    coloumn=1, ax_flag="ne_trace", sample=10)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , filename=filename_list[0], maxlines=0, color="black", \
                    name=labels[1], \
                    coloumn=2, ax_flag="ne_trace", sample=10)
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , filename=filename_list[1], maxlines=0, color="blue", \
                    name=labels[2], \
                    coloumn=1, ax_flag="P_trace", sample=10)
        self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                    self.y_range_list[2] , filename=filename_list[2], maxlines=0, color="red", \
                    name=labels[3], \
                    coloumn=1, ax_flag="P_trace", sample=10)
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                    self.y_range_list[0] , filename=filename_list[3], maxlines=0, color="blue", \
                    name=r"TRad $\rho_\mathrm{pol} = " + \
                    r"{:}$".format(labels[4]), \
                    coloumn=1, ax_flag="Te_trace", sample=100)
        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                    self.y_range_list[3] , filename=filename_list[3], maxlines=0, color="red", \
                    name=r"TRad $\rho_\mathrm{pol} = " + \
                    r"{:}$".format(labels[5]), \
                    coloumn=2, ax_flag="Te_trace", sample=100)
        # self.axlist[4], self.y_range_list[4] = self.add_plot(self.axlist[4], \
        #            self.y_range_list[4] ,filename = filename_list[4], maxlines = 0, color = "blue",\
        #            name = labels[6], \
        #            coloumn = 1, ax_flag = "cnt_trace", log_flag = True,y_scale = 1.e-3)
        # self.axlist[5], self.y_range_list[5] = self.add_plot(self.axlist[5], \
        #            self.y_range_list[5] ,filename = filename_list[4], maxlines = 0, color = "black",\
        #            name =labels[7], \
        #            coloumn = 2, ax_flag = "n_trace", log_flag = True,y_scale = 1.e-9)
        self.create_legends("stacked_1_twinx")
        self.axlist[0].set_xlim(0.0, 6.0)
        return fig

    def plot_ray(self, shot, time, ray, index=0, EQ_obj=False, H=True, R_cold=None, z_cold=None, s_cold=None, straight=False, eq_aspect_ratio=True):
        self.setup_axes("ray", "Ray", "Hamiltonian")
        # eqi = equ_map()
        equilibrium = True
        if(equilibrium):
            if(EQ_obj):
                if(EQ_obj.Ext_data):
                    R = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].R
                    z = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].z
                    rhop = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].rhop
            else:
                EQ_obj = EQData(shot)
                EQ_Slice = EQ_obj.GetSlice(time)
                R = EQ_Slice.R
                z = EQ_Slice.z
                rhop = EQ_Slice.rhop
            CS = self.axlist[0].contour(R, z, rhop.T, \
                                        levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
            CS2 = self.axlist[0].contour(R, z, rhop.T, \
                levels=[1.0], linewidths=3, colors="b", linestyles="-")
            R_ax, z_ax = EQ_obj.get_axis(time)
            rhop_spl = RectBivariateSpline(R, z, rhop)
            z_eval = np.zeros(len(R))
            z_eval = z_ax
            rhop_axis_plane = rhop_spl(R, z_eval, grid=False)
            tor_cont_list = []
            rhop_cont = np.linspace(0.1, 1.2, 12)
            for rhop_cont_entry in rhop_cont:
                if(rhop_cont_entry not in [0.0, 1.0]):
                    rhop_axis_plane_root_spl = InterpolatedUnivariateSpline(R, rhop_axis_plane - rhop_cont_entry)
                    tor_cont_list.append(rhop_axis_plane_root_spl.roots())
            tor_cont_list = np.hstack(np.array(tor_cont_list))
            rhop_axis_plane_root_spl = InterpolatedUnivariateSpline(R, rhop_axis_plane - 1.0)
            sep_R = rhop_axis_plane_root_spl = rhop_axis_plane_root_spl.roots()
            for i in range(len(sep_R)):
                self.axlist[1].add_patch(pltCircle([0.0, 0.0], sep_R[i], edgecolor='b', facecolor='none', linestyle="-"))
            self.axlist[1].add_patch(pltCircle([0.0, 0.0], R_ax, edgecolor='b', facecolor='none', linestyle="-"))
            for i in range(len(tor_cont_list)):
                self.axlist[1].add_patch(pltCircle([0.0, 0.0], tor_cont_list[i], edgecolor='k', facecolor='none', linestyle="--"))
        if(R_cold is not None and z_cold is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R_cold, z_cold], \
                name=r"Cold resonance", marker="+", color=(0.e0, 126.0 / 255, 0.e0), \
                     y_range_in=self.y_range_list[0], ax_flag="Rz")
        if(AUG):
            fconf.plt_vessel(self.axlist[0])
            fconf.plt_vessel(self.axlist[1], pol=False)
        if(np.iterable(ray)):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray[0].R, ray[0].z], \
                name=r"ECRad ray", marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                     y_range_in=self.y_range_list[0], ax_flag="Rz")
            if(s_cold is not None):
                x_spl = InterpolatedUnivariateSpline(ray[0].s, ray[0].x)
                y_spl = InterpolatedUnivariateSpline(ray[0].s, ray[0].y)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[x_spl(s_cold), y_spl(s_cold)], \
                            name=r"Cold resonance", marker="+", color=(0.e0, 126.0 / 255, 0.e0), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray[0].x, ray[0].y], \
                        name=r"ECRad ray", marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                             y_range_in=self.y_range_list[1], ax_flag="xy")
            if(H):
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray[0].s, ray[0].H], \
                            name=r"Hamiltonian $\times 10^5$", marker="-", color=(0, 0.e0, 0.0), \
                                 y_range_in=self.y_range_list_2[0], ax_flag="H", y_scale=1.e5)
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray[0].s, ray[0].N], \
                            name=r"Ray $N_{\omega, \mathrm{ray}}$ ", marker="--", color=(126.0 / 255, 0.e0, 0), \
                                 y_range_in=self.y_range_list_2[0], ax_flag="H")
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray[0].s, ray[0].N_cold], \
                            name=r"Ray $N_{\omega, \mathrm{disp}}$ ", marker="-.", color=(0.e0, 0.e0, 26.0 / 255), \
                                 y_range_in=self.y_range_list_2[0], ax_flag="H")
            if(len(ray) > 1):
                for single_ray in ray[1:len(ray)]:
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[single_ray.R, single_ray.z], \
                        marker="--", color=(0.e0, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[single_ray.R[0], single_ray.z[0]], \
                        marker="+", color=(0.e0, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[single_ray.x, single_ray.y], \
                            marker="--", color=(0.e0, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[single_ray.x[0], single_ray.y[0]], \
                            marker="+", color=(0.e0, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
            if(type(ray[0].R_tb) != int):
                if(straight):
                    ray_label = "Straight"
                else:
                    ray_label = "TORBEAM"
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray[0].R_tb, ray[0].z_tb], \
                        name=ray_label + r" ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                if(type(ray[0].R_tbp1) != int):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray[0].R_tbp1, ray[0].z_tbp1], \
                        name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray[0].R_tbp2, ray[0].z_tbp2], \
                        marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
            if(type(ray[0].x_tb) != int):
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray[0].x_tb, ray[0].y_tb], \
                            name=ray_label + r" ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
                if(type(ray[0].R_tbp1) != int):
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray[0].x_tbp1, ray[0].y_tbp1], \
                            name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray[0].x_tbp2, ray[0].y_tbp2], \
                            name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
        else:
            if(s_cold is not None):
                x_spl = InterpolatedUnivariateSpline(ray.s, ray.x)
                y_spl = InterpolatedUnivariateSpline(ray.s, ray.y)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[x_spl(s_cold), y_spl(s_cold)], \
                            name=r"Cold resonance", marker="+", color=(0.e0, 126.0 / 255, 0.e0), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R, ray.z], \
                    name=r"ECRad ray", marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
            if(H):
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray.s, ray.H], \
                            name=r"Hamiltonian $\times 10^5$", marker="-", color=(0, 0.e0, 0.0), \
                                 y_range_in=self.y_range_list_2[0], ax_flag="H", y_scale=1.e5)
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray.s, ray.N], \
                            name=r"Ray $N_{\omega, \mathrm{ray}}$ ", marker="--", color=(126.0 / 255, 0.e0, 0), \
                                 y_range_in=self.y_range_list_2[0], ax_flag="H")
                self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray.s, ray.N_cold], \
                            name=r"Ray $N_{\omega, \mathrm{disp}}$ ", marker="-.", color=(0.e0, 0.e0, 26.0 / 255), \
                                 y_range_in=self.y_range_list_2[0], ax_flag="H")
            if(type(ray.R_tb) != int):
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R_tb, ray.z_tb], \
                        name=r"TORBEAM ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                if(type(ray[0].R_tbp1) != int):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R_tbp1, ray.z_tbp1], \
                        name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R_tbp2, ray.z_tbp2], \
                        name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray.x, ray.y], \
                        name=r"ECRad ray", marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                             y_range_in=self.y_range_list[1], ax_flag="xy")
            if(type(ray.x_tb) != int):
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray.x_tb, ray.y_tb], \
                            name=r"TORBEAM ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
                if(type(ray[0].R_tbp1) != int):
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray.x_tbp1, ray.y_tbp1], \
                            name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
                    self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray.x_tbp2, ray.y_tbp2], \
                            name=ray_label + r" peripheral ray", marker=":", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
#        if(EQ_obj == False and AUG == True):
#            fconf.plt_vessel(self.axlist[1], pol=False)
#            fconf.plt_eq_tor(self.axlist[1], int(shot), float(time))
        self.create_legends("single")
        if(eq_aspect_ratio):
            self.axlist[0].set_aspect("equal")
            self.axlist[1].set_aspect("equal")
        self.axlist[0].set_xlim(0.7, 2.5)
        self.axlist[0].set_ylim(-1.75, 1.75)
        self.axlist[1].set_ylim(-1.0, 1.0)
        if(H):
            return self.fig, self.fig_2
        else:
            return self.fig

    def tb_plot(self, shot, time, folder):
        self.setup_axes("ray", "Ray", "Hamiltonian")
        pol_beamfile = np.loadtxt(os.path.join(folder, "t1_LIB.dat"))
        Rc = pol_beamfile.T[0] / 100.0
        zc = pol_beamfile.T[1] / 100.0
        s_max = 0
        i = 1
        while(i <= len(Rc) - 1):
            s_max += np.sqrt((Rc[i] - Rc[i - 1]) ** 2 + (zc[i] - zc[i - 1]) ** 2)
            i += 1
        pwfile = np.loadtxt(os.path.join(folder, "t2_LIB.dat"))
        pw_init = pwfile.T[1]
        pw = np.zeros(len(Rc))
        index = pwfile.T[3].astype(np.int)
        pw[index[0] - 1: index[-1]] = pw_init
        pw_spl = InterpolatedUnivariateSpline(np.linspace(0, s_max, len(pw)), pw, k=1)
        pw = pw_spl(np.linspace(0.0, s_max, 4000))
        R_spl = InterpolatedUnivariateSpline(np.linspace(0, s_max, len(Rc)), Rc, k=1)
        z_spl = InterpolatedUnivariateSpline(np.linspace(0, s_max, len(zc)), zc, k=1)
        Rc = R_spl(np.linspace(0.0, s_max, 4000))
        zc = z_spl(np.linspace(0.0, s_max, 4000))
        Ru = pol_beamfile.T[2] / 100.0
        zu = pol_beamfile.T[3] / 100.0
#        R_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(Ru)), Ru, k=1)
#        z_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(zu)), zu, k=1)
#        Ru = R_spl(np.linspace(0.0, 1.0, 4000))
#        zu = z_spl(np.linspace(0.0, 1.0, 4000))
        Rl = pol_beamfile.T[4] / 100.0
        zl = pol_beamfile.T[5] / 100.0
#        R_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(Rl)), Rl, k=1)
#        z_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(zl)), zl, k=1)
#        Rl = R_spl(np.linspace(0.0, 1.0, 4000))
#        zl = z_spl(np.linspace(0.0, 1.0, 4000))
        tor_beamfile = np.loadtxt(os.path.join(folder, "t1tor_LIB.dat"))
        xc = tor_beamfile.T[0] / 100.0
        yc = tor_beamfile.T[1] / 100.0
        s_max = 0
        i = 1
        while(i <= len(xc) - 1):
            s_max += np.sqrt((xc[i] - xc[i - 1]) ** 2 + (yc[i] - yc[i - 1]) ** 2)
            i += 1
        x_spl = InterpolatedUnivariateSpline(np.linspace(0, s_max, len(xc)), xc, k=1)
        y_spl = InterpolatedUnivariateSpline(np.linspace(0, s_max, len(yc)), yc, k=1)
        xc = x_spl(np.linspace(0.0, s_max, 4000))
        yc = y_spl(np.linspace(0.0, s_max, 4000))
        xu = tor_beamfile.T[2] / 100.0
        yu = tor_beamfile.T[3] / 100.0
#        x_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(xu)), xu, k=1)
#        y_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(yu)), yu, k=1)
#        xu = x_spl(np.linspace(0.0, 1.0, 4000))
#        yu = y_spl(np.linspace(0.0, 1.0, 4000))
        xl = tor_beamfile.T[4] / 100.0
        yl = tor_beamfile.T[5] / 100.0
#        x_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(xl)), xl, k=1)
#        y_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(yl)), yl, k=1)
#        xl = x_spl(np.linspace(0.0, 1.0, 4000))
#        yl = y_spl(np.linspace(0.0, 1.0, 4000))
        EQ_obj = EQData(shot, EQ_exp='AUGD', EQ_diag='EQH', EQ_ed=0)
        EQ_Slice = EQ_obj.GetSlice(time)
        R = EQ_Slice.R
        z = EQ_Slice.z
        rhop = EQ_Slice.rhop
        CS = self.axlist[0].contour(R, z, rhop.T, \
             levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[0].contour(R, z, rhop.T, \
             levels=[1.0], linewidths=3, colors="b", linestyles="-")
        fconf.plt_vessel(self.axlist[0])
        i = 0
        while(i + 2 < len(Rc)):
            if(i + 10 < len(Rc)):
                end = i + 20
            else:
                end = len(Rc)
            if(np.any(np.abs(pw)) > 1.e8 or np.any(np.isinf(pw)) or np.any(np.isnan(pw)) or np.any(pw < 0)):
                color = (0.0, 0.0, 0.0)
            else:
                color = (pw[i] / np.max(pw), 0.e0, 1.e0 - pw[i] / np.max(pw))
                if(color[0] > 1 or color[0] < 0.0 or color[2] > 1 or color[2] < 0.0):
                    print(i, pw[i], np.max(pw))
                    print("Error when determining color")
                    return
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[Rc[i:end], zc[i:end]], \
                marker="-", color=color, \
                y_range_in=self.y_range_list[0], ax_flag="Rz")
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[xc[i:end], yc[i:end]], \
                    marker="-", color=color, \
                    y_range_in=self.y_range_list[1], ax_flag="xy")
            i = end + 1
        color = (0.0, 0.0, 1.0)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[Ru, zu], \
                    marker=":", color=color, \
                    y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[Rl, zl], \
                    marker=":", color=color, \
                    y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[xu, yu], \
                marker=":", color=color, \
                y_range_in=self.y_range_list[1], ax_flag="xy")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[xl, yl], \
                marker=":", color=color, \
                y_range_in=self.y_range_list[1], ax_flag="xy")
        if(AUG):
            fconf.plt_vessel(self.axlist[1], pol=False)
            fconf.plt_eq_tor(self.axlist[1], int(shot), float(time))
        # self.create_legends("single")
        self.axlist[0].set_aspect("equal")
        self.axlist[1].set_aspect("equal")
        self.axlist[0].set_xlim(0.7, 2.5)
        self.axlist[0].set_ylim(-1.75, 1.75)
        self.axlist[1].set_ylim(-1.0, 1.0)
        self.axlist_2[0].set_ylim(0, 1)
        plt.show()

    def grid_plot(self, args):
        Callee = args[0]
        y = args[1]
        rho = args[2]
        pw = args[3]
        pw_start = args[4]
        pw_stop = args[5]
        self.setup_axes("twinx", r"Suggested grid points for RELAX")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[y, np.array(range(len(y)), dtype=np.int)], marker="+", color="black", ax_flag="Grid")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    self.y_range_list[0] , data=[y[[pw_start, pw_stop]], np.array(range(len(y)), dtype=np.int)[[pw_start, pw_stop]]], marker="^", color="red", ax_flag="Grid")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    self.y_range_list[1] , data=[rho, pw], marker="-", color="blue", ax_flag="P_ecrh_tor")
        evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
        wx.PostEvent(Callee, evt_out)

    def beam_plot(self, args):
        slicing = 3
        Callee = args[0]
        shot = args[1]
        time = args[2]
        R = args[3]
        z = args[4]
        rhop = args[5]
        R_axis = args[6]
        z_axis = args[7]
        linear_beam = args[8]
        if(len(args) > 9):
            result = args[9]
            channel_list = args[10]
            mode = args[11]
            tb_path = args[12]
        else:
            result = None
            tb_path = None
        self.setup_axes("ray_2_fig", "Beams poloidal view", "Beams top view")
#        x_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(xl)), xl, k=1)
#        y_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(yl)), yl, k=1)
#        xl = x_spl(np.linspace(0.0, 1.0, 4000))
#        yl = y_spl(np.linspace(0.0, 1.0, 4000))
        rhop_spl = RectBivariateSpline(R, z, rhop)
        z_eval = np.zeros(len(R))
        z_eval[:] = z_axis
        rhop_axis_plane = rhop_spl(R, z_eval, grid=False)
        tor_cont_list = []
        CS = self.axlist[0].contour(R, z, rhop.T, \
                                    levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[0].contour(R, z, rhop.T, \
                                     levels=[1.0], linewidths=3, colors="b", linestyles="-")
        rhop_cont = np.linspace(0.1, 1.2, 12)
        for rhop_cont_entry in rhop_cont:
            if(rhop_cont_entry not in [0.0, 1.0]):
                rhop_axis_plane_root_spl = InterpolatedUnivariateSpline(R, rhop_axis_plane - rhop_cont_entry)
                tor_cont_list.append(rhop_axis_plane_root_spl.roots())
        tor_cont_list = np.hstack(np.array(tor_cont_list))
        rhop_axis_plane_root_spl = InterpolatedUnivariateSpline(R, rhop_axis_plane - 1.0)
        sep_R = rhop_axis_plane_root_spl.roots()
        for i in range(len(sep_R)):
            self.axlist_2[0].add_patch(pltCircle([0.0, 0.0], sep_R[i], edgecolor='b', facecolor='none', linestyle="-"))
        self.axlist_2[0].add_patch(pltCircle([0.0, 0.0], R_axis, edgecolor='b', facecolor='none', linestyle="-"))
        for i in range(len(tor_cont_list)):
            self.axlist_2[0].add_patch(pltCircle([0.0, 0.0], tor_cont_list[i], edgecolor='k', facecolor='none', linestyle="--"))
        if(AUG):
            fconf.plt_vessel(self.axlist[0])
            fconf.plt_vessel(self.axlist_2[0], pol=False)
        beam_count = 1
        R_torbeams = []
        z_torbeams = []
        x_torbeams = []
        y_torbeams = []
        if(tb_path is None):
            path = "/pfs/work/g2sdenk/TB/{0:5d}_{1:1.3f}_rays/".format(shot, time)
        else:
            path = os.path.join(tb_path, "{0:5d}_{1:1.3f}_rays".format(shot, time)) + os.sep
        if(os.path.isdir(path)):
            i = 1
            filename = os.path.join(path, "Rz_beam_{0:1d}.dat".format(i))
            filename_tor = os.path.join(path, "xy_beam_{0:1d}.dat".format(i))
            while(os.path.isfile(filename) and os.path.isfile(filename_tor)):
                Rz_file = np.loadtxt(filename)
                Rz_file *= 1.e-2
                R_torbeams.append([Rz_file.T[0], Rz_file.T[2], Rz_file.T[4]])
                z_torbeams.append([Rz_file.T[1], Rz_file.T[3], Rz_file.T[5]])
                tor_file = np.loadtxt(filename_tor)
                tor_file *= 1.e-2
                x_torbeams.append([tor_file.T[0], tor_file.T[2], tor_file.T[4]])
                y_torbeams.append([tor_file.T[1], tor_file.T[3], tor_file.T[5]])
                i += 1
                filename = os.path.join(path, "Rz_beam_{0:1d}.dat".format(i))
                filename_tor = os.path.join(path, "xy_beam_{0:1d}.dat".format(i))
        for beam in linear_beam.rays:
            ray_count = 1
            pw_max = -np.inf
            ray_pw = []
            for ray in beam:
                ray_pw.append(ray["PW"])
                pw_max_ray = np.max(ray["PW"])
                if(np.isfinite(pw_max_ray)):
                    if(pw_max_ray > pw_max):
                        pw_max = pw_max_ray
            print("Maximum power for beam no. {0:d}: {1:1.3e}".format(beam_count, pw_max))
            if(not np.isfinite(pw_max)):
                print("Nans or infs in beam no. {0:d}".format(beam_count))
                multicolor = False
            elif(pw_max == 0.0):
                print("Encountered zero max deposited power in beam no. {0:d}".format(beam_count))
                multicolor = False
            else:
                cmap = plt.cm.ScalarMappable(plt.Normalize(0.0, pw_max / 1.e3), "gist_rainbow")
                ray_pw = np.array(ray_pw)
                cmap.set_array(ray_pw)
                multicolor = True
            for ray in beam[::slicing]:
                pw_ratio = np.max(ray["PW"]) / pw_max
                print("Max power fraction for ray  no. {0:d}: {1:1.3e}".format((ray_count - 1) * slicing + 1, pw_ratio))
                if(pw_ratio < 0.01):
                    print("Less than 1% power in ray - skipping.")
                    continue
                print("Plotting beam {0:d} ray {1:d}".format(beam_count, (ray_count - 1) * slicing + 1))
                i = 0
                if(not (np.all(np.isfinite(ray["R"])) and np.all (np.isfinite(ray["z"])))):
                    print("Ray no. {0:d} has non-finite coordinates".format(ray_count))
                    print("Ray no. {0:d} skipped".format(ray_count))
                    continue
                if(not np.all(np.isfinite(ray["PW"]))):
                    print("Ray no. {0:d} has non-finite power".format(ray_count))
                    print("Ray no. {0:d} skipped".format(ray_count))
                    continue
                i_start = 0
                i_end = 1
                x = ray["R"] * np.cos(ray["phi"])
                y = ray["R"] * np.sin(ray["phi"])
                if(multicolor):
                    while(i_end + 1 < len(ray["R"])):
                        while(np.abs((ray["PW"][i_start] - ray["PW"][i_end]) / pw_max) < 0.01 and i_end + 1 < len(ray["R"])):
                            i_end += 1
                        color = cmap.to_rgba((np.max(ray["PW"]) - ray["PW"][i_start]) / 1.e3, pw_ratio)
                        self.axlist[0].plot(ray["R"][i_start:i_end + 1], ray["z"][i_start:i_end + 1], color=color)
                        self.axlist_2[0].plot(x[i_start:i_end + 1], y[i_start:i_end + 1], color=color[0:3], alpha=color[3])
                        i_start = i_end
                else:
                    self.axlist[0].plot(ray["R"], ray["R"], color="b")
                    self.axlist_2[0].plot(x, y, color="b")
                ray_count += 1
            beam_count += 1
        total_rays = (beam_count - 1) * (ray_count - 1)
        for beam_count in range(len(linear_beam.rays)):
            if(beam_count < len(R_torbeams)):
                for i_tb in range(3):
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                 data=[R_torbeams[beam_count][i_tb], \
                                                                       z_torbeams[beam_count][i_tb]], \
                                                                         color=(0.0, 0.0, 0.0), marker="-", linewidth=1, \
                                                                         y_range_in=self.y_range_list[0], ax_flag="Rz")
                    self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[x_torbeams[beam_count][i_tb], \
                                                                                                     y_torbeams[beam_count][i_tb]], \
                                                                             color=(0.0, 0.0, 0.0), marker="-", linewidth=1, \
                                                                             y_range_in=self.y_range_list_2[0], ax_flag="xy")
                color = "magenta"
        print("Now rendering " + str(total_rays) + " rays - hold on a second!")
        self.axlist[0].set_aspect("equal")
        self.axlist_2[0].set_aspect("equal")
        cb = self.fig.colorbar(cmap)
        cb.set_label(r"$P_\mathrm{ray} [\si{\kilo\watt}]$")
        cb2 = self.fig_2.colorbar(cmap)
        self.axlist[0].set_xlim(0.7, 2.5)
        self.axlist[0].set_ylim(-1.75, 1.75)
        self.axlist_2[0].set_xlim(-2.3, 2.3)
        self.axlist_2[0].set_ylim(-2.3, 2.3)
        if(Callee is not None):
            evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
            wx.PostEvent(Callee, evt_out)
        else:
            return self.fig, self.fig_2

    def E_field_plot(self, args):
        Callee = args[0]
        R = args[1]
        z = args[2]
        Psi = args[3]
        R_aus = args[4]
        z_aus = args[5]
        Psi_grid = args[6]
        E_field = args[7]
        self.setup_axes("single", r"Determined $R_\mathrm{aus}$ and $z_\mathrm{aus}$", "Toroidal electric field")
#        x_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(xl)), xl, k=1)
#        y_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(yl)), yl, k=1)
#        xl = x_spl(np.linspace(0.0, 1.0, 4000))
#        yl = y_spl(np.linspace(0.0, 1.0, 4000))
        CS = self.axlist[0].contour(R, z, Psi.T, \
             levels=Psi_grid)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R_aus, z_aus], \
                        marker="+", color=[1.0, 0, 0], \
                        y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[Psi_grid, E_field], \
                        marker="-", color=[0, 0, 1.0], \
                        y_range_in=self.y_range_list_2[0], ax_flag="E_field")
        if(AUG):
            fconf.plt_vessel(self.axlist[0])
        self.axlist[0].set_aspect("equal")
        self.axlist[0].set_xlim(0.7, 2.5)
        self.axlist[0].set_ylim(-1.75, 1.75)
        self.axlist_2[0].set_xlim(-0.05, 1.0)
        evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
        wx.PostEvent(Callee, evt_out)

    def R_diff_plot(self, args):
        Callee = args[0]
        Psi = args[1]
        Rdiff = args[2]
        psi_vol = args[3]
        vol = args[4]
        R0 = args[5]
        self.setup_axes("single", r"Radial diffusion profile", "")
#        x_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(xl)), xl, k=1)
#        y_spl = InterpolatedUnivariateSpline(np.linspace(0, 1.0, len(yl)), yl, k=1)
#        xl = x_spl(np.linspace(0.0, 1.0, 4000))
#        yl = y_spl(np.linspace(0.0, 1.0, 4000))
        vol_spline = InterpolatedUnivariateSpline(psi_vol, vol)
        norm_R = vol_spline(Psi) / (2.0 * np.pi * R0)
        norm_R[norm_R < 0] = 0
        norm_R = np.sqrt(norm_R)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[norm_R , Rdiff], \
                        marker="-", color=[1.0, 0, 0], \
                        y_range_in=self.y_range_list[0], ax_flag="Rdiff")
        evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
        wx.PostEvent(Callee, evt_out)


    def plot_ray_IDA_GUI(self, shot, time, ray, channel_no):
        self.setup_axes("ray", "Ray", "Hamiltonian")
        if(shot == "Analytical"):
            R = np.linspace(0.75, 2.3, 200)
            z = np.linspace(-1.75, 1.75, 200)
            rhop = np.zeros((200, 200))
            for i in range(200):
                rhop[i] = (3.e0 * np.sqrt((-1.5 + R[i]) ** 2 + z ** 2 / 10.e0)) / 2.e0
        else:
            eqi = eqi_map()
            coords, NPFM = eqi.norm_PFM(int(shot), float(time), np.linspace(0.1, 1.2, 12), diag="EQI")
            R = coords[0][0]
            z = coords[0][1]
            rhop = NPFM[0]
        CS = self.axlist[0].contour(R, z, rhop, \
             levels=np.linspace(0.1, 1.2, 13), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[0].contour(R, z, rhop, \
             levels=[1.0], linewidths=3, colors="b", linestyles="-")

        self.axlist[0].clabel(CS, fontsize=9, inline=1)
        self.axlist[0].clabel(CS2, fontsize=12, inline=1)
        fconf.plt_vessel(self.axlist[0])
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R, ray.z], \
                    name=r"ECRad ray", marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
        if(ray.R_tb is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R_tb, ray.z_tb], \
                    name=r"TORBEAM ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray.x, ray.y], \
                    name=r"ECRad ray", marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                         y_range_in=self.y_range_list[1], ax_flag="xy")
        if(ray.x_tb is not None):
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray.x_tb, ray.y_tb], \
                    name=r"TORBEAM ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                         y_range_in=self.y_range_list[1], ax_flag="xy")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray.s, ray.H], \
                    name=r"Hamiltonian $\times 10^5$", marker="-", color=(0, 0.e0, 0.0), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H", y_scale=1.e5)
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray.s, ray.N], \
                    name=r"Ray $N_{\omega, \mathrm{ray}}$ ", marker="--", color=(126.0 / 255, 0.e0, 0), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ray.s, ray.N_cold], \
                    name=r"Ray $N_{\omega, \mathrm{disp}}$ ", marker="-.", color=(0.e0, 0.e0, 26.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        if(AUG):
            fconf.plt_vessel(self.axlist[1], pol=False)
            fconf.plt_eq_tor(self.axlist[1], int(shot), float(time))
        self.create_legends("single")
        self.axlist[0].set_aspect("equal")
        self.axlist[1].set_aspect("equal")
        self.axlist[0].set_xlim(0.7, 2.5)
        self.axlist[0].set_ylim(-1.75, 1.75)
        self.axlist[1].set_ylim(-1.0, 1.0)
        self.axlist_2[0].set_ylim(0, 1)
        return self.fig, self.fig_2

    def tb_check_plot(self, tb_ray, ECRad_ray, R_mat, z_mat, rhop, N_tb, N_ECRad):
        self.setup_axes("single", "Ray", "rhop")
        CS = self.axlist[0].contour(R_mat, z_mat, rhop, \
             levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[0].contour(R_mat, z_mat, rhop, \
             levels=[1.0], linewidths=3, colors="b", linestyles="-")
        self.axlist[0].clabel(CS, fontsize=9, inline=1)
        self.axlist[0].clabel(CS2, fontsize=12, inline=1)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[np.sqrt((tb_ray.T[0] / 100.0) ** 2 + (tb_ray.T[1] / 100.0) ** 2), tb_ray.T[2] / 100.0], \
                    name=r"TORBEAM ray", marker="--", color=(126.0 / 255, 0.e0, 126.0 / 255), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[np.sqrt(ECRad_ray.T[0] ** 2 + ECRad_ray.T[1] ** 2), ECRad_ray.T[2]], \
                    name=r"ECRad ray", marker="--", color=(0.e0, 126.0 / 255, 126.0 / 255), \
                         y_range_in=self.y_range_list[0], ax_flag="Rz")
#        rhop_spl = RectBivariateSpline(R_mat, z_mat, rhop.T)
#        rhop_new = rhop_spl(np.sqrt((tb_ray.T[0] / 100.0) ** 2 + (tb_ray.T[1] / 100.0) ** 2), tb_ray.T[2] / 100.0, grid=False)
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt((tb_ray.T[0] / 100.0) ** 2 + (tb_ray.T[1] / 100.0) ** 2), rhop_new], \
#                    name=r"$\rho_\mathrm{pol}$ ECRad along TORBEAM ray", marker="-", color=(0.e0, 120.0 / 255, 0.e0), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt(ECRad_ray.T[0] ** 2 + ECRad_ray.T[1] ** 2), ECRad_ray.T[6]], \
#                    name=r"Ray $\rho_\mathrm{pol}$ ECRad", marker="*", color=(0.e0, 180.0 / 255, 180.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt((tb_ray.T[0] / 100.0) ** 2 + (tb_ray.T[1] / 100.0) ** 2), tb_ray.T[6]], \
#                    name=r"Ray $\rho_\mathrm{pol}$ TORBEAM", marker="+", color=(0.e0, 0.e0, 120.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        rhop_ECRad = rhop_spl(np.sqrt(ECRad_ray.T[0] ** 2 + ECRad_ray.T[1] ** 2), ECRad_ray.T[2], grid=False)
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt(ECRad_ray.T[0] ** 2 + ECRad_ray.T[1] ** 2), rhop_ECRad], \
#                    name=r"$\rho_\mathrm{pol}$ ECRad along ECRad ray", marker=":", color=(120.0 / 255, 120.0 / 255, 120.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt((tb_ray.T[0] / 100.0) ** 2 + (tb_ray.T[1] / 100.0) ** 2), \
#                                                                                         np.sqrt((tb_ray.T[3]) ** 2 + (tb_ray.T[4]) ** 2 + (tb_ray.T[5]) ** 2)], \
#                    name=r" $N_\mathrm{Ray}$ TORBEAM", marker="+", color=(0.e0, 0.e0, 120.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt((ECRad_ray.T[0]) ** 2 + (ECRad_ray.T[1]) ** 2), \
#                                                                                         np.sqrt((ECRad_ray.T[3]) ** 2 + (ECRad_ray.T[4]) ** 2 + (ECRad_ray.T[5]) ** 2)], \
#                    name=r" $N_\mathrm{Ray}$ ECRad", marker="*", color=(120.0 / 255, 0.e0, 0.e0), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt((ECRad_ray.T[0]) ** 2 + (ECRad_ray.T[1]) ** 2), \
#                                                                                         N_ECRad], \
#                    name=r" $N_\mathrm{Cold}$ ECRad", marker="-", color=(0.e0, 120.0 / 255, 120.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[svec.T[1], svec.T[3]], \
#                    name=r"$\rho_\mathrm{pol}$ svec", marker="-.", color=(200.0 / 255, 120.0 / 255, 0.e0, 120.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
#        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[np.sqrt((tb_ray.T[0] / 100.0) ** 2 + (tb_ray.T[1] / 100.0) ** 2), \
#                                                                                         N_tb], \
#                    name=r" $N_\mathrm{Cold}$ TB", marker="--", color=(0.e0, 120.0 / 255, 120.0 / 255), \
#                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ECRad_ray.T[6], \
                                                                                         ECRad_ray.T[7]], \
                    name=r" $X$ ECRad", marker="-", color=(0.e0, 120.0 / 255, 120.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ECRad_ray.T[6], \
                                                                                         ECRad_ray.T[8]], \
                    name=r" $Y$ ECRad", marker="-", color=(120.0 / 255, 0.e0, 120.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[tb_ray.T[6], \
                                                                                         tb_ray.T[7]], \
                    name=r" $X$ TB", marker="-", color=(0.e0, 0.0, 240.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[tb_ray.T[6], \
                                                                                         tb_ray.T[8]], \
                    name=r" $Y$ TB", marker="-", color=(240.0 / 255, 0.e0, 120.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[ECRad_ray.T[6], \
                                                                                         1.0 - np.sqrt(ECRad_ray.T[3] ** 2 + ECRad_ray.T[4] ** 2 + ECRad_ray.T[5] ** 2)], \
                    name=r" $1 - N$ ECRad", marker="-", color=(240.0 / 255, 0.e0, 120.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[tb_ray.T[6], \
                                                                                         1.0 - np.sqrt(tb_ray.T[3] ** 2 + tb_ray.T[4] ** 2 + tb_ray.T[5] ** 2)], \
                    name=r" $1 - N$ TB", marker="*", color=(0.e0, 240.0 / 255, 120.0 / 255), \
                         y_range_in=self.y_range_list_2[0], ax_flag="H")
        self.create_legends("single")
        self.axlist[0].set_xlim(0.7, 2.5)
        self.axlist[0].set_ylim(-1.75, 1.75)
        self.axlist_2[0].set_ylim(0, 1)
        return self.fig, self.fig_2

    def resonance_plain_plot(self, resonance, shot, time, dstf):
        data_name, simplfile, use_tertiary_model, tertiary_model, dist, dist_simpl, backup_name, backup_dist, tertiary_dist = get_files_and_labels(dstf)
        self.setup_axes("resonance", "$T_\mathrm{e}= $ \SI{" + "{0:2.2f}".format(resonance.Te * 1.e-3)\
                  + "}{\kilo\electronvolt}, $2\cdot f_\mathrm{c,0} = $ \SI{" + "{0:3.0f}".format(resonance.omega_c / np.pi * 1.e-9) + \
                  "}{\giga\hertz}")
        levels = np.linspace(-13, 5, 30)
        n_max = 3
        custom_range = True
        u_par_range = [-1.5, 1.5]
        u_perp_range = [0, 1.5]
        velocity_plot = False
        cmap = plt.cm.get_cmap("plasma")  # gnuplot
        cont1 = self.axlist[0].contourf(resonance.u_perp, resonance.u_par, resonance.log10_f , levels=levels, cmap=cmap)  # ,norm = LogNorm()
        cont2 = self.axlist[0].contour(resonance.u_perp, resonance.u_par, resonance.log10_f, levels=levels, colors='k',
                            hold='on', alpha=0.25, linewidths=1)
        for c in cont2.collections:
            c.set_linestyle('solid')
        cb = self.fig.colorbar(cont1, ax=self.axlist[0], ticks=[-10, -5, 0, 5])  # ticks = levels[::6] #,
        cb.set_label(r"$\mathrm{Log}_\mathrm{10}" + dist + r"$")  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
        # cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
        cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([1.0, 5.0, 10.0])))
        cb.ax.minorticks_on()
        marker = ["-", "--", "-.", ":"]
        dist_basic = "MJ"
        if(dstf == "Mx"):
            dist_basic = "M"
        for ibeam in range(len(resonance.beam_resonance_lines)):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], self.y_range_list[0], \
                                                                 data=[resonance.beam_resonance_lines[ibeam][1], \
                                                                       resonance.beam_resonance_lines[ibeam][0]], \
                                                                 marker=":", color=(0.1, 0.1, 0.1), ax_flag="dist")
        for n in range(len(resonance.resonance_lines)):
            if(resonance.harmonic[n] > n_max):
                continue
            label = r"ECE resonance - $f_\mathrm{ECE} \approx " + "  {0:3.1f}$".format(int(resonance.omega / 2.0 / np.pi * 1.e-9)) + \
                r"\si{\giga\hertz},\, $N_\parallel" + r" = $ {0:1.2f}, ".format(np.cos(resonance.theta) * resonance.N_abs) + \
                " {0:d}.".format(resonance.harmonic[n])
            label_f_res = r"$\log_{10}(f_\mathrm{" + dist + "})$" + " {0:d}.".format(resonance.harmonic[n])
            label_tau = r"$\tau_\mathrm{" + dist + "}$" + " {0:d}.".format(resonance.harmonic[n])
            label_f_therm_res = r"$\log_{10}(f_\mathrm{" + dist_basic + "})$" + " {0:d}.".format(resonance.harmonic[n])
            label_Trad = r"$T_\mathrm{rad\, ," + dist + "}$" + " {0:d}.".format(resonance.harmonic[n])
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], self.y_range_list[0], \
                                                                 data=[resonance.resonance_lines[n][1], resonance.resonance_lines[n][0]], \
                                                                 name=label, marker=marker[n % len(marker)], ax_flag="dist")
            self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], self.y_range_list_2[0], \
                                                                 data=[resonance.resonance_lines[n][0], resonance.Trad_nth_along_res[n]], \
                                                                 name=label_Trad, marker="-", ax_flag="Trad_momentum")  #  log_flag=True
            self.axlist_2[1], self.y_range_list_2[1] = self.add_plot(self.axlist_2[1], self.y_range_list_2[1], \
                                                                 data=[resonance.resonance_lines[n][0], resonance.tau_along_res[n]], \
                                                                 name=label_tau, marker="--", ax_flag="tau_momentum")
            self.axlist_2[2], self.y_range_list_2[2] = self.add_plot(self.axlist_2[2], self.y_range_list_2[2], \
                                                                 data=[resonance.resonance_lines[n][0], resonance.log10_f_along_res[n]], \
                                                                 name=label_f_res, marker=marker[n % len(marker)], ax_flag="dist_res")
            self.axlist_2[2], self.y_range_list_2[2] = self.add_plot(self.axlist_2[2], self.y_range_list_2[2], \
                                                                 data=[resonance.resonance_lines[n][0], resonance.log10_f_therm_along_res[n]], \
                                                                 name=label_f_therm_res, marker=marker[n % len(marker)], ax_flag="dist_res")
        for iray in range(len(resonance.ray_resonance_lines)):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], self.y_range_list[0], \
                                                                 data=[resonance.ray_resonance_lines[iray][1], \
                                                                       resonance.ray_resonance_lines[iray][0]], \
                                                                 marker="--", color=(0.1, 0.1, 0.1), ax_flag="dist")
        self.create_legends("resonance")
        if(custom_range):
            self.axlist[0].set_aspect("equal")
            self.axlist[0].set_ylim(u_par_range[0], u_par_range[1])
            self.axlist[0].get_xaxis().set_major_locator(MaxNLocator(nbins=3))
#            self.axlist[0].set_xlim(u_perp_range[0], u_perp_range[1])
        return self.fig, self.fig_2


    def setup_axes(self, mode, title, title2=None, shot=None, time=None):
        if(self.setup):
            return
        else:
            self.setup = True
        self.layout_2 = [0, 1, 1]
        single = False
        if(title2 is None):
            title2 = title
        if(mode == "Ich"):
            self.layout = [6, 3, 2]
            self.grid_locations = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
            self.twinx_array = [True, True, True, True, False, False]
            self.twinx_y_share = [None, 1, None, 5, None, None]
            self.x_share_list = [None, 0, 0, 0, 0, 0]
            self.y_share_list = [None, 0, None, 4, None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [True]
            self.y_range_list_2 = []
            self.y_share_list_2
            steps = np.array([0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "BPD"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [True]
            self.twinx_y_share = [None, None]
            self.x_share_list = [None]
            self.y_range_list = []
            self.y_share_list = [None, None]
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False, False]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None, None]  # Note the twinx axes!
            steps = np.array([0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "Ich_compare"):
            self.layout = [4, 2, 2]
            self.grid_locations = [[0, 0], [0, 1], [1, 0], [1, 1]]
            self.twinx_array = [True, True, True, True]
            self.twinx_y_share = [None, 1, None, 5]
            self.x_share_list = [None, 0, 0, 0]
            self.y_share_list = [None, 0, None, 4]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [True]
            self.y_range_list_2 = []
            steps = np.array([0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0])
            steps_2 = np.array([0.5, 1.0, 2.0, 3.0])
            steps_y = steps
            steps_2_y = steps_y
            self.y_share_list_2
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "Ich_BD"):
            self.layout = [2, 2, 1]
            self.grid_locations = [[0, 0], [1, 0]]
            self.twinx_array = [True, True]
            self.twinx_y_share = [None, None]
            self.x_share_list = [None, 0, 0, 0]
            self.y_share_list = [None, None, None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [True]
            self.y_range_list_2 = []
            steps = np.array([0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0])
            steps_2 = np.array([0.5, 1.0, 2.0, 3.0])
            steps_y = steps
            steps_2_y = steps_y
            self.y_share_list_2
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "Te" or mode == "Te_twinx"):
            self.layout = [2, 2, 1]
            self.grid_locations = [[0, 0], [1, 0]]
            if('twinx' in mode):
                self.twinx_array = [True, False]
                self.y_share_list = [None, None, None]  # Note the twinx axes!
                self.twinx_y_share = [None]
            else:
                self.twinx_array = [False, False]
                self.y_share_list = [None, None]  # Note the twinx axes!
            self.x_share_list = [None, 0]
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [True]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None, None]  # Note the twinx axes!
            self.y_range_list_2 = []
            self.x_step_min = [3, 3]
            self.x_step_max = [7, 7]
            self.y_step_min = [3, 2]
            self.y_step_max = [6, 3]
            self.x_step_min_2 = [3, 3]
            self.x_step_max_2 = [7, 7]
            self.y_step_min_2 = [3, 3]
            self.y_step_max_2 = [6, 6]
            steps = np.array([1.0, 5.0, 10.0])
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2], height_ratios=[3, 1])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "double"):
            self.layout = [2, 2, 1]
            self.grid_locations = [[0, 0], [1, 0]]
            if('twinx' in mode):
                self.twinx_array = [True, False]
                self.y_share_list = [None, None, None]  # Note the twinx axes!
                self.twinx_y_share = [None]
            else:
                self.twinx_array = [False, False]
                self.y_share_list = [None, None]  # Note the twinx axes!
            self.x_share_list = [None, 0]
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None, None]  # Note the twinx axes!
            self.y_range_list_2 = []
            self.x_step_min = [3, 3]
            self.x_step_max = [7, 7]
            self.y_step_min = [3, 2]
            self.y_step_max = [6, 3]
            self.x_step_min_2 = [3, 3]
            self.x_step_max_2 = [7, 7]
            self.y_step_min_2 = [3, 3]
            self.y_step_max_2 = [6, 6]
            steps = np.array([1.0, 5.0, 10.0])
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2], height_ratios=[3, 1])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "resonance"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [False]
            self.x_share_list = [None]
            self.y_share_list = [None]  # Note the twinx axes!
            self.twinx_y_share = [None]
            self.y_range_list = []
            self.layout_2 = [2, 1, 2]
            self.grid_locations_2 = [[0, 0], [0, 1]]
            self.twinx_array_2 = [True, False]
            self.x_share_list_2 = [None, None]
            self.y_share_list_2 = [None, None, None]  # Note the twinx axes!
            self.twinx_y_share_2 = [None, None, None]
            self.y_range_list_2 = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "ray"):
            self.layout = [2, 1, 2]
            self.grid_locations = [[0, 0], [0, 1]]
            self.twinx_array = [False, False]
            self.x_share_list = [None, None]
            self.y_share_list = [None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_list_2 = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "ray_2_fig"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [False]
            self.x_share_list = [None]
            self.y_share_list = [None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_list_2 = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode.startswith("Te_no_ne")):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            if('twinx' in mode):
                self.twinx_array = [True]
                self.y_share_list = [None, None]  # Note the twinx axes!
                self.twinx_y_share = [None]
            else:
                self.twinx_array = [False]
                self.y_share_list = [None]  # Note the twinx axes!
            self.x_share_list = [None]
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [True]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None, None]  # Note the twinx axes!
            self.y_range_list_2 = []
            steps = np.array([1.0, 2.0, 2.5, 5.0, 7.5, 10.0])
            steps_y = None
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "stacked"):
            self.layout = [4, 4, 1]
            self.grid_locations = [[0, 0], [1, 0], [2, 0], [3, 0]]
            self.twinx_array = [False, False, False, False]
            self.x_share_list = [None, 0, 0, 0]
            self.y_share_list = [None, None, None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "stacked_small"):
            self.layout = [3, 3, 1]
            self.grid_locations = [[0, 0], [1, 0], [2, 0]]
            self.twinx_array = [False, False, False]
            self.x_share_list = [None, 0, 0]
            self.y_share_list = [None, None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "stacked_large"):
            self.layout = [4, 4, 1]
            self.grid_locations = [[0, 0], [1, 0], [2, 0], [3, 0]]
            self.twinx_array = [False, False, False, False]
            self.x_share_list = [None, 0, 0, 0]
            self.y_share_list = [None, None, None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "stacked_very_small"):
            self.layout = [2, 2, 1]
            self.grid_locations = [[0, 0], [1, 0]]
            self.twinx_array = [False, False]
            self.x_share_list = [None, 0]
            self.y_share_list = [None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "stacked_2_twinx"):
            self.layout = [4, 4, 1]
            self.grid_locations = [[0, 0], [1, 0], [2, 0], [3, 0]]
            self.twinx_array = [False, True, False, True]
            self.x_share_list = [None, 0, 0, 0, 0, 0]
            self.y_share_list = [None, None, None, None, None, None]  # Note the twinx axes!
            self.twinx_y_share = [None, None, None, None]
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "stacked_1_twinx"):
            self.layout = [3, 3, 1]
            self.grid_locations = [[0, 0], [1, 0], [2, 0]]
            self.twinx_array = [False, True, False]
            self.x_share_list = [None, 0, 0, 0, 0, 0]
            self.y_share_list = [None, None, None, None]  # Note the twinx axes!
            self.twinx_y_share = [None, None, None]
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "single"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [False, False]
            self.x_share_list = [None, 0]
            self.y_share_list = [None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False, False]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None, None]
            self.y_range_listv = []
            steps = np.array([1.0, 2.0, 4.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
            single = True
        elif(mode == "twinx"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [True]
            self.x_share_list = [None]
            self.twinx_y_share = [None]
            self.y_share_list = [None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 2.0, 4.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
            single = True
        elif(mode == "freq"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [False]
            self.x_share_list = [None, 0]
            self.y_share_list = [None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0])
            steps_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "Te_spec"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [False]
            self.x_share_list = [None, 0]
            self.y_share_list = [None]  # Note the twinx axes!
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_listv = []
            steps = np.array([1.0, 2.5, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "double"):
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            self.twinx_array = [True]
            self.x_share_list = [None, 0]
            self.y_share_list = [None, None]  # Note the twinx axes!
            self.y_range_list = []
            self.twinx_y_share = [None, None]
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [True]
            self.x_share_list_2 = [None, 0]
            self.y_share_list_2 = [None, None]  # Note the twinx axes!
            self.twinx_y_share_2 = [None, None]
            self.y_range_listv = []
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "QL_calc"):
            self.layout = [3, 3, 1]
            self.grid_locations = [[0, 0], [1, 0], [2, 0]]
            self.twinx_array = [False, False, True]
            self.x_share_list = [None, 0, 0, 0, 0]
            self.y_share_list = [None, None, None, None]  # Note the twinx axes!
            self.twinx_y_share = [None, None, None, None]
            self.y_range_list = []
            self.layout_2 = [2, 1, 2]
            self.grid_locations_2 = [[0, 0], [0, 1]]
            self.twinx_array_2 = [False, False]
            self.x_share_list_2 = [None, None]
            self.y_share_list_2 = [None, None]  # Note the twinx axes!
            self.y_range_list_2 = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        elif(mode == "overlap_plot"):
            self.layout = [2, 1, 2]
            self.grid_locations = [[0, 0], [0, 1]]
            self.twinx_array = [True, False]
            self.x_share_list = [None, None]
            self.y_share_list = [None, None, None]
            self.twinx_y_share = [None, None, None]
            self.y_range_list = []
            self.layout_2 = [1, 1, 1]
            self.grid_locations_2 = [[0, 0]]
            self.twinx_array_2 = [False]
            self.x_share_list_2 = [None]
            self.y_share_list_2 = [None]  # Note the twinx axes!
            self.y_range_list_2 = []
            steps = np.array([1.0, 5.0, 10.0])
            steps_y = steps
            steps_2 = steps
            steps_2_y = steps_y
            self.gridspec = plt.GridSpec(self.layout[1], self.layout[2])
            self.gridspec_2 = plt.GridSpec(self.layout_2[1], self.layout_2[2])
        else:
            self.layout = [1, 1, 1]
            self.grid_locations = [[0, 0]]
            single = True
            self.twinx_array = [False]
        self.model_marker_index = np.zeros(self.layout[0], dtype=np.int)
        self.model_color_index = np.zeros(self.layout[0], dtype=np.int)
        self.diag_marker_index = np.zeros(self.layout[0], dtype=np.int)
        self.diag_color_index = np.zeros(self.layout[0], dtype=np.int)
        self.line_marker_index = np.zeros(self.layout[0], dtype=np.int)
        self.line_color_index = np.zeros(self.layout[0], dtype=np.int)
        self.model_marker_index_2 = np.zeros(self.layout_2[0], dtype=np.int)
        self.model_color_index_2 = np.zeros(self.layout_2[0], dtype=np.int)
        self.diag_marker_index_2 = np.zeros(self.layout_2[0], dtype=np.int)
        self.diag_color_index_2 = np.zeros(self.layout_2[0], dtype=np.int)
        self.line_marker_index_2 = np.zeros(self.layout_2[0], dtype=np.int)
        self.line_color_index_2 = np.zeros(self.layout_2[0], dtype=np.int)
        for i in range(self.layout[0]):
            grid_loc = self.gridspec[self.grid_locations[i][0], self.grid_locations[i][1]]
            if(self.x_share_list[i] is None and \
                self.y_share_list[i] is None):
                self.axlist.append(self.fig.add_subplot(grid_loc))
            elif(self.x_share_list[i] is not None and \
                    self.y_share_list[i] is None):
                self.axlist.append(self.fig.add_subplot(grid_loc, \
                    sharex=self.axlist[self.x_share_list[i]]))
            elif(self.x_share_list[i] is None and \
                    self.y_share_list[i] is not None):
                self.axlist.append(self.fig.add_subplot(grid_loc, \
                    sharey=self.axlist[self.y_share_list[i]]))
            else:
                self.axlist.append(self.fig.add_subplot(grid_loc, \
                    sharex=self.axlist[self.x_share_list[i]], \
                    sharey=self.axlist[self.y_share_list[i]]))
            self.y_range_list.append([np.inf, -np.inf])
            if(self.twinx_array[i] == True):
                self.axlist.append(self.axlist[-1].twinx())
                self.y_range_list.append([np.inf, -np.inf])
                if(self.twinx_y_share[i] is not None):
                    self.axlist[self.twinx_y_share[i]].get_shared_y_axes().join(self.axlist[self.twinx_y_share[i]], \
                                                                                self.axlist[-1])
            else:
                pass
        if("Ich" not in mode):
            ratio_x = default_x1 / self.fig.get_size_inches()[0]
            ratio_y = default_y1 / self.fig.get_size_inches()[1]
        else:
            ratio_x = default_x2 / self.fig.get_size_inches()[0]
            ratio_y = default_y2 / self.fig.get_size_inches()[1]
        # print(ratio_x ,ratio_y)
        left_margin = 0.1
        if("Ich" in mode and mode != "Ich_BD"):
            if(self.title):
                self.gridspec.tight_layout(self.fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.90 - 0.05 * (1.0 - 1.0 / ratio_y)])
                left_margin = 0.075 * ratio_x
            else:
                self.gridspec.tight_layout(self.fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
                left_margin = 0.075 * ratio_x
        elif(mode == "double" or mode == "ray"):
            self.gridspec.tight_layout(self.fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
            left_margin = 0.075 * ratio_x
        elif(mode == "stacked_2_twinx"):
            self.gridspec.tight_layout(self.fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
            left_margin = 0.075 * ratio_x
        elif(mode == "overlap_plot"):
            ratio_x = default_x1 / self.fig.get_size_inches()[0]
            ratio_y = default_y1 / self.fig.get_size_inches()[1]
            # print(ratio_x ,ratio_y)
            self.gridspec.tight_layout(self.fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
            left_margin = 0.075 * ratio_x
        else:
            ratio_x = default_x1 / self.fig.get_size_inches()[0]
            ratio_y = default_y1 / self.fig.get_size_inches()[1]
            # print(ratio_x ,ratio_y)
            self.gridspec.tight_layout(self.fig, pad=0.0, h_pad=1.0, w_pad=1.0,
                         rect=[0.05 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
            left_margin = 0.05 * ratio_x
        if(self.title):
            self.fig.suptitle(title)
        elif(shot is not None):
            shotstr = r"\# {0:d}".format(shot)
            if(time is not None):
                shotstr += " $t$ = \SI{" + "{0:1.3f}".format(time) + r"}{\second}"
            self.axlist[0].text(left_margin, 0.05, shotstr,
                verticalalignment='bottom', horizontalalignment='left',
                transform=self.axlist[0].transAxes,
                color='black', fontsize=plt.rcParams['axes.titlesize'])
        if(self.fig_2 is None):
            self.axlist_2 = []
            return
        for i in range(self.layout_2[0]):
            grid_loc = self.gridspec_2[self.grid_locations_2[i][0], self.grid_locations_2[i][1]]
            if(self.x_share_list_2[i] is None and \
                self.y_share_list_2[i] is None):
                self.axlist_2.append(self.fig_2.add_subplot(grid_loc))
            elif(self.x_share_list_2[i] is not None and \
                    self.y_share_list_2[i] is None):
                self.axlist_2.append(self.fig_2.add_subplot(grid_loc, \
                    sharex=self.axlist_2[self.x_share_list_2[i]]))
            elif(self.x_share_list_2[i] is None and \
                    self.y_share_list_2[i] is not None):
                self.axlist_2.append(self.fig_2.add_subplot(grid_loc, \
                    sharey=self.axlist_2[self.y_share_list_2[i]]))
            else:
                self.axlist_2.append(self.fig_2.add_subplot(grid_loc, \
                    sharex=self.axlist_2[self.x_share_list_2[i]], \
                    sharey=self.axlist_2[self.y_share_list_2[i]]))
            self.y_range_list_2.append([np.inf, -np.inf])
            if(self.twinx_array_2[i] == True):
                self.axlist_2.append(self.axlist_2[-1].twinx())
                self.y_range_list_2.append([np.inf, -np.inf])
            else:
                pass
        if("Ich" not in mode):
            ratio_x = default_x1 / self.fig_2.get_size_inches()[0]
            ratio_y = default_y1 / self.fig_2.get_size_inches()[1]
        else:
            ratio_x = default_x2 / self.fig_2.get_size_inches()[0]
            ratio_y = default_y2 / self.fig_2.get_size_inches()[1]
        # print(ratio_x ,ratio_y)
        if(self.title):
            if(title2 is not None):
                self.fig_2.suptitle(title2)
            else:
                self.fig_2.suptitle(title)
        if(mode == "stacked_2_twinx"):
            self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                         rect=[0.05 * ratio_x, 0.075 * ratio_y, 0.95 - 0.05 * (1.0 - 1.0 / ratio_x), 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
        elif(mode == "double"):
            self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                         rect=[0.05 * ratio_x, 0.1 * ratio_y, 0.95 - 0.05 * (1.0 - 1.0 / ratio_x), 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
        elif(mode == "resonance"):
            ratio_x = default_x1 / self.fig.get_size_inches()[0]
            ratio_y = default_y1 / self.fig.get_size_inches()[1]
            # print(ratio_x ,ratio_y)
            self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                             rect=[0.075, 0.075 * ratio_y, 0.95 - 0.05 * (1.0 - 1.0 / ratio_x), 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
        elif(mode == "QL_calc"):
            ratio_x = default_x1 / self.fig.get_size_inches()[0]
            ratio_y = default_y1 / self.fig.get_size_inches()[1]
            # print(ratio_x ,ratio_y)
            self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                             rect=[0.075, 0.075 * ratio_y, 0.95 - 0.05 * (1.0 - 1.0 / ratio_x), 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
        elif(mode == "ray"):
            ratio_x = default_x1 / self.fig.get_size_inches()[0]
            ratio_y = default_y1 / self.fig.get_size_inches()[1]
            self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
        else:
            ratio_x = default_x1 / self.fig_2.get_size_inches()[0]
            ratio_y = default_y1 / self.fig_2.get_size_inches()[1]
            # print(ratio_x ,ratio_y)
            if(single):
                self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95 , 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
            else:
                self.gridspec_2.tight_layout(self.fig_2, pad=0.0, h_pad=0.0, w_pad=1.0,
                         rect=[0.075 * ratio_x, 0.075 * ratio_y, 0.95, 0.95 - 0.05 * (1.0 - 1.0 / ratio_y)])
#        self.gridlist_2.append(gridspec.GridSpec(self.layout_2[1],self.layout_2[2]))
#        for i in range(self.layout_2[0]):
#            k = i%2
#            if(self.layout_2[2] == 1):
#                k = 0
#            if(self.layout_2[2] == 1):
#                grid_loc = self.gridlist_2[k][i/self.layout[2]]
#            elif(self.layout_2[2] == 2):
#                grid_loc = self.gridlist_2[k][i/self.layout[2]]
#            if(self.x_share_list_2[i] is None and \
#                self.y_share_list_2[i] is None):
#                self.axlist_2.append(self.fig2.add_subplot(grid_loc))
#            elif(self.x_share_list_2[i] is not None and \
#                    self.y_share_list_2[i] is None):
#                self.axlist_2.append(self.fig2.add_subplot(grid_loc,\
#                    sharex = self.axlist_2[self.x_share_list_2[i]]))
#            elif(self.x_share_list_2[i] is None and \
#                    self.y_share_list_2[i] is not None):
#                self.axlist_2.append(self.fig2.add_subplot(grid_loc,\
#                    sharey = self.axlist_2[self.y_share_list_2[i]]))
#            else:
#                self.axlist_2.append(self.fig2.add_subplot(grid_loc,\
#                    sharex = self.axlist_2[self.x_share_list_2[i]], \
#                    sharey = self.axlist_2[self.y_share_list_2[i]]))
#            self.y_range_list_2.append([np.inf, -np.inf])
#            self.axlist_2[-1].get_xaxis().set_major_locator(NLocator(prune='lower'))
#            #print(int(i/2),i%2)
#            if(self.twinx_array_2[i] == True):
#                #print("Twin")
#                self.axlist_2[-1].get_xaxis().set_major_locator(NLocator(nbins = 4, steps=steps_2))
#                self.axlist_2[-1].get_xaxis().set_minor_locator(NLocator(nbins = 8, steps=steps_2/2.0))
#                self.axlist_2[-1].get_yaxis().set_major_locator(NLocator(nbins = 4, steps=steps_2_y))
#                self.axlist_2[-1].get_yaxis().set_minor_locator(NLocator(nbins = 8, steps=steps_2_y/2.0))
#                self.axlist_2.append(self.axlist_2[-1].twinx())
#                self.axlist_2[-1].get_yaxis().set_major_locator(NLocator(nbins = 4, steps=steps_2_y))
#                self.axlist_2[-1].get_yaxis().set_minor_locator(NLocator(nbins = 8, steps=steps_2_y/2.0))
#
#                self.y_range_list_2.append([np.inf, -np.inf])
#                if(self.twinx_y_share[i] is not None):
#                    self.axlist[self.twinx_y_share[i]].get_shared_y_axes().join(self.axlist[self.twinx_y_share[i]],\
#                                                                                    self.axlist[-1])
#            else:
#                self.axlist_2[-1].get_xaxis().set_minor_locator(NLocator(nbins = 8, steps=steps_2/2.0))
#                if(steps_y is None):
#                    self.axlist_2[-1].get_yaxis().set_major_locator(NLocator(nbins = 4))
#                    self.axlist_2[-1].get_yaxis().set_minor_locator(NLocator(nbins = 8))
#                else:
#                    self.axlist_2[-1].get_yaxis().set_major_locator(NLocator(nbins = 4, steps=steps_y))
#                    self.axlist_2[-1].get_yaxis().set_minor_locator(NLocator(nbins = 8, steps=steps_y/2.0))
#        #print(len(self.axlist_2))
#        self.fig2.suptitle(title2)
#        if("Ich" not in mode):
#            ratio_x =   default_x2 / self.fig2.get_size_inches()[0]
#            ratio_y =   default_y2 /self.fig2.get_size_inches()[1]
#            if(single):
#                self.gridlist_2[0].tight_layout(self.fig2, pad=0.0, h_pad=0.0, w_pad=1.0,
#                         rect=[0.1,0.075 * ratio_y, 0.95 - 0.05 * (1 - 1 /ratio_x), 0.90])
#            else:
#                self.gridlist_2[0].tight_layout(self.fig2, pad=0.0, h_pad=0.0, w_pad=1.0,
#                         rect=[0.1,0.05* ratio_y, 0.95- 0.05 * (1 - 1 /ratio_x), 0.90])
#        else:
#            ratio_x =   default_x1 / self.fig2.get_size_inches()[0]
#            ratio_y =   default_y1 /self.fig2.get_size_inches()[1]
#        ratio_x =   default_x2 / self.fig2.get_size_inches()[0]
#        ratio_y =   default_y2 /self.fig2.get_size_inches()[1]
#        self.gridlist_2[0].tight_layout(self.fig2, pad=0.0, h_pad=3.0, w_pad=5.0,
#                        rect=[0.15,0.05 *ratio_y, 0.95- 0.05 * (1 - 1 /ratio_x), 0.9])
#        if(len(self.gridlist_2) > 1):
#            self.gridlist_2[1].tight_layout(self.fig2, pad=0.0, h_pad=3.0, w_pad=5.0,
#                         rect=[0.15,0.05 *ratio_y, 0.95- 0.05 * (1 - 1 /ratio_x), 0.9])
#
#        for i in range(len(self.axlist)):
#            self.axlist[i].ticklabel_format(style='sci', axis='y', scilimits=(-2,3))
#        for i in range(len(self.axlist_2)):
#            self.axlist_2[i].ticklabel_format(style='sci', axis='y', scilimits=(-2,3))

    def double_check_ylimit(self, mode):
        if(mode.startswith("ich")):
            self.axlist[0].set_y


    def create_legends(self, mode):
        if(mode.startswith("Ich")):
            lns = self.axlist[0].get_lines() + self.axlist[1].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[0].legend(lns, labs)
            leg.get_frame().set_alpha(0.5)
            leg.draggable()
            lns = self.axlist[2].get_lines() + self.axlist[3].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[2].legend(lns, labs)
            leg.get_frame().set_alpha(0.5)
            leg.draggable()
            labs = []
            lns_short = []
            for i in range(len(self.axlist_2)):
                print("i leg", i)
                lns = self.axlist_2[i].get_lines()
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        labs.append(lns[l].get_label())
                        lns_short.append(lns[l])
            leg = self.axlist_2[1].legend(lns_short, labs)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
            if(mode != "Ich_BD"):
                lns = self.axlist[4].get_lines() + self.axlist[5].get_lines()
                labs = [l.get_label() for l in lns]
                leg = self.axlist[5].legend(lns, labs)
                leg.get_frame().set_alpha(0.5)
                leg.draggable()
                lns = self.axlist[6].get_lines() + self.axlist[7].get_lines()
                labs = [l.get_label() for l in lns]
                leg = self.axlist[7].legend(lns, labs)
                leg.get_frame().set_alpha(0.5)
                leg.draggable()
                if(mode != "Ich_compare"):
                    lns = self.axlist[8].get_lines()
                    labs = [l.get_label() for l in lns]
                    leg = self.axlist[8].legend(lns, labs)
                    leg.get_frame().set_alpha(0.5)
                    leg.draggable()
                    lns = self.axlist[9].get_lines()
                    labs = [l.get_label() for l in lns]
                    leg = self.axlist[9].legend(lns, labs)
                    leg.get_frame().set_alpha(0.5)
                    leg.draggable()
            else:
                labs = []
                lns_short = []
                for i in range(len(self.axlist_2)):
                    lns = self.axlist_2[i].get_lines()
                    for l in range(len(lns)):
                        if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                            labs.append(lns[l].get_label())
                            lns_short.append(lns[l])
                leg2 = self.axlist_2[1].legend(lns_short, labs)
                leg2.get_frame().set_alpha(0.5)
                leg2.draggable()
        elif(mode == "BDP"):
            labs = []
            lns_short = []
            for i in range(len(self.axlist)):
                lns = self.axlist[i].get_lines()
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        labs.append(lns[l].get_label())
                        lns_short.append(lns[l])
            leg = self.axlist[1].legend(lns_short, labs, loc="best")
            leg.get_frame().set_alpha(0.5)
            leg.draggable()
        elif(mode == "BPD_twix"):
            for i in range(0, 3, 2):
                labs = []
                lns_short = []
                lns = self.axlist[i].get_lines() + self.axlist[i + 1].get_lines()
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        labs.append(lns[l].get_label())
                        lns_short.append(lns[l])
                leg = self.axlist[i + 1].legend(lns_short, labs, loc="best")
                leg.draggable()
                leg.get_frame().set_alpha(0.5)
        elif(mode == "double"):
            lns = self.axlist[0].get_lines() + self.axlist[1].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[0].legend(lns, labs)
            leg.get_frame().set_alpha(0.5)
            leg.draggable()
            if(len(self.axlist_2) > 0):
                lns = self.axlist_2[0].get_lines() + self.axlist_2[1].get_lines()
                labs = [l.get_label() for l in lns]
                leg2 = self.axlist_2[0].legend(lns, labs)
                leg2.get_frame().set_alpha(0.5)
                leg2.draggable()
        elif(mode == "Te" or mode == "Te_no_ne"):
            for i in range(len(self.axlist)):
                leg = self.axlist[i].legend(loc="best")
                if(leg is not None):
                    leg.draggable()
            if(len(self.axlist_2) > 0):
                try:
                    lns = self.axlist_2[0].get_lines() + self.axlist_2[1].get_lines()
                    labs = []
                    lns_filtered = []
                    for l in lns:
                        if(not (l.get_label().startswith("_") or l.get_label().startswith("\\"))):
                            labs.append(l.get_label())
                            lns_filtered.append(l)
                    leg2 = self.axlist_2[0].legend(lns_filtered, labs)
                    if(leg2 is not None):
                        leg2.get_frame().set_alpha(1)
                        leg2.draggable()
                except IndexError:
                    leg2 = self.axlist_2[0].legend()
                    if(leg2 is not None):
                        leg2.get_frame().set_alpha(1)
                        leg2.draggable()
        elif(mode == "Te_twinx" or mode == "Te_no_ne_twinx"):
            lines, labels = self.axlist[0].get_legend_handles_labels()
            lines2, labels2 = self.axlist[1].get_legend_handles_labels()
            leg = self.axlist[1].legend(lines + lines2, labels + labels2, loc=0)
            leg.draggable()
            if(mode == "Te_twinx"):
                leg = self.axlist[2].legend(loc="best")
                if(leg is not None):
                    leg.draggable()
            if(len(self.axlist_2) > 0):
                try:
                    lns = self.axlist_2[0].get_lines() + self.axlist_2[1].get_lines()
                    labs = []
                    lns_filtered = []
                    for l in lns:
                        if(not (l.get_label().startswith("_") or l.get_label().startswith("\\"))):
                            labs.append(l.get_label())
                            lns_filtered.append(l)
                    leg2 = self.axlist_2[0].legend(lns_filtered, labs)
                    if(leg2 is not None):
                        leg2.get_frame().set_alpha(1)
                        leg2.draggable()
                except IndexError:
                    leg2 = self.axlist_2[0].legend()
                    if(leg2 is not None):
                        leg2.get_frame().set_alpha(1)
                        leg2.draggable()
        elif(mode == "stacked_2_twinx"):
            lns = self.axlist[0].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[0].legend(lns, labs)
            leg.get_frame().set_alpha(1)
            leg.draggable()
            lns = self.axlist[1].get_lines() + self.axlist[2].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[2].legend(lns, labs)
            leg.get_frame().set_alpha(1)
            leg.draggable()
            lns = self.axlist[3].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[3].legend(lns, labs)
            leg.get_frame().set_alpha(1)
            leg.draggable()
            lns = self.axlist[4].get_lines() + self.axlist[5].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[5].legend(lns, labs)
            # self.axlist[5].add_artist(leg)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
        elif(mode == "stacked_1_twinx"):
            lns = self.axlist[0].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[0].legend(lns, labs)
            leg.draggable()
            leg.get_frame().set_alpha(1)
            lns = self.axlist[1].get_lines() + self.axlist[2].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[2].legend(lns, labs)
            leg.get_frame().set_alpha(1)
            leg.draggable()
            lns = self.axlist[3].get_lines()
            labs = [l.get_label() for l in lns]
            leg = self.axlist[3].legend(lns, labs)
            leg.draggable()
            # self.axlist[5].add_artist(leg)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
        elif(mode == "vessel"):
            for i in range(len(self.axlist)):
                lns = self.axlist[i].get_lines()
                labs = []
                lns_short = []
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        try:
                            labs.append(lns[l].get_label())
                            lns_short.append(lns[l])
                        except ValueError:
                            print("A label was skipped, because of wrong dimensions")
                leg = self.axlist[i].legend(lns_short, labs)
                leg.draggable()
            leg.get_frame().set_alpha(0.5)
            for i in range(len(self.axlist_2)):
                lns = self.axlist_2[i].get_lines()
                labs = []
                lns_short = []
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        labs.append(lns[l].get_label())
                        lns_short.append(lns[l])
                leg = self.axlist_2[i].legend(lns_short, labs)
                leg.get_frame().set_alpha(0.5)
                leg.draggable()
        elif(mode == "errorbar"):
            for i in range(len(self.axlist)):
                self.axlist[i].legend(loc="best")
            for i in range(len(self.axlist_2)):
                self.axlist_2[i].legend(loc="best")
        elif(mode == "errorbar_twinx"):
            handles_primary, labels_primary = self.axlist[0].get_legend_handles_labels()
            handles_twinx, labels_twinx = self.axlist[1].get_legend_handles_labels()
            handles = handles_primary + handles_twinx
            labels = labels_primary + labels_twinx
            leg = self.axlist[0].legend(handles, labels)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
#            lns = self.axlist[0].get_lines()
#            lns = np.concatenate([lns, self.axlist[1].get_lines()])
#            labs = []
#            lns_short = []
#            for l in range(len(lns)):
#                if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
#                    labs.append(lns[l].get_label())
#                    lns_short.append(lns[l])
#            leg = self.axlist[0].legend(lns_short, labs)
        elif(mode == "only_axis_1"):
            for i in range(len(self.axlist) - 1):
                lns = self.axlist[i].get_lines()
                if(i == len(self.axlist) - 2):
                    lns = np.concatenate([lns, self.axlist[i + 1].get_lines()])
                labs = []
                lns_short = []
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        labs.append(lns[l].get_label())
                        lns_short.append(lns[l])
                    # else:
                    #    print("Label " + lns[l].get_label() + " ignored")
                leg = self.axlist[i].legend(lns_short, labs)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
        elif(mode == "resonance"):
            i = 0
            lns = self.axlist[i].get_lines()
            labs = []
            lns_short = []
            for l in range(len(lns)):
                if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                    labs.append(lns[l].get_label())
                    lns_short.append(lns[l])

            lns = self.axlist_2[0].get_lines() + self.axlist_2[1].get_lines()
            labs = []
            lns_short = []
            leg = None
            for l in range(len(lns)):
                if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                    labs.append(lns[l].get_label())
                    lns_short.append(lns[l])

                    # else:
                    #    print("Label " + lns[l].get_label() + " ignored")
                leg = self.axlist_2[0].legend(lns_short, labs)
            if(leg is not None):
                leg.get_frame().set_alpha(1.0)
                leg.draggable()
            i = 2
            lns = self.axlist_2[i].get_lines()
            labs = []
            lns_short = []
            for l in range(len(lns)):
                if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                    labs.append(lns[l].get_label())
                    lns_short.append(lns[l])
                # else:
                #    print("Label " + lns[l].get_label() + " ignored")
            leg = self.axlist_2[i].legend(lns_short, labs)
            leg.draggable()
        else:
            for i in range(len(self.axlist)):
                lns = self.axlist[i].get_lines()
                labs = []
                lns_short = []
                for l in range(len(lns)):
                    if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                        labs.append(lns[l].get_label())
                        lns_short.append(lns[l])
                    # else:
                    #    print("Label " + lns[l].get_label() + " ignored")
                leg = self.axlist[i].legend(lns_short, labs)
                leg.draggable()
                leg.get_frame().set_alpha(1.0)
            if(len(self.axlist_2) > 9):
                for i in range(len(self.axlist_2)):
                    lns = self.axlist_2[i].get_lines()
                    labs = []
                    lns_short = []
                    for l in range(len(lns)):
                        if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
                            labs.append(lns[l].get_label())
                            lns_short.append(lns[l])
                        # else:
                        #    print("Label " + lns[l].get_label() + " ignored")
                    leg = self.axlist_2[i].legend(lns_short, labs)
                    leg.get_frame().set_alpha(0.5)
                    leg.draggable()
        # self.finishing_touches()

    # Once all information about a plot is known apply some finishing touches
    def finishing_touches(self):
        steps = np.array([ 1.0, 0.5, 0.1, 2.0, 0.2, 0.25, 2.5, 0.3, 0.4, 0.6, 0.8])
        for i in range(len(self.axlist + self.axlist_2)):
            ax = (self.axlist + self.axlist_2)[i]
            plotxrange = ax.get_xlim()
            plotyrange = ax.get_ylim()
            if(np.log10(plotxrange[1] - plotxrange[0]) < 0):
                x_scale = 10.0 ** (int(np.log10(plotxrange[1] - plotxrange[0]) - 1.0))
            else:
                x_scale = 10.0 ** (int(np.log10(plotxrange[1] - plotxrange[0])))
            if(np.log10(plotyrange[1] - plotyrange[0]) < 0):
                y_scale = 10.0 ** (int(np.log10(plotyrange[1] - plotyrange[0]) - 1.0))
            else:
                y_scale = 10.0 ** (int(np.log10(plotyrange[1] - plotyrange[0])))
            x_length = (plotxrange[1] - plotxrange[0]) / x_scale
            y_length = (plotyrange[1] - plotyrange[0]) / y_scale
            x_step = 0.0
            y_step = 0.0
            # print("x length", i + 1,x_length)
            # print("y length", i + 1,y_length)
            change_x = True
            change_y = True
            if(len(self.x_step_min) == len(self.axlist) and \
               len(self.x_step_min_2) == len(self.axlist_2) and \
               None not in self.x_step_min and None not in self.x_step_min_2):
                if(i < len(self.axlist)):
                    x_step_min = self.x_step_min[i]
                    x_step_max = self.x_step_max[i]
                    y_step_min = self.y_step_min[i]
                    y_step_max = self.y_step_max[i]
                else:
                    x_step_min = self.x_step_min_2[i - len(self.axlist)]
                    x_step_max = self.x_step_max_2[i - len(self.axlist)]
                    y_step_min = self.y_step_min_2[i - len(self.axlist)]
                    y_step_max = self.y_step_max_2[i - len(self.axlist)]
            else:
                if(plot_mode == "Presentation"):
                    x_step_max = 4
                    x_step_min = 3
                    y_step_max = 5
                    y_step_min = 3
                elif(plot_mode == "Article"):
                    x_step_max = 5
                    x_step_min = 3
                    y_step_max = 7
                    y_step_min = 3
                elif(plot_mode == "Software"):
                    x_step_max = 6
                    x_step_min = 4
                    y_step_max = 8
                    y_step_min = 4
            for step in steps:
                N_x = int(x_length / step)
                N_y = int(y_length / step)
                if(x_step == 0 and N_x <= x_step_max and N_x >= x_step_min):
                    x_0 = np.floor(plotxrange[0] / x_scale) * x_scale
                    if(x_0 + step * x_scale < plotxrange[0]):
                        x_0 += step * x_scale
                    x_0 += step * x_scale  # Remove the first point
                    x_1 = x_0
                    while(x_1 + step * x_scale <= plotxrange[1]):
                        x_1 += step * x_scale
                    x_1 += step * x_scale  # Due to the nature of arange we actually need a larger x_1
                    # if(x_1 > plotxrange[1]):
                    #    x_1 -= step * x_scale
                    # x_0 = np.round(x_0, 3)
                    # x_1 = np.round(x_1, 3)
#                    N_x = int((x_1 - x_0) / x_scale / step + 0.5)
#                    if(np.round(np.abs(x_1 - x_0) / N_x,3) != np.round(step * x_scale,3)):
#                            change_x = False
#                            print("Could not find step x major")
#                            print("N , start, stop, step", N_x, x_0, x_1, np.abs(x_0 - x_1) / N_x / x_scale)
                    loc = np.arange(x_0, x_1, step * x_scale)
#                    print("x_0", x_0)
#                    print("x_1", x_1)
#                    print("plot x range", plotxrange)
#                    print(" x_step for ax: ", ax.get_lines()[0].get_label())
#                    print("loc", loc)
#                    print("Plot scale: ",x_scale)
#                    print("x_step found", step, int(x_length / step))
#                    print("x-length: ",(x_length / steps).astype(int))
                    if(change_x and ax.get_xaxis().get_scale() != "log"):
                        ax.get_xaxis().set_major_locator(FixedLocator(loc))
                    x_0 = np.floor(plotxrange[0] / x_scale) * x_scale
                    if(x_0 + step * x_scale < plotxrange[0]):
                        x_0 += step * x_scale
                    x_0 += 0.5 * step * x_scale
                    x_1 = x_0
                    while(x_1 + step * x_scale <= plotxrange[1]):
                        x_1 += step * x_scale
                    x_1 += step * x_scale
#                    if(x_1 > plotxrange[1]):
#                        x_1 -= step * x_scale
#                    x_0 = np.round(x_0, 3)
#                    x_1 = np.round(x_1, 3)
#                    N_x_min = int((x_1 - x_0) / x_scale / step)
#                    if(np.round(np.abs(x_1 - x_0) / N_x_min,3) != np.round(step * x_scale,3)):
#                            change_x = False
#                            print("Could not find step x minor")
#                            print("N , start, stop, step", N_x_min, x_0, x_1, np.abs(x_0 - x_1) / N_x_min / x_scale )
                    loc = np.arange(x_0, x_1, step * x_scale)
                    if(ax.get_xaxis().get_scale() != "log"):
                        ax.get_xaxis().set_minor_locator(FixedLocator(loc))
                    x_step = step
                if(y_step == 0 and N_y <= y_step_max and N_y > y_step_min):
                    y_0 = np.floor(plotyrange[0] / y_scale) * y_scale
                    if(y_0 + step * y_scale < plotyrange[0]):
                        y_0 += step * y_scale
                    y_1 = y_0
                    while(y_1 + step * y_scale <= plotyrange[1]):
                        y_1 += step * y_scale
                    y_1 += step * y_scale  # need one extra step since the last one is not accounted for by np.arange
#                    y_0 = np.round(y_0, 3)
#                    y_1 = np.round(y_1, 3)
#                    N_y = int((y_1 - y_0) / y_scale / step + 1)
#                    if(np.round(np.abs(y_1 - y_0) / N_y,3) != np.round(step * y_scale,3)):
#                            change_y = False
#                            print("Could not find step y major")
#                            print("N , start, stop, step", N_y, y_0, y_1, np.abs(y_0 - y_1) / N_y / y_scale )
                    loc = np.arange(y_0, y_1, step * y_scale)
#                    print("y_0", y_0)
#                    print("y_1", y_1)
#                    print("plot y range", plotyrange)
#                    print(" y_step for ax: ", ax.get_lines()[0].get_label())
#                    print("loc", loc)
#                    print("Plot scale: ",y_scale)
#                    print("y_step found", step, int(y_length / step))
#                    print("y-length: ",(y_length / steps).astype(int))
                    if(ax.get_yaxis().get_scale() != "log"):
                        ax.get_yaxis().set_major_locator(FixedLocator(loc))
                    y_0 = np.floor(plotyrange[0] / y_scale) * y_scale
                    if(y_0 + step * y_scale < plotyrange[0]):
                        y_0 += step * y_scale
                    y_0 += 0.5 * step * y_scale
                    y_1 = y_0
                    while(y_1 + step * y_scale <= plotyrange[1]):
                        y_1 += step * y_scale
                    y_1 += step * y_scale
                    y_0 = np.round(y_0, 3)
                    y_1 = np.round(y_1, 3)
#                    N_y_min = int((y_1 - y_0) / y_scale / step)
#                    if(np.round(np.abs(y_1 - y_0) / N_y_min,3) != np.round(step * y_scale,3)):
#                            change_y = False
#                            print("Could not find step y minor")
#                            print("N , start, stop, step", N_y_min, y_0, y_1, np.abs(y_0 - y_1) / N_y_min / y_scale)
                    loc = np.arange(y_0, y_1, step * y_scale)
                    if(ax.get_yaxis().get_scale() != "log"):
                        ax.get_yaxis().set_minor_locator(FixedLocator(loc))
                    y_step = step
                if(x_step != 0.0 and y_step != 0.0):
                    break
            if(x_step == 0.0):
                print("Failed to find x_step for ax: ", ax.get_label())
                print("Plot scale: ", x_scale)
                print("plot x range", plotxrange)
                print("x-length: ", (x_length / steps).astype(int))
            if(y_step == 0.0):
                print("Failed to find y_step for ax: ", ax.get_label())
                print("Plot scale: ", y_scale)
                print("plot y range", plotyrange)
                print("y-length: ", (y_length / steps).astype(int))
#        for i in range(len(self.axlist_2)):
#            if(x_step == 0 and int(x_length / step) <= 6 and int(x_length / step) >= 4):
#                x_0 = int(plotxrange[0] / x_scale) * x_scale
#                x_1 = x_0
#                while(x_1 + step * x_scale < plotxrange[1]):
#                    x_1 += step * x_scale
#                N = (x_1 - x_0) / x_scale / step + 1
#                loc = np.linspace(x_0, x_1, N, True)
# #                    print("x_step found", step, int(x_length / step))
# #                    print(" x_step for ax: ",i)
# #                    print("loc", loc)
# #                    print("Plot scale: ",x_scale)
# #                    print("plot x range", plotxrange)
# #                    print("x-length: ",(x_length / steps).astype(int))
# #                    print("x_0", y_0)
#                self.axlist_2[i].get_xaxis().set_major_locator(FixedLocator(loc))
#                x_0 = int(plotxrange[0] / x_scale) * x_scale + 0.5 * step * x_scale
#                x_1 = x_0
#                while(x_1 + step * x_scale < plotxrange[1]):
#                    x_1 += step * x_scale
#                loc = np.linspace(x_0,x_1, N,True)
#                self.axlist_2[i].get_xaxis().set_minor_locator(FixedLocator(loc))
#                x_step = step
#            if(y_step == 0 and int(y_length / step) <= 5 and int(y_length / step) >= 3):
#                y_0 = int(plotyrange[0] / y_scale) * y_scale
#                y_1 = y_0
#                while(y_1 + step * y_scale < plotyrange[1]):
#                    y_1 += step * y_scale
#                N = (y_1 - y_0) / y_scale / step + 1
#                loc = np.linspace(y_0, y_1, N, True)
# #                    print("y_step found", step, int(y_length / step))
# #                    print(" y_step for ax: ",i)
# #                    print("loc", loc)
# #                    print("Plot scale: ",y_scale)
# #                    print("plot y range", plotyrange)
# #                    print("y-length: ",(y_length / steps).astype(int))
# #                    print("y_0", y_0)
#                self.axlist_2[i].get_yaxis().set_major_locator(FixedLocator(loc))
#                y_0 = int(plotyrange[0] / y_scale) * y_scale + 0.5 * step * y_scale
#                y_1 = int(plotyrange[1] / y_scale) * y_scale + 0.5 * step * y_scale
#                loc = np.linspace(y_0,y_1, N,True)
#                self.axlist_2[i].get_yaxis().set_minor_locator(FixedLocator(loc))
#                y_step = step
#            if(x_step != 0.0 and y_step != 0.0):
#                break
#            if(x_step == 0.0):
#                print("Failed to find x_step for fig2 ax: ",i)
#                print("Plot scale: ",x_scale)
#                print("plot x range", plotxrange)
#                print("x-length: ",(x_length / steps).astype(int))
#            if(y_step == 0.0):
#                print("Failed to find y_step for fig2 ax: ",i)
#                print("plot y range", plotyrange)
#                print("Plot scale: ",y_scale)
#                print("y-length: ",(y_length / steps).astype(int))
        if(self.fig_2 is None):
            return self.fig
        else:
            return self.fig, self.fig_2


    def add_plot(self, ax, y_range_in, filename=None, data=None, x_error=None, y_error=None, maxlines=0, name=None, first_invoke=None, \
                        marker=None, coloumn=1, color=None, mode=111, ax_flag=None, xlabel=None, ylabel=None, \
                        vline=None, x_scale=1.0, y_scale=1.0, log_flag=False, sample=1, label_x=True, \
                        linewidth=plt.rcParams['lines.linewidth']):
        if(filename is None and data is None):
            return ax, None, [0.0, 0.0]
        if(data is not None):
            x = np.atleast_1d(data[0])
            y = np.atleast_1d(data[1])
            if(x.shape[0] == 0 or y.shape[0] == 0 or not np.any(np.isfinite(x)) or not np.any(np.isfinite(y))):
                print("Warning empty array encountered while plotting!!")
                return ax, y_range_in
        elif(filename is not None):
            x, y = self.read_file(filename, maxlines, coloumn)
        else:
            print("Neither data nor file to be plotted specified.")
            return ax, y_range_in
        x = x * x_scale
        y = y * y_scale
        if(x.shape != y.shape):
            print("x and y do not have the same shape !")
            if(name is not None):
                print("Plot name: ", name)
            if(ax_flag is not None):
                print("ax_flag: ", ax_flag)
            print(x.shape, y.shape)
            print("Nothing was plotted this time")
            return ax, y_range_in
        scale_down = False
        # if(np.any(y > 100) and ax_flag == "I"):
        #    y *= 1.e-3
        #    scale_down = True
        if(ax_flag == "ne"):
            y = y * 1.e1
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        if(marker is None and name is None and color is None):
            if(x_error is not None or y_error is not None):
                if(x_error is None  and y_error is not None):
                    y_error = y_error * y_scale
                    ax.errorbar(x[::sample], y[::sample], yerr=y_error[::sample], markeredgecolor='black', markeredgewidth=1.25)
                elif(x_error is not None  and y_error is None):
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], markeredgecolor='black', markeredgewidth=1.25)
                else:
                    x_error = x_error * x_scale
                    y_error = y_error * y_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], yerr=y_error[::sample], \
                                markeredgecolor='black', markeredgewidth=1.25)
            else:
                ax.plot(x[::sample], y[::sample])
        elif(marker is not None and name is None and color is None):
            if(x_error is not None or y_error is not None):
                if(x_error is None  and y_error is not None):
                    y_error = y_error * y_scale
                    ax.errorbar(x[::sample], y[::sample], yerr=y_error[::sample], fmt=marker)
                elif(x_error is not None  and y_error is None):
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], fmt=marker)
                else:
                    y_error = y_error * y_scale
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], yerr=y_error[::sample], marker=marker)
            else:
                ax.plot(x[::sample], y[::sample], marker, mfc=(0.0, 0.0, 0.0, 0.0))
        elif(marker is None and name is not None and color is None):
            if(x_error is not None or y_error is not None):
                if(x_error is None  and y_error is not None):
                    y_error = y_error * y_scale
                    ax.errorbar(x[::sample], y[::sample], yerr=y_error[::sample], label=name)
                elif(x_error is not None  and y_error is None):
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], label=name)
                else:
                    y_error = y_error * y_scale
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], yerr=y_error[::sample], label=name)
            else:
                ax.plot(x[::sample], y[::sample], label=name, linewidth=linewidth)
        elif(marker is not None and color is None and name is not None):
            if(x_error is not None or y_error is not None):
                if(x_error is None  and y_error is not None):
                    y_error = y_error * y_scale
                    ax.errorbar(x[::sample], y[::sample], yerr=y_error[::sample], fmt=marker, label=name)
                elif(x_error is not None  and y_error is None):
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], fmt=marker, label=name)
                else:
                    y_error = y_error * y_scale
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], yerr=y_error[::sample], fmt=marker, label=name)
            else:
                ax.plot(x[::sample], y[::sample], marker, label=name, linewidth=linewidth)
        elif(marker is None and color is not None and name is not None):
            if(x_error is not None or y_error is not None):
                if(x_error is None  and y_error is not None):
                    y_error = y_error * y_scale
                    ax.errorbar(x[::sample], y[::sample], y_err=y_error[::sample], label=name, color=color, mfc="none", mec=color)
                elif(x_error is not None  and y_error is None):
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], x_err=x_error[::sample], label=name, color=color, mfc="none", mec=color)
                else:
                    y_error = y_error * y_scale
                    x_error = x_error * x_scale
                    ax.errorbar(x[::sample], y[::sample], x_err=x_error[::sample], y_err=y_error[::sample], label=name, color=color, mfc="none", mec=color)
            else:
                ax.plot(x[::sample], y[::sample], label=name, color=color, mfc="none", mec=color, linewidth=linewidth)
        else:
            if(marker != "D"):
                if(x_error is not None or y_error is not None):
                    if(x_error is None  and y_error is not None):
                        y_error = y_error * y_scale
                        ax.errorbar(x[::sample], y[::sample], yerr=y_error[::sample], fmt=marker, \
                                 label=name, color=color, mfc="none", mec=color)
                    elif(x_error is not None  and y_error is None):
                        x_error = x_error * x_scale
                        ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], fmt=marker, \
                                 label=name, color=color, mfc="none", mec=color)
                    else:
                        y_error = y_error * y_scale
                        x_error = x_error * x_scale
                        ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], yerr=y_error[::sample], fmt=marker, \
                                 label=name, color=color, mfc="none", mec=color)
                else:
                    ax.plot(x[::sample], y[::sample], marker, linewidth=linewidth, \
                                 label=name, color=color, mfc="none", mec=color)
            else:
                if(x_error is not None or y_error is not None):
                    if(x_error is None  and y_error is not None):
                        y_error = y_error * y_scale
                        ax.errorbar(x[::sample], y[::sample], yerr=y_error[::sample], fmt=marker, \
                                 label=name, color=color)
                    elif(x_error is not None  and y_error is None):
                        x_error = x_error * x_scale
                        ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], fmt=marker, \
                                 label=name, color=color)
                    else:
                        y_error = y_error * y_scale
                        x_error = x_error * x_scale
                        ax.errorbar(x[::sample], y[::sample], xerr=x_error[::sample], yerr=y_error[::sample], fmt=marker, \
                                 label=name, color=color)
                else:
                    ax.plot(x[::sample], y[::sample], marker, label=name, color=color, linewidth=linewidth)
        if(log_flag == True and np.any(y > 0)):
            ax.set_yscale('log')
        if(y_range_in is not None):
            if(y_error is not None):
                ymax = np.nanmax([y_range_in[1], np.nanmax(y[np.abs(y) != np.inf] + y_error[np.abs(y_error) != np.inf])])
                ymin = np.nanmin([y_range_in[0], np.nanmin(y[np.abs(y) != np.inf] - y_error[np.abs(y_error) != np.inf])])
            else:
                ymax = np.nanmax([y_range_in[1], np.nanmax(y[np.abs(y) != np.inf])])
                ymin = np.nanmin([y_range_in[0], np.nanmin(y[np.abs(y) != np.inf])])
            if(ax_flag == "T"):
                ymax = np.nanmax([y_range_in[1], np.nanmax(y[np.abs(y) != np.inf])])
                ymin = np.nanmin([y_range_in[0], np.nanmin(y[np.abs(y) != np.inf]), 0])
            if(ymax != np.inf and ymin != -np.inf and not np.isnan(ymax) and not np.isnan(ymin)):
                if(ymin == ymax):
                    ymin = ymax * 0.75
                    ymax = ymax * 1.25
                else:
                    ymin = ymin * 0.95
                    ymax = ymax * 1.05
                y_range = [ymin, ymax]
            else:
                y_range = y_range_in
        else:
            ymax = max(y)
            ymin = min(y)
            if(ymin == ymax):
                ymin = ymax * 0.75
                ymax = ymax * 1.25
            else:
                ymin = ymin * 0.95
                ymax = ymax * 1.05
            y_range = [ymin, ymax]
        if(ax_flag is not None):
            if(ax_flag.startswith("Te") or  ax_flag.startswith("ne") or \
               ax_flag.startswith("P_")):
                y_range = [0, ymax]
            elif(ax_flag.startswith("cnt")):
                y_range = [1, ymax]
            elif(ax_flag == "n-trace"):
                y_range = [100, ymax]
                y_range = [100, ymax]
        y_max = np.max(y_range)
        y_min = np.min(y_range)
        y_range[0] = y_min
        y_range[1] = y_max
        if(y_range[0] == y_range[1]):
            if(y_range[0] == 0.e0):
                y_range[1] = 1.e0
            else:
                y_range[0] *= 0.5e0
                y_range[1] *= 2.e0
        if(ax_flag is not None):
            if(plot_mode == "Software"):
                if(ax_flag == "Te_trace"):
                    ax.set_xlabel(r"$t$ [s]")
                    ax.set_ylabel(r"$T_\mathrm{rad}$  [keV]")
                    if(vline is not None):
                        if(color is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                        else:
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Te_Te_Trad"):
                    ax.set_ylabel(r"$T_\mathrm{e/rad}$ [keV]")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$", fontsize=int(plt.rcParams['axes.labelsize'] * 1.33))
                elif(ax_flag == "Te"):
                    ax.set_ylabel(r"$T_\mathrm{e}$ [keV]")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$", fontsize=int(plt.rcParams['axes.labelsize'] * 1.33))
                elif(ax_flag == "Te_tor"):
                    ax.set_ylabel(r"$T_\mathrm{e}$ [keV]")
                    ax.set_xlabel(r"$\rho_\mathrm{tor}$")
                elif(ax_flag == "tau"):
                    ax.set_ylabel(r"$\tau_\omega$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                elif(ax_flag == "Trad_trace"):
                        ax.set_xlabel(r"$t$ [s]")
                        ax.set_ylabel(r"$T_\mathrm{rad}$ [keV]")
                        if(vline is not None):
                            if(color is not None):
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                            else:
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "diag_trace"):
                        ax.set_xlabel(r"$t$ [s]")
                        ax.set_ylabel(r"$c$ [mV/s]")
                        if(vline is not None):
                            if(color is not None):
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                            else:
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Calib"):
                    ax.set_xlabel(r"$f$ [GHz]")
                    ax.set_ylabel(r"$c$ [keV/(V/s)]")
                elif(ax_flag == "Calib_trace"):
                    ax.set_xlabel(r"$t$ [s]")
                    ax.set_ylabel(r"$c$ [eV/(V/s)]")
                elif(ax_flag == "Sig_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"Sig [V/s]")
                elif(ax_flag == "Sig_vs_Trad_small"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"Sig [mV/s]")
                elif(ax_flag == "Sig_vs_time"):
                    ax.set_xlabel(r"$t$ [s]")
                    ax.set_ylabel(r"Sig [V/s]")
                elif(ax_flag == "Sig_vs_Trad"):
                    ax.set_xlabel(r"Sig [V/s]")
                    ax.set_ylabel(r"$T_\mathrm{rad,mod}$ [keV]")
                elif(ax_flag == "Ang_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "Ang_vs_Sig"):
                    ax.set_xlabel(r"Sig [V/s]")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "calib_vs_launch"):
                    ax.set_xlabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                    ax.set_ylabel(r"$c$ [keV/(V/s)]")
                elif(ax_flag == "Rz"):
                    ax.set_xlabel(r"$R$ [m]")
                    ax.set_ylabel(r"$z$ [m]")
                elif(ax_flag == "E_field"):
                    ax.set_xlabel(r"$\Psi$  [V s]")
                    ax.set_ylabel(r"$E_\mathrm{tor}$ [V/m]")
                elif(ax_flag == "Rdiff"):
                    ax.set_xlabel(r"$\tilde{r} [m]$")
                    ax.set_ylabel(r"$R_\mathrm{diff}$ [m^2/s]$")
                elif(ax_flag == "xy"):
                    ax.set_xlabel(r"$x$ [m]")
                    ax.set_ylabel(r"$y$ [m]")
                elif(ax_flag == "Rphi"):
                    ax.set_xlabel(r"$R$ [m]")
                    ax.set_ylabel(r"$\phi \,[^{\circ}]$")
                elif(ax_flag == "j_weighted"):
                    ax.set_ylabel(r"$D_\omega$ [1/m]")  # \left[\si{\kilo\electronvolt}\right]
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    if(y_range[0] - y_range[0] <= 0.0 and vline is not None):
                        if(color is not None):
                            ax.vlines(vline, 0, 2000, linestyle='dotted', color=color)
                        else:
                            ax.vlines(vline, 0, 2000, linestyle='dotted')
                    elif(vline is not None):
                        if(color is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                        else:
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "j_weighted_Trad"):
                    ax.set_ylabel(r"$D_\omega$ [keV/m]")  #
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    if(y_range[0] - y_range[0] <= 0.0 and vline is not None):
                        if(color is not None):
                            ax.vlines(vline, 0, 2000, linestyle='dotted', color=color)
                        else:
                            ax.vlines(vline, 0, 2000, linestyle='dotted')
                    elif(vline is not None):
                        if(color is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                        else:
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
            else:
                if(ax_flag.startswith("Te")):
                    x_ax = "rho"
                    ax.set_ylabel(r"$T_\mathrm{e} \,\left[\si{\kilo\electronvolt}\right]$")  # /rad
                    if(ax_flag.endswith("_R")):
                        ax_flag = ax_flag.replace("_R", "")
                        x_ax = "R"
                    elif(ax_flag.endswith("_s")):
                        ax_flag = ax_flag.replace("_s", "")
                        x_ax = "s"
                    elif(ax_flag == "Te_trace"):
                        x_ax = "t"
                        ax.set_ylabel(r"$T_\mathrm{e} \, \left[\si{\kilo\electronvolt}\right]$")
                        if(vline is not None):
                            if(color is not None):
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                            else:
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    elif(ax_flag == "Te_I"):
                        x_ax = "R"
                        if(vline is not None):
                            if(color is not None):
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                            else:
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    elif(ax_flag == "Te_spec"):
                        x_ax = "f"
                        ax.set_ylabel(r"$T_\mathrm{e} \,\left[\si{\kilo\electronvolt}\right]$")
                    elif(ax_flag == "Te"):
                        ax.set_ylabel(r"$T_\mathrm{e} \,\left[\si{\kilo\electronvolt}\right]$")
                    elif(ax_flag == "Te_rad"):
                        ax.set_ylabel(r"$T_\mathrm{rad}\,\left[\si{\kilo\electronvolt}\right]$")
                    elif(ax_flag == "Te_Te_rad"):
                        ax.set_ylabel(r"$T_\mathrm{e/rad}\,\left[\si{\kilo\electronvolt}\right]$")
                    if(label_x == True):
                        if(x_ax == "s"):
                            ax.set_xlabel(r"$s \left[\si{\meter}\right]")
                        elif(x_ax == "R"):
                            ax.set_xlabel(r"$R \left[\si{\meter}\right]")
                        elif(x_ax == "t"):
                            ax.set_xlabel(r"$t \,\left[\si{\second}\right]$")
                        elif(x_ax == "f"):
                            ax.set_xlabel(r"$f \left[\si{\giga\hertz}\right]$")
                        else:
                            ax.set_xlabel(r"$\rho_\mathrm{pol}$", fontsize=int(plt.rcParams['axes.labelsize'] * 1.2))
                elif(ax_flag == "NPA_trace"):
                    ax.set_xlabel(r"$t \,\left[\si{\second}\right]$")
                    ax.set_ylabel(r"a. u.")
                    if(vline is not None):
                        if(color is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                        else:
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == 'rel_diff_Trad'):
                    ax.set_xlabel(r"\Huge$\rho_\mathrm{pol}$\normalsize")
                    ax.set_ylabel(r"$\Delta T_\mathrm{rad}\,\left[\si{\percent}\right]$")
                elif(ax_flag == "diag_trace"):
                        ax.set_xlabel(r"$t \,\left[\si{\second}\right]$")
                        ax.set_ylabel(r"$c \left[\si{\milli\volt\per\second}\right]$")
                        if(vline is not None):
                            if(color is not None):
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                            else:
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "ne_trace"):
                    ax.set_xlabel(r"$t / \mathrm{s}$")
                    ax.set_ylabel(r"$n_\mathrm{e}\left[\SI{1.e19}{\per\cubic\metre}\right]$")
                elif(ax_flag == "P_trace"):
                    ax.set_xlabel(r"$t$ $/$ $\mathrm{s}$")
                    ax.set_ylabel(r"$P \left[\si{\mega\watt}\right]$")
                elif(ax_flag == "cnt_trace"):
                    ax.set_xlabel(r"$t [\mathrm{s}]$")
                    ax.set_ylabel(r"cnt rate $  \left[\si{\kilo\hertz}\right]$")
                elif(ax_flag == "Calib"):
                    ax.set_xlabel(r"$f \left[\si{\giga\hertz}\right]$")
                    ax.set_ylabel(r"$c \left[\si{\kilo\electronvolt \per\volt\per\second}\right]$")
                elif(ax_flag == "Trad_trace"):
                        ax.set_xlabel(r"$t$ [s]")
                        ax.set_ylabel(r"$T_\mathrm{rad} \left[\si{\kilo\electronvolt}\right]$")
                        if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Calib_trace"):
                    ax.set_xlabel(r"$t\,[\mathrm{s}]$")
                    ax.set_ylabel(r"$\vert c \vert \left[\si{\electronvolt \per \volt\per\second}\right]$")
                elif(ax_flag == "Grid"):
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    ax.set_ylabel(r"index")
                elif(ax_flag == "E_field"):
                    ax.set_xlabel(r"$\Psi  [\si{\volt \second}]$")
                    ax.set_ylabel(r"$E_\mathrm{tor} [\si{\volt \per\metre}]$")
                elif(ax_flag == "calib_vs_launch"):
                    ax.set_xlabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                    ax.set_ylabel(r"$c$ [keV/(V/s)]")
                elif(ax_flag == "Calib_ch"):
                    ax.set_xlabel(r"ch no.")
                    ax.set_ylabel(r"$c \left[\si{\electronvolt \per\volt\per\second}\right]$")
                elif(ax_flag == "Calib_dev"):
                    ax.set_xlabel(r"$f \left[\si{\giga\hertz}\right]$")
                    ax.set_ylabel(r"$\overline{c - \overline{c}} / N /\overline{c}\left[\si{\percent}\right]$")
                elif(ax_flag == "Calib_dev_ch"):
                    ax.set_xlabel(r"ch no.")
                    ax.set_ylabel(r"$\overline{c - \overline{c}} / N /\overline{c}\left[\si{\percent}\right]$")
                elif(ax_flag == "n_trace"):
                    ax.set_xlabel(r"$t$ $/$ $\mathrm{s}$")
                    ax.set_ylabel(r"N-rate $  \left[\si{\giga\hertz}\right]$")
                elif(ax_flag == "ne"):
                    ax.set_ylabel(r"$n_\mathrm{e}\, \left[\SI{e19}{\per\cubic\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                elif(ax_flag == "ne_tor"):
                    ax.set_ylabel(r"$n_\mathrm{e}\, \left[\SI{e19}{\per\cubic\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{tor}$")
                elif(ax_flag == "ne_R"):
                    ax.set_ylabel(r"$n_\mathrm{e}\, \left[\SI{e19}{\per\cubic\meter}\right]$")
                    ax.set_xlabel(r"$R \left[\si{\meter}\right]")
                elif(ax_flag == "P_ecrh_tor"):
                    ax.set_ylabel(r"$dP/dV\, \left[\si{\mega\watt\per\square\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{tor}$")
                elif(ax_flag == "P_ecrh"):
                    ax.set_ylabel(r"$P\, \left[\si{\mega\watt\per\square\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                elif(ax_flag == "P_ecrh_kw"):
                    ax.set_ylabel(r"$P\, \left[\si{\kilo\watt\per\square\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                elif(ax_flag == "j_eccd_tor"):
                    ax.set_ylabel(r"$j\, \left[\si{\mega\ampere\per\square\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{tor}$")
                elif(ax_flag == "j_eccd"):
                    ax.set_ylabel(r"$j\, \left[\si{\mega\ampere\per\square\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                elif(ax_flag == "j_eccd_kA"):
                    ax.set_ylabel(r"$j\, \left[\si{\kilo\ampere\per\square\meter}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                elif(ax_flag == "f_cut"):
                    ax.set_xlabel(r"$\vert u_\Vert \vert=  u_\perp$")
                    ax.set_ylabel(r"$\log_\mathrm{10}(f)$")
                elif(ax_flag == "f"):
                    ax.set_ylabel(r"$f \left[\si{\giga\hertz}\right]$")
                    ax.set_xlabel(r"$R \left[\si{\meter}\right]$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "rf"):
                    ax.set_ylabel(r"$B_\mathrm{OERT} / B_\mathrm{IDA} - 1$")
                    ax.set_xlabel(r"$R \left[\si{\meter}\right]$")
                elif(ax_flag == "shift"):
                    ax.set_ylabel(r"$f(\omega_\mathrm{c,obs})$")
                    ax.set_xlabel(r"$\frac{\omega_\mathrm{c,obs}}{\omega_\mathrm{c,0}}$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "shift_freq"):
                    ax.set_ylabel(r"$\mathrm{p.\,d.\,f.}  [\si{\per\giga\hertz}]$")
                    ax.set_xlabel(r"$\omega_\mathrm{c,obs} [\si{\giga\hertz}]$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "alpha_Y"):
                    ax.set_ylabel(r"$\alpha_\omega \left[\si{\per\milli\metre}\right]$")
                    ax.set_xlabel(r"$\frac{\omega_\mathrm{c,0}}{\omega}$")
                elif(ax_flag == "N_Y"):
                    ax.set_ylabel(r"$N_\omega$")
                    ax.set_xlabel(r"$\frac{\omega_\mathrm{c,0}}{\omega}$")
                elif(ax_flag == "shift"):
                    ax.set_ylabel(r"$f(\omega_\mathrm{c,obs})$")
                    ax.set_xlabel(r"$\frac{\omega_\mathrm{c,obs}}{\omega_\mathrm{c,0}}$")
                elif(ax_flag == "X_frac"):
                    ax.set_xlabel(r"\Huge$\rho_\mathrm{pol}$\normalsize")
                    ax.set_ylabel(r"$\left(\vec{e}_\mathrm{X} \cdot \vec{p}\right)^2\,\left[\si{\percent}\right]$")
                elif(ax_flag == "shift_limit"):
                    ax.set_ylabel(r"$f(\omega_\mathrm{c,obs})$")
                    ax.set_xlabel(r"$\frac{\omega_\mathrm{c,obs}}{\omega_\mathrm{c,0}}$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dashed', colors="r")
                elif(ax_flag == "alpha"):
                    ax.set_ylabel(r"$\alpha_\omega\,\left[\si{\per\milli\meter}\right]$")
                    # ax.set_xlabel(r"$R$ $/$ $\mathrm{m}$")
                    ax.set_xlabel(r"$s\,\left[\si{\meter}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "alpha_s"):
                    ax.set_ylabel(r"$\alpha_\omega\,\left[\si{\per\milli\meter}\right]$")
                    # ax.set_xlabel(r"$R$ $/$ $\mathrm{m}$")
                    ax.set_xlabel(r"$s\,\left[\si{\meter}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "alpha_R"):
                    ax.set_ylabel(r"$\alpha_\omega\,\left[\si{\per\milli\meter}\right]$")
                    # ax.set_xlabel(r"$R$ $/$ $\mathrm{m}$")
                    ax.set_xlabel(r"$R\,\left[\si{\meter}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "alpha_rhop"):
                    ax.set_ylabel(r"$\alpha_\omega\,\left[\si{\per\milli\meter}\right]$")
                    # ax.set_xlabel(r"$R$ $/$ $\mathrm{m}$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "alpha_rho"):
                    ax.set_ylabel(r"$\alpha_\omega\,\left[\si{\per\milli\meter}\right]$")
                    # ax.set_xlabel(r"$R$ $/$ $\mathrm{m}$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "tau"):
                    ax.set_ylabel(r"$\tau_\omega$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "T"):
                    ax.set_ylabel(r"$T_\omega$")
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    y_range[1] *= 2.5
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "T_s"):
                    ax.set_ylabel(r"$T_\omega$")
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    y_range[1] = 1.25
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "T_R"):
                    ax.set_ylabel(r"$T_\omega$")
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    y_range[1] = 1.25
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "T_rho"):
                    ax.set_ylabel(r"$T_\omega$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    y_range[1] *= 1.2
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "T_rhop"):
                    ax.set_ylabel(r"$T_\omega$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    y_range[1] *= 1.2
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "N"):
                    ax.set_ylabel(r"$N_\omega$")
                    ax.set_xlabel(r"\Huge$\rho_\mathrm{pol}$\normalsize")
                    # ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    # y_range[1] *= 2.5
                    # ax.set_xlim(min(x),max(x))
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Sig_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                    ax.set_ylabel(r"Sig [\si{\volt\per\second}]")
                elif(ax_flag == "Sig_vs_Trad_small"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                    ax.set_ylabel(r"Sig [\si{\mili\volt\per\second}]")
                elif(ax_flag == "Ang_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ \si{\kilo\electronvolt}")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "Trad_vs_Sig"):
                    ax.set_xlabel(r"Sig [\si{\volt\per\second}]")
                    ax.set_ylabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                elif(ax_flag == "Ang_vs_Sig"):
                    ax.set_xlabel(r"Sig [\si{\volt\per\second}]")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "N_omega"):
                    ax.set_ylabel(r"$N_\omega$")
                    ax.set_xlabel(r"$\frac{\omega_\mathrm{c,0}}{\omega}$")
                    # ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    # y_range[1] *= 2.5
                    # ax.set_xlim(min(x),max(x))
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "j"):
                    ax.set_ylabel(r"$j_\omega \,\left[\si{\nano\watt\per\cubic\metre\per\steradian\per\hertz}\right]$")
                    ax.set_xlabel(r"$s \left[\si{\meter}\right]$")
                elif(ax_flag == "j_s"):
                    ax.set_ylabel(r"$j_\omega \,\left[\si{\nano\watt\per\cubic\metre\per\steradian\per\hertz}\right]$")
                    ax.set_xlabel(r"$s \left[\si{\meter}\right]$")
                elif(ax_flag == "j_R"):
                    ax.set_ylabel(r"$j_\omega \,\left[\si{\nano\watt\per\cubic\metre\per\steradian\per\hertz}\right]$")
                    ax.set_xlabel(r"$R \left[\si{\meter}\right]$")
                elif(ax_flag == "j_rhop"):
                    ax.set_ylabel(r"$j_\omega \,\left[\si{\nano\watt\per\cubic\metre\per\steradian\per\hertz}\right]$")
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    # ax.set_xlabel(r"$R \left[\si{\meter}\right]$")
                elif(ax_flag == "j_weighted"):
                    ax.set_ylabel(r"$D_\omega\,\left[\si{\per\metre}\right]$")  # \left[\si{\kilo\electronvolt}\right]
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "j_weighted_Trad"):
                    ax.set_ylabel(r"$D_\omega \,\left[\si{\kilo\electronvolt\per\metre}\right]$")  #
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "j_weighted_s"):
                    ax.set_ylabel(r"$D_\omega\,\left[\si{\per\metre}\right]$")  # \left[\si{\kilo\electronvolt}\right]
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "j_weighted_R"):
                    ax.set_ylabel(r"$D_\omega\,\left[\si{\per\metre}\right]$")  # \left[\si{\kilo\electronvolt}\right]
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "j_weighted_s_Trad"):
                    ax.set_ylabel(r"$D_\omega(\rho_\mathrm{pol}) \,\left[\si{\kilo\electronvolt\per\metre}\right]$")  #
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Iabs"):
                    ax.set_ylabel(r"$\alpha \cdot I / $W$/($m$^3$srHz$)$")
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                elif(ax_flag == "I"):
                    if(scale_down):
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")  # /\,\mathrm{mW}/(\mathrm{mm}^2\mathrm{srHz})
                    else:
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                elif(ax_flag == "I_s"):
                    if(scale_down):
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")  # /\,\mathrm{mW}/(\mathrm{mm}^2\mathrm{srHz})
                    else:
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                elif(ax_flag == "I_R"):
                    if(scale_down):
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")  # /\,\mathrm{mW}/(\mathrm{mm}^2\mathrm{srHz})
                    else:
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                elif(ax_flag == "I_rhop"):
                    if(scale_down):
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")  # /\,\mathrm{mW}/(\mathrm{mm}^2\mathrm{srHz})
                    else:
                        ax.set_ylabel(r"$I_\omega\,\left[\si{\pico\watt\per\square\meter\per\steradian\per\hertz}\right]$")
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                elif(ax_flag == "omega"):
                    ax.set_ylabel(r"$\frac{\omega_\mathrm{2X}}{\omega}$")
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                elif(ax_flag == "ang"):
                    ax.set_ylabel(r"$\theta \,\left[\si{\degree}\right]$")
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    ax.set_ylim(0.9 * y_range[0], 1.10 * y_range[1])
                elif(ax_flag == "theta_phi_trace"):
                    ax.set_ylabel(r"$\theta/\phi \,\left[\si{\degree}\right]$")
                    ax.set_xlabel(r"$t \,\left[\si{\second}\right]$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                    ax.set_ylim(0.9 * y_range[0], 1.10 * y_range[1])
                elif(ax_flag == "dist_par"):
                    ax.set_ylabel(r"$\log_\mathrm{10}\left(f(u_\mathrm{\perp} = 0.0,\,u_\mathrm{\Vert})\right)$")
                    ax.set_xlabel(r"$u_\mathrm{\Vert}$")
                elif(ax_flag == "dist_res"):
                    ax.set_ylabel(r"$\log_\mathrm{10}\left(f(u_\mathrm{\perp, res},\,u_\mathrm{\Vert,res})\right)$")
                    ax.set_xlabel(r"$u_\mathrm{\Vert}$")
                elif(ax_flag == "dist"):
                    ax.set_ylabel(r"$u_\Vert$")
                    ax.set_xlabel(r"$u_\perp$")
                elif(ax_flag == "dist_u"):
                    ax.set_ylabel(r"$f$")
                    ax.set_xlabel(r"$u$")
                elif(ax_flag == "dist_E"):
                    ax.set_ylabel(r"$f\,[\si{\per\kilo\electronvolt}]$")
                    ax.set_xlabel(r"$E_\mathrm{kin}\,[\si{\kilo\electronvolt}]$")
                elif(ax_flag == "dist_beta"):
                    ax.set_ylabel(r"$f$")
                    ax.set_xlabel(r"$\beta$")
                elif(ax_flag == "tau_momentum"):
                    ax.set_xlabel(r"$u_\Vert$")
                    ax.set_ylabel(r"$\tau_\omega$")
                elif(ax_flag == "Trad_momentum"):
                    ax.set_xlabel(r"$u_\Vert$")
                    ax.set_ylabel(r"$T_\mathrm{e}$ [\si{\electronvolt}]")
                elif(ax_flag == "kappa"):
                    ax.set_ylabel(r"$\kappa_\mathrm{Refl.}$")
                    ax.set_xlabel(r"$\tau_\omega$")
                elif(ax_flag == "ratio_u_par"):
                    ax.set_ylabel(r"$\frac{\omega_\mathrm{c,obs}}{\omega_\mathrm{c,0}}$")
                    ax.set_xlabel(r"$u_\parallel$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dashed')
                elif(ax_flag == "dist_E"):
                    ax.set_ylabel(r"$\log_\mathrm{10}\left(f(u(E_\mathrm{kin}))\right)$")
                    ax.set_xlabel(r"$E_\mathrm{kin} / \si{\kilo\electronvolt}$")
                elif(ax_flag == "H"):
                    ax.set_ylabel(r"$H / N_\omega$")
                    ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                elif(ax_flag == "freq"):
                    ax.set_ylabel(r"$I_\omega \,\left[\si{\pico\watt\per\square\meter\per\hertz}\right]$")
                    ax.set_xlabel(r"$f_\mathrm{ECE} \,\left[\si{\giga\hertz}\right]$")
                elif(ax_flag == "Rz"):
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    ax.set_ylabel(r"$z \,\left[\si{\meter}\right]$")
                elif(ax_flag == "xy"):
                    ax.set_xlabel(r"$x \,\left[\si{\meter}\right]$")
                    ax.set_ylabel(r"$y \,\left[\si{\meter}\right]$")
                elif(ax_flag == "Rphi"):
                    ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    ax.set_ylabel(r"$\phi \,\left[^{\circ}\right]$")
                elif(ax_flag == "B"):
                    ax.set_ylabel(r"$B\,\left[\si{\tesla}\right]$")
                    ax.set_xlabel(r"$R$ $/$ $\mathrm{m}$")
                    if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
        if(xlabel is not None):
            ax.set_xlabel(xlabel)
        if(ylabel is not None):
            ax.set_ylabel(ylabel)
        ax.set_ylim(y_range[0], y_range[1])
        return ax, y_range

    def read_file(self, filename, maxlines=0, coloumn=1):
        afile = open(filename)
        astring = afile.readlines()
        afile.close()
        x = []
        y = []
        if(maxlines == 0):
            maxlines = len(astring)
        for i in range(maxlines):
            astring[i] = self.clean_string(astring[i])
            try:
                if(not ("NaN" in astring[i].split(" ")[0] or
                    "NaN" in astring[i].split(" ")[coloumn])):
                        x.append(float(astring[i].split(" ")[0]))
                        y.append(float(astring[i].split(" ")[coloumn]))
            except (ValueError, IndexError) as e:
                pass
                # print(e)
        return np.array(x), np.array(y)

    def clean_string(self, i):
        while True:
            if(i.startswith(" ")):
                i = i[1:len(i)]
            elif("  " in i):
                i = i.replace("  ", " ")
            else:
                break
        return i


