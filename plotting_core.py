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
from GlobalSettings import globalsettings
from Diags import Diag
import scipy.constants as cnst
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline, SmoothBivariateSpline
from scipy import stats
from wxEvents import ThreadFinishedEvt, Unbound_EVT_DONE_PLOTTING
import wx
from colorsys import hls_to_rgb
from distribution_functions import Juettner2D
from __builtin__ import False
non_therm_dist_Relax = True
home = '/afs/ipp-garching.mpg.de/home/s/sdenk/'


# Central plotting routine for ECRad and the AECM GUI
# Very much overcomplicated - sorry I did not know any better

class plotting_core:
    def __init__(self, fig, fig_2=None, title=False):
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

    def plot_Trad(self, time, rhop, Trad, Trad_comp, rhop_Te, Te, \
                  diags, diag_names, dstf, model_2=True, \
                  X_mode_fraction=None, X_mode_fraction_comp=None, \
                  multiple_models=False, label_list=None):
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
        if(dstf == "Th"):
            dist_simpl = r"Fa"
            dist = r"Alb"
            dist = r"[" + dist + r"]"
        elif(dstf == "Re"):
            dist_simpl = r"Th"
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
        rhop_max = 0.0
        if(multiple_models):
            mask = np.zeros(len(Trad[0]), dtype=np.bool)
        else:
            mask = np.zeros(len(Trad), dtype=np.bool)
        if(len(diags.keys()) > 0):
            mask[:] = False
            for key in diags.keys():
                # For some reason there is no automatic cast from unicode to string here -> make it explicit
                if(multiple_models):
                    mask[str(diags[key].name)==diag_names[0]] = True
                else:
                    mask[str(diags[key].name)==diag_names] = True
            if(np.all(mask == False)):
                mask[:] = True
        else:
            mask[:] = True
        if(model_2 and len(Trad_comp) > 0):
            if(multiple_models):
                rhop_max = max(np.max(rhop[0][mask]), rhop_max)
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    data=[rhop[0][mask], Trad_comp[mask]], \
                    name=r"$T_" + mathrm + "{rad,mod}" + dist_simpl + r"$", \
                    marker="s", color=(0.0, 0.0, 0.0), \
                    y_range_in=self.y_range_list[0], ax_flag=ax_flag)
            else:
                rhop_max = max(np.max(rhop[mask]), rhop_max)
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
        if(multiple_models):
            self.model_color_index = 0
            for rhop_entry, Trad_entry, diag_name_entry, label in zip(rhop, Trad, diag_names, label_list):
                try:
                    cur_mask = np.zeros(len(Trad_entry), dtype=np.bool)
                    cur_mask[:] = False
                    nice_label = str(label)
                    nice_label = nice_label.replace("_max", "$_\mathrm{max}$")
                    nice_label = nice_label.replace("rad_reac", "rad. reac.")
                    nice_label = nice_label.replace("diff", "$_\mathrm{diff}$")
                    nice_label = nice_label.replace("V_loop", "$V_\mathrm{loop}$")
                    for key in diags.keys():
                        # For some reason there is no automatic cast from unicode to string here -> make it explicit
                        cur_mask[str(diags[key].name)==diag_name_entry] = True
                    if(np.all(cur_mask == False)):
                        cur_mask[:] = True
                    rhop_max = max(np.max(rhop_entry[cur_mask]), rhop_max)
                    self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                                                                         data=[rhop_entry[cur_mask], Trad_entry[cur_mask]], \
                                                                         name=nice_label, \
                                                                         marker="v", color=self.model_colors[self.model_color_index], \
                                                                         y_range_in=self.y_range_list[0], ax_flag=ax_flag)
                    self.model_color_index += 1
                    if(self.model_color_index >= len(self.model_colors)):
                        print("Too many models -> ran out of unique colors")
                except KeyError:
                    print("THe result with the name " + label + "caused an index error")
                    print("Most likely it does not have the correct amount of modeled channels for the currently selected diagnostic")
        else:
            rhop_max = max(np.max(rhop[mask]), rhop_max)
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
        for key in diags.keys():
            if(diags[key].is_prof):
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    data=[diags[key].rhop, diags[key].val],\
                    marker="--", \
                    color="black", \
                    y_range_in=self.y_range_list[0], ax_flag=ax_flag)
            else:
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    data=[diags[key].rhop, diags[key].val], y_error=diags[key].unc, \
                    name=key, marker=self.diag_markers[self.diag_marker_index[0]], \
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
        self.axlist[0].set_xlim(0.0, rhop_max * 1.05)
        self.create_legends("Te_no_ne" + twinx)
        return self.fig

    def plot_tau(self, time, rhop, tau, tau_comp, rhop_IDA, Te_IDA, dstf, model_2, use_tau):
        if(plot_mode == "Software"):
            self.setup_axes("twinx", r"$\tau_{\omega}$, $T_{e}$ ", r"Optical depth $\tau_\omega$")
        else:
            self.setup_axes("twinx", r"$\tau_{\omega}$, $T_\mathrm{e}$", \
                            r"Optical depth $\tau_\omega$")
        mathrm = r"\mathrm"
        if(dstf == "Th"):
            dist_simpl = r"Fa"
            dist = r"Alb"
            dist = r"[" + dist + r"]"
        elif(dstf == "Re"):
            dist_simpl = r"Th"
            dist = r"RELAX"
            dist = r"[" + dist + r"]"
        if(len(dist_simpl) > 0):
            dist_simpl = r"[" + dist_simpl + r"]"
        ax_flag = "T_rho"
        quant_name = "T"
        if(use_tau):
            ax_flag = "tau"
            quant_name = r"\tau"
        if(model_2 and tau_comp is not None):
            if(not use_tau):
                val = np.exp(-tau_comp)
            else:
                val = tau_comp
            self.axlist[0], self.y_range_list[0] = \
                self.add_plot(self.axlist[0], \
                              data=[rhop, val], \
                              name=r"$" + quant_name + "_" + mathrm + "{mod}" + dist_simpl + r"$", \
                              marker="s", color=(126.0 / 255, 0.0, 126.0 / 255), \
                              y_range_in=self.y_range_list[0], ax_flag=ax_flag)
        if(not use_tau):
            val = np.exp(-tau)
        else:
            val = tau
        self.axlist[0], self.y_range_list[0] = \
            self.add_plot(self.axlist[0], \
                          data=[rhop, val], \
                          name=r"$T_" + mathrm + "{mod}" + dist + r"$", \
                          marker="v", color=(126.0 / 255, 126.0 / 255, 0.e0), \
                          y_range_in=self.y_range_list[0], ax_flag=ax_flag)
        self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ rhop_IDA, Te_IDA], \
            name=r"$T_" + mathrm + "{e}$", coloumn=1, marker="-", color=(0.0, 0.0, 0.0), \
                 y_range_in=self.y_range_list[1], y_scale=1.0, ax_flag="Te")  # \times 100$
        self.create_legends("BDP")
        if(len(rhop_IDA) > 0):
            self.axlist[0].set_xlim(0.0, 1.05 * np.max([np.max(rhop_IDA), np.max(rhop)]))
        if(not use_tau):
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
        if(globalsettings.AUG):
            self.plt_vessel(self.axlist[0])
#        steps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
#        steps_y = steps
#        self.axlist[0].get_xaxis().set_major_locator(MaxNLocator(nbins=4, steps=steps, prune='lower'))
#        self.axlist[0].get_xaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps / 4.0))
#        self.axlist[0].get_yaxis().set_major_locator(MaxNLocator(nbins=3, steps=steps_y))
#        self.axlist[0].get_yaxis().set_minor_locator(MaxNLocator(nbins=6, steps=steps_y / 4.0)

    def B_plot(self, Result, itime, ich, mode_str, ray_launch, N_ray):
        self.setup_axes("single", "Frequencies")
        if(N_ray == 1):
            mask = Result.ray["rhop" + mode_str][itime][ich] > 0
            s = Result.ray["s" + mode_str][itime][ich][mask]
            R = np.sqrt(Result.ray["x" + mode_str][itime][ich][mask]**2 + \
                        Result.ray["y" + mode_str][itime][ich][mask]**2)
            freq = np.zeros(len(R))
            freq[:] = ray_launch[itime]["f"][ich] / 1.e9
            f_c_1 = Result.ray["Y" + mode_str][itime][ich][mask] * freq
            f_p = np.sqrt(Result.ray["X" + mode_str][itime][ich][mask]) * freq
        else:
            mask = Result.ray["rhop" + mode_str][itime][ich][0] > 0
            s = Result.ray["s" + mode_str][itime][ich][0][mask]
            R = np.sqrt(Result.ray["x" + mode_str][itime][ich][0][mask]**2 + \
                        Result.ray["y" + mode_str][itime][ich][0][mask]**2)
            freq = np.zeros(len(R))
            freq[:] = ray_launch[itime]["f"][ich] / 1.e9
            f_c_1 = Result.ray["Y" + mode_str][itime][ich][0][mask] * freq
            f_p = np.sqrt(Result.ray["X" + mode_str][itime][ich][0][mask]) * freq
        f_c_2 = f_c_1 * 2.0
        f_c_3 = f_c_1 * 3.0
        f_R = f_c_1 * (np.sqrt(1.0 + 4.0 * (f_p * 2.0 * np.pi) ** 2 / (f_c_1 * 2.0 * np.pi) ** 2) + 1.0) / 2.0
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, f_c_1], \
                    name=r"$f_\mathrm{c}$", marker="--", color=(0.4, 0.4, 0.0), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, f_c_2], \
                name=r"$2 f_\mathrm{c}$", marker="-", color=(0.2, 0.6, 0.0), \
                     y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, f_p], \
                    name=r"$f_\mathrm{p}$", marker=":", color=(0.2, 0.0, 0.6), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, f_c_3], \
                    name=r"$3 f_\mathrm{c}$", marker="--", color=(0.0, 0.4, 0.4), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, freq], \
                    name=r"$f_\mathrm{ECE}$", marker="-", color=(0.0, 0.0, 0.0), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, f_R], \
                    name=r"$f_\mathrm{R}$", marker=":", color=(0.6, 0.0, 0.3), \
                         y_range_in=self.y_range_list[0], ax_flag="f")
        self.model_color_index = 0
        for n, color in zip(range(1, 4),[(0.4, 0.4, 0.0), (0.2, 0.6, 0.0), (0.0, 0.4, 0.4)]):
            R_spl = InterpolatedUnivariateSpline(s, R)
            fc_spl = InterpolatedUnivariateSpline(s, f_c_1 * n - freq)
            s_res = fc_spl.roots()
            if(len(s_res) > 0):
                self.axlist[0].vlines(R_spl(s_res), self.y_range_list[0][0], self.y_range_list[0][1], linestyle='dotted', color=color)
        self.create_legends("single")
        return self.fig

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
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[data_list[i].T[0][data_list[i].T[2] < 10], data_list[i].T[1][data_list[i].T[2] < 10]], \
                y_error=data_list[i].T[1][data_list[i].T[2] < 10] * data_list[i].T[2][data_list[i].T[2] < 10] / 100.e3, \
                name=r"$c$ for \# " + str(shot_list[i]), marker="+", \
                     y_range_in=self.y_range_list[0], y_scale=1.e-3, ax_flag="Calib")
        self.create_legends("errorbar")
        return self.fig, self.fig_2

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

    def diag_calib_avg(self, diag, freq, calib, rel_dev, title):
        self.setup_axes("single", title, r"Rel. mean scatter for diagn. " + diag.name)
        mask = np.abs(rel_dev * 100.0) > 1.0
        print("Not plotting the following channels due to more than 100% statistical uncertainty")
        print(np.array(range(1, len(calib) + 1))[np.logical_not(mask)])
        print("Not plotting the following channels since their calibration factor is more than 20 times larger than median of the calibration factors of all channels")
        print(np.array(range(1, len(calib) + 1))[np.abs(calib) > np.median(np.abs(calib)) * 20])
        mask[np.abs(calib) > np.median(np.abs(calib)) * 20] = False
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            data=[freq[mask], calib[mask]], \
            y_error=calib[mask] * rel_dev[mask] / 100.e0, \
            name=title, marker="+", \
                 y_range_in=self.y_range_list[0], ax_flag="Calib")
        self.create_legends("errorbar")
        self.fig.tight_layout()
        return self.fig, self.fig_2

    def diag_calib_slice(self, diag, freq, calib, std_dev, title):
        self.setup_axes("single", title, r"Rel. mean scatter for diagn. " + diag.name)
        mask = np.abs(std_dev) < calib
        print("Not plotting the following channels due to more than 100% statistical uncertainty")
        print(np.array(range(1, len(calib) + 1))[np.logical_not(mask)])
        print("Not plotting the following channels since their calibration factor is more than 20 times larger than median of the calibration factors of all channels")
        print(np.array(range(1, len(calib) + 1))[np.abs(calib) > np.median(np.abs(calib)) * 20])
        mask[np.abs(calib) > np.median(np.abs(calib)) * 20] = False
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
            data=[freq[mask], calib[mask]], \
            y_error=std_dev[mask], \
            name=title, marker="+", \
                 y_range_in=self.y_range_list[0], ax_flag="Calib")
        self.create_legends("errorbar")
        self.fig.tight_layout()
        return self.fig, self.fig_2


    def calib_evolution(self, diag, ch, ECRad_result_list, heating_array=None, time_ne = None, ne=None):
        self.title = False
        extra_info = heating_array is not None and time_ne is not None and  ne is not None
        plot_power = False
        if(extra_info):
            for i, P_trace in enumerate(heating_array):
                if(np.any(P_trace[1] > 1.e-3)):
                    plot_power = True
        if(extra_info):
            if(plot_power):
                self.setup_axes("twinx_double", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
            else:
                self.setup_axes("twinx_double_single_second", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
        else:
            self.setup_axes("twinx", "\"+\" = $c$, \"-\" = $T_\mathrm{rad,mod}$  " + diag, r"Rel. mean scatter for diagn. " + diag)
        # \"--\" $= T_\mathrm{e}$
        i = 0
        for result in ECRad_result_list:
            if(len(ECRad_result_list) - 1 == 0):
                color = (0.0, 0.0, 1.0)
                label_c = r"$\vert c \vert$"
                label_Trad = r"$T_\mathrm{rad,mod}$"
            else:
                color = self.diag_cmap.to_rgba(float(i) / float(len(ECRad_result_list) - 1))
                label_c = r"$\vert c \vert$" + r" ed {0:d} ch {1:d}".format(result.edition, ch + 1)
                label_Trad = r"$T_\mathrm{rad,mod}$"  + r" ed {0:d} ch {1:d}".format(result.edition, ch + 1)
            Trad = []
            for itime in range(len(result.time)):
                if(result.masked_time_points[diag][itime]):
                    Trad.append(result.Trad[itime][result.Scenario.ray_launch[itime]["diag_name"]  == diag])
            Trad = np.array(Trad)
#            self.axlist[0].plot([], [], color=color, label=r"$T_\mathrm{rad, mod}$" + r" \# {0:d} ed {1:d} ch {2:d}".format(result.Config.shot, result.edition, ch + 1))
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[result.time[result.masked_time_points[diag]], np.abs(result.calib_mat[diag].T[ch])], \
                y_error=result.std_dev_mat[diag].T[ch], \
                name=label_c, marker="+", color=color, y_range_in=self.y_range_list[0], ax_flag="Calib_trace")
            if(len(ECRad_result_list) - 1 == 0):
                color = (0.0, 0.0, 0.0)
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                data=[result.time[result.masked_time_points[diag]], Trad.T[ch]], \
                color=color, marker="--",  name=label_Trad, y_range_in=self.y_range_list[1], ax_flag="Trad_trace")
            if(extra_info):
                heating_labels = [r"$P_\mathrm{ECRH}$", r"$P_\mathrm{NBI}$", r"$P_\mathrm{ICRH}$"]
                heating_color = ["blue", "red", "green"]
                for i, P_trace in enumerate(heating_array):
                    if(np.any(P_trace[1] > 1.e-3)):
                        self.axlist[3], self.y_range_list[3] = self.add_plot(self.axlist[3], \
                                                                             data=[P_trace[0], P_trace[1]], \
                                                                             color=heating_color[i], marker="-",  name=heating_labels[i],\
                                                                             y_range_in=self.y_range_list[3], ax_flag="P_trace")
                self.axlist[2], self.y_range_list[2] = self.add_plot(self.axlist[2], \
                                                                     data=[time_ne, ne / 1.e19], \
                                                                     color="black", marker="--",  name=r"$n_\mathrm{e}$", \
                                                                     y_range_in=self.y_range_list[2], ax_flag="ne_trace")
            i += 1
#         self.axlist[1].set_ylim(0.0, self.y_range_list[1][1])
        self.axlist[0].set_ylim(0.0, self.y_range_list[0][1])
        self.axlist[0].text(0.75, 0.05,  r" \# {0:d}".format(result.Scenario.shot),
                verticalalignment='bottom', horizontalalignment='left',
                transform=self.axlist[0].transAxes,
                color='black', fontsize=plt.rcParams['axes.titlesize'])
        if(extra_info):
            if(plot_power):
                self.create_legends("errorbar_double_twinx")
            else:
                self.create_legends("errorbar_double_twinx_single_second")
            self.axlist[0].get_xaxis().set_visible(False)
        else:
            self.create_legends("errorbar_twinx")
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
            label = r"$V_\mathrm{diag}^*$"
            label_reg = "linear regression"
            if(len(ECRad_result_list) - 1 == 0):
                color = (0.0, 0.0, 1.0)
            else:
                color = self.diag_cmap.to_rgba(float(i) / float(len(ECRad_result_list) - 1))
                label += " ed {0: d} ch. no. {1: d}".format(result.edition, ch + 1)
                label_reg += " ed {0: d} ch. no. {1: d}".format(result.edition, ch + 1)
            Trad = []
            for itime in range(len(result.time)):
                if(result.masked_time_points[diag][itime]):
                    Trad.append(result.Trad[itime][result.Scenario.ray_launch[itime]["diag_name"]  == diag])
            Trad = np.array(Trad)
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                    data=[Trad.T[ch], \
                          diag_data[i]], \
                    y_error=std_dev_data[i], name=label, \
                    marker="+", color=color, \
                    y_range_in=self.y_range_list[0], ax_flag="V_vs_Trad_small", y_scale=1.e3)
            if(pol_angle_list is not None):
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                    data=[Trad.T, pol_angle_list[i][result.masked_time_points[actual_diag]]], \
                    marker="*", color="red", name=r"$\theta_\mathrm{pol}$", \
                    y_range_in=self.y_range_list[1], ax_flag="Ang_vs_Trad")
            Trad_ax = np.linspace(0.0, np.max(Trad.T) * 1.2, 100)
            art_data = Trad_ax * popt_list[i][1] + popt_list[i][0]
            if(len(ECRad_result_list) - 1 == 0):
                color = (0.0, 0.0, 0.0)
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                data=[Trad_ax, art_data], \
                y_range_in=self.y_range_list[0], marker="-", color=color, \
                name=label_reg, ax_flag="V_vs_Trad_small", y_scale=1.e3)
        if(pol_angle_list is not None):
            self.create_legends("errorbar_twinx")
        else:
            self.create_legends("errorbar")
        if(self.y_range_list[0][1] > 0.0):
            self.axlist[0].set_ylim(0.0, self.y_range_list[0][1])
        else:
            self.axlist[0].set_ylim(self.y_range_list[0][0], 0.0)
        self.axlist[0].set_xlim(0.0, np.max(Trad.T) * 1.2)
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

    def plot_BPD(self, time, rhop_signed, D, D_comp, rhop_IDA, Te_IDA, dstf, rhop_cold, scale_w_Trad=False):
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
            name = r"$D_\omega$"
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                self.y_range_list[0] , data=[rhop_signed, D ], color=(0.6, 0.0, 0.0), marker="-", \
                name=name, \
                vline=rhop_cold, ax_flag=ax_flag)

        if(self.title):
            name = r"$D_\omega$, $\rho_\mathrm{pol,res} = " + \
                r"{0:1.2f}".format(rhop_cold) + "$ on " + ch_Hf_str
        else:
            name = r"$D_\omega [" + r"\matrhm{2nd\,model}"  + "$"
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
        if(globalsettings.AUG):
            self.plt_vessel(self.axlist[2])
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
        if(globalsettings.AUG):
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
                    EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, eq_aspect_ratio=True, \
                    vessel_bd=None):
        shotstr = "\#" + str(shot) + " t = " + "{0:2.3f}".format(time) + " s  "
        self.setup_axes("single", r"ECE cold res. and ray " + shotstr, "Toroidal angle")
        if(Rays is not None):
            if(not np.isscalar(Rays[0][0][0])):  # multiple rays stored
                for i in range(len(Rays)):
                    Rays[i] = Rays[i][0]  # Select the central ray
        if(Rays is not None):
            if(tb_Rays is not None or straight_Rays is not None):
                i_axis = len(Rays[0][1])
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ Rays[0][0][:i_axis] , Rays[0][1][:i_axis] ], marker="-", color="blue", \
                      name=r"Rays according to ECRad", ax_flag="Rz")
            else:
                i_axis = len(Rays[0][1])
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[ Rays[0][0][:i_axis] , Rays[0][1][:i_axis] ], marker="-", color="blue", \
                      ax_flag="Rz")
            for ray in Rays[1:len(Rays) - 1]:
                try:
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
        if(equilibrium and EQ_obj):
            R = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].R
            z = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].z
            rhop = EQ_obj.slices[np.argmin(np.abs(EQ_obj.times - time))].rhop
        CS = self.axlist[0].contour(R, z, rhop.T, \
            levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
        CS2 = self.axlist[0].contour(R, z, rhop.T, \
            levels=[1.0], linewidths=3, colors="b", linestyles="-")
        plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS2, inline=1, fontsize=12)
        if(globalsettings.AUG):
            self.plt_vessel(self.axlist[0])
        elif(vessel_bd is not None):
             self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[vessel_bd[0], vessel_bd[1]], marker="-", color=(0.0, 0.0, 0.0), \
                      ax_flag="Rz")
        if(eq_aspect_ratio):
            self.axlist[0].set_aspect("equal")
        if(globalsettings.AUG):
            self.axlist[0].set_xlim((0.8, 2.4))
            self.axlist[0].set_ylim((-0.75, 1.0))
        self.create_legends("vessel")
        return self.fig    

    def plt_vessel(self, ax, pol=True, polygon=False):
        # Stolen from fconf by M. Cavedon -> all praise or blame goes there
        # Currently only AUG vessel supported, but could be extended by generating the corresponding *_vesse_data files
        if(not globalsettings.AUG):
            print("plt_vessel method in plotting_core only supports AUG at this time")
            return
        if polygon:
            from matplotlib.patches import Polygon
        if pol:
            f = open("/afs/ipp-garching.mpg.de/u/sdenk/public/pol_vessel.data", 'r')
        else:
            f = open("/afs/ipp-garching.mpg.de/u/sdenk/public/tor_vessel.data", 'r')
    
        lines = f.readlines()
        f.close()
        vessel = []
        r = []
        z = []
        for line in lines:
            if '#' in line:
                continue
            if line == '\n':
                if len(r) > 0:
                    if pol:
                        vessel.append({'r':r, 'z':z})
                    else:
                        vessel.append({'r':np.array(r) * np.cos(np.deg2rad(-22.5 * 3.)) - np.array(z) * np.sin(np.deg2rad(-22.5 * 3.)), 'z':np.array(r) * np.sin(np.deg2rad(-22.5 * 3.)) + np.array(z) * np.cos(np.deg2rad(-22.5 * 3.)) })
                r = []
                z = []
                continue
            val = line.split()
            r.append(float(val[0]))
            z.append(float(val[1]))
    
        # embed()
    
    
        # if pol==False:
    
            # rotate to new coordinate system
        #    xnew=(np.array(r)*np.cos(np.deg2rad(-22.5*3.))-np.array(z)*np.sin(np.deg2rad(-22.5*3.))).tolist()
        #    ynew = (np.array(r)*np.sin(np.deg2rad(-22.5*3.))+np.array(z)*np.cos(np.deg2rad(-22.5*3.))).tolist()
    
        #    embed()
        #    r=xnew
        #    z=ynew
    
        for key in range(len(vessel)):
            if polygon:
                ax.add_patch(Polygon(zip(vessel[key]['r'], vessel[key]['z']), facecolor='grey', edgecolor='none'))
            else:
                ax.plot(vessel[key]['r'], vessel[key]['z'], 'k-')


    def plot_ray(self, shot, time, ray, index=0, Eq_Slice=None, H=True, R_cold=None, z_cold=None, \
                 s_cold=None, straight=False, eq_aspect_ratio=True, R_other_list=[], z_other_list=[], \
                 x_other_list = [], y_other_list = [], label_list=None, vessel_bd=None):
        self.setup_axes("ray", "Ray", "Hamiltonian")
        # eqi = equ_map()
        equilibrium = True
        if(equilibrium):
            R = Eq_Slice.R
            z = Eq_Slice.z
            rhop = Eq_Slice.rhop
            CS = self.axlist[0].contour(R, z, rhop.T, \
                                        levels=np.linspace(0.1, 1.2, 12), linewidths=1, colors="k", linestyles="--")
            CS2 = self.axlist[0].contour(R, z, rhop.T, \
                levels=[1.0], linewidths=3, colors="b", linestyles="-")
            rhop_spl = RectBivariateSpline(R, z, rhop)
            z_eval = np.zeros(len(R))
            z_eval = Eq_Slice.z_ax
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
            self.axlist[1].add_patch(pltCircle([0.0, 0.0], Eq_Slice.R_ax, edgecolor='b', facecolor='none', linestyle="-"))
            for i in range(len(tor_cont_list)):
                self.axlist[1].add_patch(pltCircle([0.0, 0.0], tor_cont_list[i], edgecolor='k', facecolor='none', linestyle="--"))
        if(R_cold is not None and z_cold is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R_cold, z_cold], \
                name=r"Cold resonance", marker="+", color=(0.e0, 126.0 / 255, 0.e0), \
                     y_range_in=self.y_range_list[0], ax_flag="Rz")
        if(globalsettings.AUG):
            self.plt_vessel(self.axlist[0])
            self.plt_vessel(self.axlist[1], pol=False)
        elif(vessel_bd is not None):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], \
                      self.y_range_list[0], data=[vessel_bd[0], vessel_bd[1]], marker="-", color=(0.0, 0.0, 0.0), \
                      ax_flag="Rz")
            R_vessel_min = np.min(vessel_bd[0])
            phi = np.linspace(0, np.pi * 2.0, 100)
            x_vessel = R_vessel_min * np.cos(phi)
            y_vessel = R_vessel_min * np.sin(phi)
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                      self.y_range_list[1], data=[x_vessel, y_vessel], marker="-", color=(0.0, 0.0, 0.0), \
                      ax_flag="xy")
            R_vessel_max = np.max(vessel_bd[0])
            x_vessel = R_vessel_max * np.cos(phi)
            y_vessel = R_vessel_max * np.sin(phi)
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], \
                      self.y_range_list[1], data=[x_vessel, y_vessel], marker="-", color=(0.0, 0.0, 0.0), \
                      ax_flag="xy")
        if(label_list  is None):
            ECRad_main_label = r"ECRad ray"
        else:
            ECRad_main_label = label_list[0]
        if(np.iterable(ray)):
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray[0].R, ray[0].z], \
                name=ECRad_main_label, marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
                     y_range_in=self.y_range_list[0], ax_flag="Rz")
            if(s_cold is not None):
                x_spl = InterpolatedUnivariateSpline(ray[0].s, ray[0].x)
                y_spl = InterpolatedUnivariateSpline(ray[0].s, ray[0].y)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[x_spl(s_cold), y_spl(s_cold)], \
                            name=r"Cold resonance", marker="+", color=(0.e0, 126.0 / 255, 0.e0), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
            self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[ray[0].x, ray[0].y], \
                        name=ECRad_main_label, marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
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
            if(s_cold is not None and s_cold > 0.0):
                x_spl = InterpolatedUnivariateSpline(ray.s, ray.x)
                y_spl = InterpolatedUnivariateSpline(ray.s, ray.y)
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[x_spl(s_cold), y_spl(s_cold)], \
                            name=r"Cold resonance", marker="+", color=(0.e0, 126.0 / 255, 0.e0), \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
            self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[ray.R, ray.z], \
                    name=ECRad_main_label, marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
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
                        name=ECRad_main_label, marker="-", color=(0.e0, 126.0 / 255, 0.e0), \
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
        if(label_list is not None):
            self.line_color_index[0] = 0
            self.line_color_index[1] = 0
            for R, z, x,y, label in zip(R_other_list, z_other_list, x_other_list, y_other_list, label_list[1:]):
                self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R, z], \
                        name=label, marker="--", color=self.line_colors[self.line_color_index[0]], \
                             y_range_in=self.y_range_list[0], ax_flag="Rz")
                self.axlist[1], self.y_range_list[1] = self.add_plot(self.axlist[1], data=[x, y], \
                            name=label, marker="--", color=self.line_colors[self.line_color_index[1]], \
                                 y_range_in=self.y_range_list[1], ax_flag="xy")
                self.line_color_index[0] += 1
                if(self.line_color_index[0] >= len(self.line_colors)):
                    self.line_color_index[0] = 0
                self.line_color_index[1] += 1
                if(self.line_color_index[1] >= len(self.line_colors)):
                    self.line_color_index[1] = 0
#        if(EQ_obj == False and AUG == True):
#            self.plt_vessel(self.axlist[1], pol=False)
#            self.plt_eq_tor(self.axlist[1], int(shot), float(time))
        self.create_legends("single")
        if(eq_aspect_ratio):
            self.axlist[0].set_aspect("equal")
            self.axlist[1].set_aspect("equal")
        if(globalsettings.AUG):
            self.axlist[0].set_xlim(0.7, 2.5)
            self.axlist[0].set_ylim(-1.75, 1.75)
            self.axlist[1].set_xlim(-2.8, 2.8)
            self.axlist[1].set_ylim(-2.8, 2.8)
        if(H):
            return self.fig, self.fig_2
        else:
            return self.fig

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
        if(globalsettings.AUG):
            self.plt_vessel(self.axlist[0])
            self.plt_vessel(self.axlist_2[0], pol=False)
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
                cmap = plt.cm.ScalarMappable(plt.Normalize(0.0, pw_max / 1.e3), "inferno")
                ray_pw = np.array(ray_pw)
                cmap.set_array(ray_pw)
                multicolor = True
            for ray in beam[::slicing]:
                if(multicolor):
                    pw_ratio = np.max(ray["PW"]) / pw_max
                    print("Max power fraction for ray  no. {0:d}: {1:1.3e}".format((ray_count - 1) * slicing + 1, pw_ratio))
                    if(pw_ratio < 0.01):
                        print("Less than 1% power in ray - skipping.")
                        continue
                    if(not np.all(np.isfinite(ray["PW"]))):
                        print("Ray no. {0:d} has non-finite power".format(ray_count))
                        print("Ray no. {0:d} skipped".format(ray_count))
                        continue
                print("Plotting beam {0:d} ray {1:d}".format(beam_count, (ray_count - 1) * slicing + 1))
                i = 0
                if(not (np.all(np.isfinite(ray["R"])) and np.all (np.isfinite(ray["z"])))):
                    print("Ray no. {0:d} has non-finite coordinates".format(ray_count))
                    print("Ray no. {0:d} skipped".format(ray_count))
                    continue
                x = ray["R"] * np.cos(ray["phi"])
                y = ray["R"] * np.sin(ray["phi"])
                if(multicolor):
                    i_start = 0
                    i_end = 1
                    while(i_end + 1 < len(ray["R"])):
                        while(np.abs((ray["PW"][i_start] - ray["PW"][i_end]) / pw_max) < 0.01 and i_end + 1 < len(ray["R"])):
                            i_end += 1
                        color = cmap.to_rgba((np.max(ray["PW"]) - ray["PW"][i_start]) / 1.e3, pw_ratio)
                        self.axlist[0].plot(ray["R"][i_start:i_end + 1], ray["z"][i_start:i_end + 1], color=color)
                        self.axlist_2[0].plot(x[i_start:i_end + 1], y[i_start:i_end + 1], color=color[0:3], alpha=color[3])
                        i_start = i_end
                else:
                    self.axlist[0].plot(ray["R"], ray["z"], color="b")
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
        if(multicolor):
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
        CS = self.axlist[0].contour(R, z, Psi.T, \
             levels=Psi_grid)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[R_aus, z_aus], \
                        marker="+", color=[1.0, 0, 0], \
                        y_range_in=self.y_range_list[0], ax_flag="Rz")
        self.axlist_2[0], self.y_range_list_2[0] = self.add_plot(self.axlist_2[0], data=[Psi_grid, E_field], \
                        marker="-", color=[0, 0, 1.0], \
                        y_range_in=self.y_range_list_2[0], ax_flag="E_field")
        if(globalsettings.AUG):
            self.plt_vessel(self.axlist[0])
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
        vol_spline = InterpolatedUnivariateSpline(psi_vol, vol)
        norm_R = vol_spline(Psi) / (2.0 * np.pi * R0)
        norm_R[norm_R < 0] = 0
        norm_R = np.sqrt(norm_R)
        self.axlist[0], self.y_range_list[0] = self.add_plot(self.axlist[0], data=[norm_R , Rdiff], \
                        marker="-", color=[1.0, 0, 0], \
                        y_range_in=self.y_range_list[0], ax_flag="Rdiff")
        evt_out = ThreadFinishedEvt(Unbound_EVT_DONE_PLOTTING, Callee.GetId())
        wx.PostEvent(Callee, evt_out)

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
        elif(mode == "twinx_double"):
            self.layout = [2, 2, 1]
            self.grid_locations = [[0, 0], [1, 0]]
            self.twinx_array = [True, True]
            self.x_share_list = [None, 0, 0, 0]
            self.y_share_list = [None, None, None, None]  # Note the twinx axes!
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
        elif(mode == "twinx_double_single_second"):
            self.layout = [2, 2, 1]
            self.grid_locations = [[0, 0], [1, 0]]
            self.twinx_array = [True, False]
            self.x_share_list = [None, 0, 0]
            self.y_share_list = [None, None, None]  # Note the twinx axes!
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
        elif(mode == "errorbar_double_twinx"):
            handles_primary, labels_primary = self.axlist[0].get_legend_handles_labels()
            handles_twinx, labels_twinx = self.axlist[1].get_legend_handles_labels()
            handles = handles_primary + handles_twinx
            labels = labels_primary + labels_twinx
            leg = self.axlist[0].legend(handles, labels)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
            handles_primary, labels_primary = self.axlist[2].get_legend_handles_labels()
            handles_twinx, labels_twinx = self.axlist[3].get_legend_handles_labels()
            handles = handles_primary + handles_twinx
            labels = labels_primary + labels_twinx
            leg2 = self.axlist[2].legend(handles, labels)
            leg2.get_frame().set_alpha(1.0)
            leg2.draggable()
#            lns = self.axlist[0].get_lines()
#            lns = np.concatenate([lns, self.axlist[1].get_lines()])
#            labs = []
#            lns_short = []
#            for l in range(len(lns)):
#                if(not ("_"  in lns[l].get_label() and "$" not in lns[l].get_label())):
#                    labs.append(lns[l].get_label())
#                    lns_short.append(lns[l])
#            leg = self.axlist[0].legend(lns_short, labs)
        elif(mode == "errorbar_double_twinx_single_second"):
            handles_primary, labels_primary = self.axlist[0].get_legend_handles_labels()
            handles_twinx, labels_twinx = self.axlist[1].get_legend_handles_labels()
            handles = handles_primary + handles_twinx
            labels = labels_primary + labels_twinx
            leg = self.axlist[0].legend(handles, labels)
            leg.get_frame().set_alpha(1.0)
            leg.draggable()
            handles, labels = self.axlist[2].get_legend_handles_labels()
            leg2 = self.axlist[2].legend(handles, labels)
            leg2.get_frame().set_alpha(1.0)
            leg2.draggable()
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
            if(len(self.axlist_2) > 0):
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
                mask = np.logical_or(np.abs(y) != np.inf,np.abs(y_error) != np.inf)
                ymax = np.nanmax([y_range_in[1], np.nanmax(y[mask] + y_error[mask])])
                ymin = np.nanmin([y_range_in[0], np.nanmin(y[mask] - y_error[mask])])
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
                    ax.set_ylabel(r"$c$ [keV/(V)]")
                elif(ax_flag == "Calib_trace"):
                    ax.set_xlabel(r"$t$ [s]")
                    ax.set_ylabel(r"$c$ [eV/(V)]")
                elif(ax_flag == "Sig_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"Sig [V]")
                elif(ax_flag == "V_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"$V^*_\mathrm{diag}$ [V]")
                elif(ax_flag == "V_vs_Trad_small"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"$V^*_\mathrm{diag}$ [mV]")
                elif(ax_flag == "Sig_vs_Trad_small"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"Sig [mV]")
                elif(ax_flag == "Sig_vs_time"):
                    ax.set_xlabel(r"$t$ [s]")
                    ax.set_ylabel(r"Sig [V]")
                elif(ax_flag == "Sig_vs_Trad"):
                    ax.set_xlabel(r"Sig [V]")
                    ax.set_ylabel(r"$T_\mathrm{rad,mod}$ [keV]")
                elif(ax_flag == "Ang_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [keV]")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "Ang_vs_Sig"):
                    ax.set_xlabel(r"Sig [V]")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "calib_vs_launch"):
                    ax.set_xlabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                    ax.set_ylabel(r"$c$ [keV/(V)]")
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
                        ax.set_ylabel(r"$c \left[\si{\milli\volt}\right]$")
                        if(vline is not None):
                            if(color is not None):
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted', color=color)
                            else:
                                ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "ne_trace"):
                    ax.set_xlabel(r"$t [\mathrm{s}]$")
                    ax.set_ylabel(r"$n_\mathrm{e}\left[\SI{1.e19}{\per\cubic\metre}\right]$")
                elif(ax_flag == "P_trace"):
                    ax.set_xlabel(r"$t \si{\second}$")
                    ax.set_ylabel(r"$P \left[\si{\mega\watt}\right]$")
                elif(ax_flag == "cnt_trace"):
                    ax.set_xlabel(r"$t [\mathrm{s}]$")
                    ax.set_ylabel(r"cnt rate $  \left[\si{\kilo\hertz}\right]$")
                elif(ax_flag == "Calib"):
                    ax.set_xlabel(r"$f \left[\si{\giga\hertz}\right]$")
                    ax.set_ylabel(r"$c \left[\si{\kilo\electronvolt \per\volt}\right]$")
                elif(ax_flag == "Trad_trace"):
                        ax.set_xlabel(r"$t$ [s]")
                        ax.set_ylabel(r"$T_\mathrm{rad} \left[\si{\kilo\electronvolt}\right]$")
                        if(vline is not None):
                            ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Calib_trace"):
                    ax.set_xlabel(r"$t\,[\mathrm{s}]$")
                    ax.set_ylabel(r"$\vert c \vert \left[\si{\electronvolt \per \volt}\right]$")
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
                    ax.set_ylabel(r"$c \left[\si{\electronvolt \per\volt}\right]$")
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
                elif(ax_flag == "NR"):
                    ax.set_ylabel(r"$N_\omega$")
                    ax.set_xlabel(r"$R$ [\si{\metre}]")
                    # ax.set_xlabel(r"$s \,\left[\si{\meter}\right]$")
                    # ax.set_xlabel(r"$R \,\left[\si{\meter}\right]$")
                    # y_range[1] *= 2.5
                    # ax.set_xlim(min(x),max(x))
                    if(vline is not None):
                        ax.vlines(vline, -10 * np.abs(y_range[0]), 10 * y_range[1], linestyle='dotted')
                elif(ax_flag == "Sig_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                    ax.set_ylabel(r"Sig [\si{\volt}]")
                elif(ax_flag == "Sig_vs_Trad_small"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                    ax.set_ylabel(r"Sig [\si{\milli\volt}]")
                elif(ax_flag == "V_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                    ax.set_ylabel(r"$V^*_\mathrm{diag}$  [\si{\volt}]")
                elif(ax_flag == "V_vs_Trad_small"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                    ax.set_ylabel(r"$V^*_\mathrm{diag}$  [\si{\milli\volt}]")
                elif(ax_flag == "Ang_vs_Trad"):
                    ax.set_xlabel(r"$T_\mathrm{rad,mod}$ \si{\kilo\electronvolt}")
                    ax.set_ylabel(r"$\theta_\mathrm{pol}$ [$^\circ$]")
                elif(ax_flag == "Trad_vs_Sig"):
                    ax.set_xlabel(r"Sig [\si{\volt}]")
                    ax.set_ylabel(r"$T_\mathrm{rad,mod}$ [\si{\kilo\electronvolt}]")
                elif(ax_flag == "Ang_vs_Sig"):
                    ax.set_xlabel(r"Sig [\si{\volt}]")
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


