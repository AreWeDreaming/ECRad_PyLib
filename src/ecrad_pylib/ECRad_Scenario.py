'''
Created on Mar 20, 2019

@author: Severin Denk
'''
from collections import OrderedDict as od
import os
from scipy.io import loadmat
from ecrad_pylib.Global_Settings import globalsettings
import numpy as np
np.set_printoptions(threshold=np.inf)
from ecrad_pylib.Equilibrium_Utils import EQDataExt, EQDataSlice
from scipy.interpolate import InterpolatedUnivariateSpline
from plasma_math_tools.geometry_utils import get_theta_pol_phi_tor_from_two_points
from ecrad_pylib.Diag_Types import CECE_diag, Diag, ECRH_diag, ECI_diag, EXT_diag, CECE_diag
from ecrad_pylib.Distribution_IO import load_f_from_mat
from ecrad_pylib.Distribution_Classes import Distribution, Gene, GeneBiMax
from netCDF4 import Dataset
if(globalsettings.AUG):
    from ecrad_pylib.ECRad_DIAG_AUG import DefaultDiagDict
else:
    from ecrad_pylib.Diag_Types import DefaultDiagDict
# THis class holds all the input data provided to ECRad with the exception of the ECRad configuration


class Plasma(dict):

    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self) -> None:
        super().clear()
        self["eq_dim"] = 2
        self["dist_obj"] = None
        self["GENE_obj"] = None
        self["rhop_prof"] = []
        self["rhot_prof"] = []
        self["prof_reference"] = "rhop_prof"
        self["2D_prof"] = False
        self["vessel_bd"] = None
        self["Te"] = []
        self["ne"] = []
        self["Bt_vac_scale"] = 1.0
        self["eq_data_2D"] = EQDataExt( Ext_data=True)
        self["eq_data_3D"] = {}
        self["eq_data_3D"]["equilibrium_file"] = ""
        self["eq_data_3D"]["equilibrium_type"] = ''
        self["eq_data_3D"]["use_mesh"] = False
        self["eq_data_3D"]["use_symmetry"] = True
        self["eq_data_3D"]["B_ref"] = 1.0
        self["eq_data_3D"]["s_plus"] = 1.0
        self["eq_data_3D"]["s_max"] = 1.2
        self["eq_data_3D"]["interpolation_acc"] = 1.e-12
        self["eq_data_3D"]["fourier_coeff_trunc"] = 1.e-12
        self["eq_data_3D"]["h_mesh"] = 1.5e-2 # meters
        self["eq_data_3D"]["delta_phi_mesh"] = 2.0 # Degrees
        self["eq_data_3D"]["vessel_filename"] = ""

class ECRadScenario(dict):
    def __init__(self, noLoad=False):
        self.scenario_file = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_last_scenario.nc")
        self.reset()
        if(not noLoad):
            try:
                self.load(filename=self.scenario_file)
            except Exception as e:
                print("Failed to import last used Scenario")
                print("Cause: " + str(e))
                self.reset()


    def reset(self):
        self["dimensions"] = {}
        self["dimensions"]["N_time"] = 0
        self["dimensions"]["N_profiles"] = 0
        self["dimensions"]["N_eq_2D_R"] = 0
        self["dimensions"]["N_eq_2D_z"] = 0
        self["dimensions"]["N_vessel_bd"] = 0
        self["dimensions"]["N_vessel_dim"] = 2
        self["dimensions"]["N_ch"] = 0
        self["dimensions"]["N_used_diags"] = 0
        self["time"] = []
        self["shot"] = 0
        self["scaling"] = {}
        self["scaling"]["Bt_vac_scale"] = 1.0
        self["scaling"]["Te_rhop_scale"] = 1.0
        self["scaling"]["ne_rhop_scale"] = 1.0
        self["scaling"]["Te_scale"] = 1.0
        self["scaling"]["ne_scale"] = 1.0
        self["plasma"] = Plasma()
        self["diagnostic"] = {}
        self['diagnostic']["f"] = []
        self['diagnostic']["df"] = []
        self['diagnostic']["R"] = []
        self['diagnostic']["phi"] = []
        self['diagnostic']["z"] = []
        self['diagnostic']["theta_pol"] = []
        self['diagnostic']["phi_tor"] = []
        self['diagnostic']["dist_focus"] = []
        self['diagnostic']["width"] = []
        self['diagnostic']["pol_coeff_X"] = []
        self['diagnostic']["diag_name"] = []
        self["used_diags_dict"] = od()
        self["avail_diags_dict"] = DefaultDiagDict
        self["AUG"] = {}
        if(globalsettings.AUG):
            self["AUG"]["IDA_exp"] = "AUGD"
            self["AUG"]["IDA_ed"] = 0
            self["AUG"]["EQ_exp"] = "AUGD"
            self["AUG"]["EQ_diag"] = "EQH"
            self["AUG"]["EQ_ed"] = 0
            self["plasma"]["Bt_vac_scale"] = 1.005
            self.default_diag = "ECE"
        self.data_source = "Unknown"
        self.default_diag = "EXT"
        # Define a couple of lables used in plotting
        self.labels = {}
        self.labels["Te"] = r"$T_" + globalsettings.mathrm + r"{e}$"
        self.labels["ne"] = r"$n_" + globalsettings.mathrm + r"{e}$"
        self.labels["rhop"] = r"$\rho_" + globalsettings.mathrm + r"{pol}$"
        self.labels["Br"] = r"$B_" + globalsettings.mathrm + r"{r}$"
        self.labels["Bt"] = r"$B_" + globalsettings.mathrm + r"{t}$" 
        self.labels["Bz"] = r"$B_" + globalsettings.mathrm + r"{z}$"
        self.units = {}
        self.units["Te"] = r"[keV]"
        self.units["ne"] = r"$[10^{19}$m$^{-3}$]"
        self.units["rhop"] = r""
        self.units["Br"] = r"[T]"
        self.units["Bt"] = r"[T]"
        self.units["Bz"] = r"[T]"
        self.diags_set = False
        self.plasma_set = False

    def load(self, filename=None, mdict=None, rootgrp=None):
        if(filename is not None):
            ext = os.path.splitext(filename)[1]
            if(ext == ".mat"):
                self.from_mat(path_in=filename)
            elif(ext == ".nc"):
                self.from_netcdf(filename=filename)
            else:
                print("Extension " + ext + " is unknown")
                raise(ValueError)
        elif(mdict is not None):
            self.from_mat(mdict)
        elif(rootgrp is not None):
            self.from_netcdf(rootgrp=rootgrp)

    def set_up_launch_from_imas(self, ece, times=None):
        for key in self['diagnostic']:
            self['diagnostic'][key] = []
        if(times is None):
            times = ece.channel[0].time
            if len(times) == 0:
                times = [0.0]   
        for time in times:
            self['diagnostic']["f"].append([])
            for ch in ece.channel:
                try:
                    itime = np.argmin(np.abs(time - ch.time))
                except ValueError:
                    itime = 0
                self['diagnostic']["f"][-1].append(ch.frequency.data[itime])
        self['diagnostic']["f"] = np.array(self['diagnostic']["f"])
        self["dimensions"]["N_ch"] = len(self["diagnostic"]["f"][0])
        for key in self['diagnostic']:
            if(key == "f"):
                continue
            elif(key == "pol_coeff_X"):
                self['diagnostic'][key] = -np.ones(self['diagnostic']["f"].shape)
            else:
                self['diagnostic'][key] = np.zeros(self['diagnostic']["f"].shape)
            self['diagnostic'][key] = np.zeros(self['diagnostic']["f"].shape)
        self['diagnostic']["R"][:] = ece.line_of_sight.first_point.r
        self['diagnostic']["phi"][:] = np.rad2deg(ece.line_of_sight.first_point.phi)
        self['diagnostic']["z"][:] = ece.line_of_sight.first_point.z
        x1_vec = np.array([
                ece.line_of_sight.first_point.r * np.cos(ece.line_of_sight.first_point.phi),
                ece.line_of_sight.first_point.r * np.sin(ece.line_of_sight.first_point.phi), 
                ece.line_of_sight.first_point.z])
        x2_vec = np.array([
                ece.line_of_sight.second_point.r * np.cos(ece.line_of_sight.second_point.phi),
                ece.line_of_sight.second_point.r * np.sin(ece.line_of_sight.second_point.phi), 
                ece.line_of_sight.second_point.z])
        # Phi is defined as the angle between the k_1 = -r_1 and k_2 = r_2 - r_1
        self['diagnostic']["theta_pol"][:], self['diagnostic']["phi_tor"][:] = \
                get_theta_pol_phi_tor_from_two_points(x1_vec, x2_vec)

    def add_imas_time_slices(self, core_profiles, equilibrium, times):
        for time in times:
            itime_profiles = np.argmin(np.abs(core_profiles.time - time))
            itime_equilibrium = np.argmin(np.abs(equilibrium.time - time))
            self["plasma"]["Te"].append(
                core_profiles.profiles_1d[itime_profiles].electrons.temperature)
            prof_size = len(self["plasma"]["Te"][-1])
            if(prof_size == 0):
                raise ValueError("No profile data!")
            self["plasma"]["ne"].append(
                core_profiles.profiles_1d[itime_profiles].electrons.density)
            if( self["plasma"]["prof_reference"] == "rhop_prof"):
                try:
                    self["plasma"]["rhop_prof"].append(
                        np.sqrt((equilibrium.time_slice[itime_equilibrium].global_quantities.psi_axis - \
                                core_profiles.profiles_1d[itime_profiles].grid.psi) /\
                                (equilibrium.time_slice[itime_equilibrium].global_quantities.psi_axis - \
                                equilibrium.time_slice[itime_equilibrium].global_quantities.psi_boundary)))
                    if(np.any(np.isnan(self["plasma"]["rhop_prof"][-1]))):
                        raise ValueError("psi not properly set up")
                    if(len(self["plasma"]["rhop_prof"][-1]) != prof_size):
                        raise ValueError("Wrong size of grid")
                    self["plasma"]["rhop_prof"] = np.array(self["plasma"]["rhop_prof"])
                except Exception:
                    self["plasma"]["rhop_prof"].append(
                        core_profiles.profiles_1d[itime_profiles].grid.rho_pol_norm)
                    if(np.any(np.isnan(self["plasma"]["rhop_prof"][-1]))):
                        raise ValueError("psi not properly set up")
                    if(len(self["plasma"]["rhop_prof"][-1]) != prof_size):
                        raise ValueError("Wrong size of grid")
            else:
                
                self["plasma"]["rhot_prof"].append(
                    core_profiles.profiles_1d[itime_profiles].grid.rho_tor_norm)
                if(np.any(np.isnan(self["plasma"]["rhot_prof"][-1]))):
                    raise ValueError("psi not properly set up")
                if(len(self["plasma"]["rhot_prof"][-1]) != prof_size):
                    raise ValueError("Wrong size of grid")
                mask = equilibrium.time_slice[itime_equilibrium].profiles_1d.rho_tor_norm < 1.0
                rhot_to_psi_spl = InterpolatedUnivariateSpline(equilibrium.time_slice[itime_equilibrium].profiles_1d.rho_tor_norm[mask], \
                        equilibrium.time_slice[itime_equilibrium].profiles_1d.psi[mask])
                psi = rhot_to_psi_spl(self["plasma"]["rhot_prof"])
                rhop = np.sqrt((equilibrium.time_slice[itime_equilibrium].global_quantities.psi_axis - 
                                psi) /
                                (equilibrium.time_slice[itime_equilibrium].global_quantities.psi_axis - 
                                equilibrium.time_slice[itime_equilibrium].global_quantities.psi_boundary))
                self["plasma"]["rhop_prof"].append(
                    core_profiles.profiles_1d[itime_profiles].grid.rho_tor_norm)

    def set_up_profiles_from_imas(self, core_profiles, equilibrium, times):
        # Restore initial state
        self["plasma"].clear()
        try:
            self.add_imas_time_slices(core_profiles, equilibrium, times)
        except Exception:
            if(self["plasma"]["prof_reference"] == "rhop_prof"):
                # Reset the plasma
                self["plasma"].clear()
                self["plasma"]["prof_reference"] = "rhot_prof"
                print("rho_pol not viable here using rho_tor")
                self.add_imas_time_slices(core_profiles, equilibrium, times)
                self["plasma"][self["plasma"]["prof_reference"]] = \
                        np.array(self["plasma"][self["plasma"]["prof_reference"]])
                # ECRad cannot work with rho_tor for 2D equilbria
                # Hence, we need to get rho_pol profiles
                self["plasma"]["prof_reference"] = "rhop_prof"
        self["plasma"]["Te"] = np.array(self["plasma"]["Te"])
        self["plasma"]["ne"] = np.array(self["plasma"]["ne"])
        self["plasma"][self["plasma"]["prof_reference"]] = \
                np.array(self["plasma"][self["plasma"]["prof_reference"]])

    def set_up_equilibrium_from_imas(self, equilibrium, wall, times):
        self["plasma"]["eq_dim"] = True
        self["plasma"]["eq_data_2D"] = EQDataExt(Ext_data=True)
        EQ_slices = []
        for time in times:
            itime = np.argmin(np.abs(equilibrium.time - time))
            rect_index = -1
            for grid_index in range(len(equilibrium.time_slice[itime].profiles_2d)):
                if(equilibrium.time_slice[itime].profiles_2d[grid_index].grid_type.index == 1):
                    rect_index = grid_index
                    break
            if(rect_index == -1):
                print("Failed to find a rectangular psi grid. The following grids are available:")
                for grid_index in range(len(equilibrium.time_slice[itime].profiles_2d)):
                    print(f"Grid type of grid {grid_index} is {equilibrium.time_slice[itime].profiles_2d[grid_index].grid_type.index}")
                print("Trying the first one and praying")
                grid_index = 0
            prof_2D = equilibrium.time_slice[itime].profiles_2d[0]
            EQ_slices.append(EQDataSlice(\
                time, prof_2D.grid.dim1,prof_2D.grid.dim2,\
                prof_2D.psi, prof_2D.b_field_r, prof_2D.b_field_tor, prof_2D.b_field_z,
                equilibrium.time_slice[itime].global_quantities.psi_axis,\
                equilibrium.time_slice[itime].global_quantities.psi_boundary,\
                equilibrium.time_slice[itime].global_quantities.magnetic_axis.r,\
                equilibrium.time_slice[itime].global_quantities.magnetic_axis.z))
        self["plasma"]["eq_data_2D"].set_slices_from_ext(times, EQ_slices)
        r = np.array(wall.description_2d[0].limiter.unit[0].outline.r)
        z = np.array(wall.description_2d[0].limiter.unit[0].outline.z)
        for i in range(1, len(wall.description_2d[0].limiter.unit)):
            r_append = wall.description_2d[0].limiter.unit[i].outline.r
            z_append = wall.description_2d[0].limiter.unit[i].outline.z
            if (( r[-1] - r_append[0])**2 + (z[-1] - z_append[0])**2
                < (r[-1] - r_append[-1])**2 + (z[-1] - z_append[-1])**2):
                r = np.concatenate([r, r_append])
                z = np.concatenate([z, z_append])
            else:
                r = np.concatenate([r, r_append[::-1]])
                z = np.concatenate([z, z_append[::-1]])
        self["plasma"]["vessel_bd"] = np.array([r,z]).T
            

    def set_up_from_imas(self, equilibrium_ids, profile_ids, ece_ids, wall_ids, times):
        self.set_up_launch_from_imas(ece_ids, times)
        self.set_up_equilibrium_from_imas(equilibrium_ids, wall_ids, times)
        self.set_up_profiles_from_imas(profile_ids, equilibrium_ids, times)
        self.set_up_dimensions()

    def set_up_launch_from_omas(self, ods):
        for key in self['diagnostic']:
            self['diagnostic'][key] = []
        self["dimensions"]["N_ch"] = len(ods['ece']['channel'])
        N_time = len(ods['ece.time'])
        self['diagnostic']["f"] = np.zeros((N_time, self["dimensions"]["N_ch"]))
        for ich, ch in enumerate(ods['ece']['channel'].values()):
            self['diagnostic']["f"][...,ich] = ch['frequency.data']
        for key in self['diagnostic']:
            if(key == "f"):
                continue
            elif(key == "pol_coeff_X"):
                self['diagnostic'][key] = -np.ones(self['diagnostic']["f"].shape)
            elif(key == "diag_name"):
                name_first = ods['ece']['channel[0].name']
                name_last = ods['ece']['channel[{0:d}].name'.format(self["dimensions"]["N_ch"] - 1)]
                i = 0
                while i < min(len(name_first), len(name_last)):
                    if(name_first[i] == name_last[i]):
                        i += 1
                    else:
                        break
                self['diagnostic'][key] = np.zeros(self['diagnostic']["f"].shape, dtype="|S{0:d}".format(i))
                self['diagnostic'][key][:] = name_first[:i]
            else:
                self['diagnostic'][key] = np.zeros(self['diagnostic']["f"].shape)
        self['diagnostic']["R"][:] = ods['ece']['line_of_sight']['first_point']["r"]
        self['diagnostic']["phi"][:] = np.rad2deg(ods['ece']['line_of_sight']['first_point']["phi"])
        self['diagnostic']["z"][:] = ods['ece']['line_of_sight']['first_point']["z"]
        x1_vec = np.array([
                ods['ece']['line_of_sight']['first_point']["r"] * np.cos(ods['ece']['line_of_sight']['first_point']["phi"]),
                ods['ece']['line_of_sight']['first_point']["r"] * np.sin(ods['ece']['line_of_sight']['first_point']["phi"]), 
                ods['ece']['line_of_sight']['first_point']["z"]])
        x2_vec = np.array([
                ods['ece']['line_of_sight']['second_point']["r"] * np.cos(ods['ece']['line_of_sight']['second_point']["phi"]),
                ods['ece']['line_of_sight']['second_point']["r"] * np.sin(ods['ece']['line_of_sight']['second_point']["phi"]),
                ods['ece']['line_of_sight']['second_point']["z"]])
        self['diagnostic']["theta_pol"][:], self['diagnostic']["phi_tor"][:] = \
                get_theta_pol_phi_tor_from_two_points(x1_vec, x2_vec)

    def set_up_profiles_from_omas(self, ods, times):
        self["plasma"].clear()
        self["plasma"]["prof_reference"] = "rhop_prof"
        for time in times:
            itime_profiles = np.argmin(np.abs(ods['core_profiles']['time'] - time))
            itime_equilibrium = np.argmin(np.abs(ods['equilibrium']['time'] - time))
            try:
                rhp_pol = ods['core_profiles']['profiles_1d'][itime_profiles]['grid']['rho_pol_norm']
            except ValueError:
                try:
                    rhp_pol = np.sqrt((ods['equilibrium']['time_slice'][itime_equilibrium]['global_quantities']['psi_axis'] - \
                                ods['core_profiles']['profiles_1d'][itime_profiles]['grid']['psi']) /\
                                (ods['equilibrium']['time_slice'][itime_equilibrium]['global_quantities']['psi_axis'] - \
                                ods['equilibrium']['time_slice'][itime_equilibrium]['global_quantities']['psi_boundary']))
                except ValueError:
                    rhp_pol = ods.physics_remap_flux_coordinates(itime_equilibrium, "rho_tor_norm", 
                                                       "rho_pol_norm", ods['core_profiles']['profiles_1d'][itime_profiles]['grid']['rho_tor_norm'])
            mask = np.logical_not(np.isnan(rhp_pol))
            self["plasma"]["rhop_prof"].append(rhp_pol[mask])
            self["plasma"]["Te"].append(
                ods['core_profiles']['profiles_1d'][itime_profiles]['electrons']['temperature'][mask])
            try:
                self["plasma"]["ne"].append(
                    ods['core_profiles']['profiles_1d'][itime_profiles]['electrons']['density'][mask])
            except ValueError:
                self["plasma"]["ne"].append(
                    ods['core_profiles']['profiles_1d'][itime_profiles]['electrons']['density_thermal'][mask])
        self["plasma"]["rhop_prof"] = np.array(self["plasma"]["rhop_prof"])
        self["plasma"]["Te"] = np.array(self["plasma"]["Te"])
        self["plasma"]["ne"] = np.array(self["plasma"]["ne"])
        confined = self["plasma"]["rhop_prof"] < 1.0
        if np.any(np.logical_or(np.isnan(self["plasma"]["Te"][confined]),
                  np.isnan(self["plasma"]["ne"][confined]))):
            raise ValueError("There are NaNs in the profile data inside the confined region")
        self["plasma"]["Te"][np.isnan(self["plasma"]["Te"])] = 20.e-3
        self["plasma"]["ne"][np.isnan(self["plasma"]["ne"])] = 1.e15

    def set_up_equilibrium_from_omas(self, ods, times):
        self["plasma"]["eq_dim"] = True
        self["plasma"]["eq_data_2D"] = EQDataExt(Ext_data=True)
        EQ_slices = []
        for time in times:
            itime = np.argmin(np.abs(ods['equilibrium']['time'] - time))
            prof_2D = ods['equilibrium']['time_slice'][itime]['profiles_2d.0']
            try:
                r_ax = ods['equilibrium']['time_slice'][itime]['global_quantities']['magnetic_axis']['r']
                z_ax = ods['equilibrium']['time_slice'][itime]['global_quantities']['magnetic_axis']['z']
            except ValueError:
                r_ax = None
                z_ax = None
            EQ_slices.append(EQDataSlice(\
                time, prof_2D["grid"]["dim1"],prof_2D["grid"]["dim2"],\
                prof_2D["psi"], prof_2D["b_field_r"], prof_2D["b_field_tor"], prof_2D["b_field_z"],
                ods['equilibrium']['time_slice'][itime]['global_quantities']['psi_axis'],\
                ods['equilibrium']['time_slice'][itime]['global_quantities']['psi_boundary'],\
                r_ax,\
                z_ax))
        self["plasma"]["eq_data_2D"].set_slices_from_ext(times, EQ_slices)
        self["plasma"]["vessel_bd"] = np.array([ \
            ods['wall.description_2d.0.limiter.unit.0.outline.r'],
            ods['wall.description_2d.0.limiter.unit.0.outline.z']]).T

    def set_up_from_omas(self, ods, times):
        self.set_up_launch_from_omas(ods, times)
        self.set_up_equilibrium_from_omas(ods, times)
        self.set_up_profiles_from_omas(ods, times)
        self.set_up_dimensions()    

    def set_up_dimensions(self):
        self["dimensions"]["N_time"] = len(self["time"])
        if(not self["plasma"]["2D_prof"]):
            self["dimensions"]["N_profiles"] = len(self["plasma"][self["plasma"]["prof_reference"]][0])
        if(self["plasma"]["eq_dim"] == 2):
            self["dimensions"]["N_eq_2D_R"] = self["plasma"]["eq_data_2D"].eq_shape[0]
            self["dimensions"]["N_eq_2D_z"] = self["plasma"]["eq_data_2D"].eq_shape[1]
        self["dimensions"]["N_vessel_bd"] = len(self["plasma"]["vessel_bd"])
        self["dimensions"]["N_vessel_dim"] = 2
        self["dimensions"]["N_ch"] = len(self["diagnostic"]["f"][0])
        self["N_used_diags"] = len(list(self["used_diags_dict"].keys()))
        

    def drop_time_point(self, itime):
        time = self["time"][itime]
        self["time"] = np.delete(self["time"], itime)
        for sub_key in ["rhop_prof", "rhot_prof", "Te", "ne"]:
            if(len(self["plasma"][sub_key]) == 0):
                continue
            self["plasma"][sub_key] = np.delete(self["plasma"][sub_key], itime, 0)
        if(self["plasma"]["eq_dim"] == 2):
            self["plasma"]["eq_data_2D"].RemoveSlice(time)
        for sub_key in self["diagnostic"].keys():
            self["diagnostic"][sub_key] = np.delete(self["diagnostic"][sub_key], itime, 0) 
        self["dimensions"]["N_time"] -= 1    

    def from_mat(self, mdict=None, path_in=None, load_plasma_dict=True):
        self.reset()
        if(mdict is None):
            if(path_in is None):
                filename = os.path.join(os.path.expanduser("~"), ".ECRad_GUI_last_scenario.mat")
            else:
                filename = path_in
            try:
                mdict = loadmat(filename, chars_as_strings=True, squeeze_me=True, uint16_codec='ascii')
            except IOError as e:
                print(e)
                print("Error: " + filename + " does not exist")
                return False
            except TypeError as e:
                print(e)
                print("Error: File appears to be corrupted does not exist")
                return False
        increase_time_dim = False
        if(np.isscalar(mdict["time"])):
            self["time"] = np.array([mdict["time"]])
            increase_time_dim = True
        else:
            self["time"] = mdict["time"]
        try:
            self["plasma"]["2D_prof"] = mdict["profile_dimension"] > 1
        except KeyError:
            if(increase_time_dim):
                self["plasma"]["2D_prof"] = mdict["Te"].ndim > 1
            else:
                self["plasma"]["2D_prof"] = mdict["Te"][0].ndim > 1
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["diag", "time", "Diags_exp", "Diags_diag", "Diags_ed", "Extra_arg_1", "Extra_arg_2", "Extra_arg_3", \
                            "used_diags"]
        at_least_2d_keys = ["eq_R", "eq_z", "diag_name", "launch_f", "launch_df", "launch_R", "launch_phi", \
                             "launch_z", "launch_tor_ang" , "launch_pol_ang", "launch_dist_focus", "launch_diag_name", \
                             "launch_width", "launch_pol_coeff_X", "eq_special", "eq_special_complete"  ]
        at_least_3d_keys = ["eq_Psi", "eq_rhop", "eq_Br", "eq_Bt", "eq_Bz"]
        if(self["plasma"]["2D_prof"]  == 2):
            for key in ["Te", "ne"]:
                at_least_3d_keys.append(key)
        else:
            for key in ["rhop_prof", "Te", "ne",  "rhot_prof"]:
                at_least_2d_keys.append(key)
        for key in mdict:
            if(not key.startswith("_")):  # throw out the .mat specific information
                try:
                    if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                        mdict[key] = np.atleast_1d(mdict[key])
                    elif(key in at_least_2d_keys):
                        mdict[key] = np.atleast_2d(mdict[key])
                    elif(key in at_least_3d_keys):
                        if(increase_time_dim):
                            mdict[key] = np.array([mdict[key]])
                except Exception as e:
                    print(key)
                    print(e)
        self["shot"] = mdict["shot"]
        if(globalsettings.AUG):
            try:
                self["AUG"]["IDA_exp"] = mdict["IDA_exp"]
                self["AUG"]["IDA_ed"] = mdict["IDA_ed"]
                self["AUG"]["EQ_exp"] = mdict["EQ_exp"]
                self["AUG"]["EQ_diag"] = mdict["EQ_diag"]
                self["AUG"]["EQ_ed"] = mdict["EQ_ed"]
            except KeyError:
                print("WARNING:: Failed to load AUG specific keywords from Scenarios.")
                print("WARNING:: Replacing them with default values.")
                self["AUG"]["IDA_exp"] = "AUGD"
                self["AUG"]["IDA_ed"] = 0
                self["AUG"]["EQ_exp"] = "AUGD"
                self["AUG"]["EQ_diag"] = "EQH"
                self["AUG"]["EQ_ed"] = 0
            self["used_diags_dict"] = od()
        self.default_diag = mdict["used_diags"][0]
        self["dimensions"]["N_used_diags"] = len(mdict["used_diags"])
        for i in range(len(mdict["used_diags"])):
            diagname = mdict["used_diags"][i]
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI")): #and globalsettings.AUG):
                self["used_diags_dict"].update({diagname: ECI_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                                                   mdict["Extra_arg_1"][i], mdict["Extra_arg_2"][i], int(mdict["Extra_arg_3"][i]))})
            elif("CT" in diagname or "IEC" in diagname):
                try:
                    extra_arg_3 = bool(int(mdict["Extra_arg_3"][i]))
                except ValueError:
                    extra_arg_3 =  mdict["Extra_arg_3"][i] == "True"
                    if(not extra_arg_3):
                        extra_arg_3 =  mdict["Extra_arg_3"][i] == "None"
                self["used_diags_dict"].update({diagname: ECRH_diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]), \
                                              int(mdict["Extra_arg_1"][i]), float(mdict["Extra_arg_2"][i]), extra_arg_3)})
            elif(diagname == "EXT"):
                if("Ext_launch_pol" in mdict):
                    self["used_diags_dict"].update({diagname: EXT_diag(diagname, mdict["Ext_launch_geo"], mdict["Ext_launch_pol"])})
                else:
                    self["used_diags_dict"].update({diagname: EXT_diag(diagname, mdict["Ext_launch_geo"], -1)})
            else:#if(globalsettings.AUG):
                self["used_diags_dict"].update({diagname: \
                        Diag(diagname, mdict["Diags_exp"][i], mdict["Diags_diag"][i], int(mdict["Diags_ed"][i]))})
            for diag_key in self["avail_diags_dict"]:
                if(diag_key in list(self["used_diags_dict"])):
                    self["avail_diags_dict"].update({diag_key: self["used_diags_dict"][diag_key]})
        try:
            self["scaling"]["Bt_vac_scale"] = mdict["bt_vac_correction"]
        except KeyError:
            self["scaling"]["Bt_vac_scale"] = 1.005
        try:
            self["scaling"]["Te_rhop_scale"] = mdict["Te_rhop_scale"]
            self["scaling"]["ne_rhop_scale"] = mdict["ne_rhop_scale"]
        except KeyError:
            self["scaling"]["Te_rhop_scale"] = mdict["Te_rhop_scale"]
            self["scaling"]["ne_rhop_scale"] = mdict["ne_rhop_scale"]
        try:
            self["scaling"]["Te_scale"] = mdict["Te_scale"]
            self["scaling"]["ne_scale"] = mdict["ne_scale"]
        except KeyError:
            self["scaling"]["Te_scale"] = 1.0
            self["scaling"]["ne_scale"] = 1.0
        if("launch_R" in mdict):
            self['dimensions']["N_ch"] = len(mdict["launch_f"][0])
            self['diagnostic']["f"] = mdict["launch_f"]
            self['diagnostic']["df"] = mdict["launch_df"]
            self['diagnostic']["R"] = mdict["launch_R"]
            self['diagnostic']["phi"] = mdict["launch_phi"]
            self['diagnostic']["z"] = mdict["launch_z"]
            self['diagnostic']["theta_pol"] = mdict["launch_pol_ang"]
            self['diagnostic']["phi_tor"] = mdict["launch_tor_ang"]
            self['diagnostic']["dist_focus"] = mdict["launch_dist_focus"]
            self['diagnostic']["width"] = mdict["launch_width"]
            self['diagnostic']["pol_coeff_X"] = mdict["launch_pol_coeff_X"]
            try:
                self['diagnostic']["diag_name"] = mdict["launch_diag_name"]
            except:
                self['diagnostic']["diag_name"] = mdict["diag_name"]
            self.diags_set = True
        else:
            print("No launch information in this save file, you will have to reload the diagnostic info from the AUG database")
        if(not load_plasma_dict):
            return
        try:
            if(bool(mdict["Use_3D_used"])):
                self["plasma"]["eq_dim"] = 3
            else:
                self["plasma"]["eq_dim"] = 2
        except KeyError:
            self["plasma"]["eq_dim"] = 2
        if(self["plasma"]["2D_prof"]):
            self["plasma"]["prof_reference"] = "2D"
        else:            
            try:
                self["plasma"]["rhot_prof"] = mdict["rhot_prof"]
            except KeyError:
                print("Could not find rho_tor profile")
                if(self["plasma"]["eq_dim"] == 3):
                    print("INFO: 3D Scenario identified")
                    print("INFO: Overriding rhot_prof with rhop_prof")
                    self["plasma"]["rhot_prof"] = mdict["rhop_prof"]
            if(self["plasma"]["eq_dim"] != 3):
                self["plasma"]["rhop_prof"] = mdict["rhop_prof"]
            try:
                self["plasma"]["prof_reference"] = mdict["prof_reference"]
            except KeyError:
                if(self["plasma"]["eq_dim"] == 3):
                    self["plasma"]["prof_reference"] = "rhot_prof"
                else:
                    print("INFO: Could not find profile axis type. Falling  back to rho_pol")
                    self["plasma"]["prof_reference"] = "rhop_prof"
        self["plasma"]["Te"] = mdict["Te"]
        self["plasma"]["ne"] = mdict["ne"]
        if(self["plasma"]["eq_dim"] == 2):
            self["plasma"]["eq_data_2D"] = EQDataExt(Ext_data=True)
            slices = []
            for i in range(len(self["time"])):
                if("eq_special_complete" in mdict):
                    entry = mdict["eq_special_complete"][i]
                    slices.append(EQDataSlice(self["time"][i], \
                                              mdict["eq_R"][i], mdict["eq_z"][i], \
                                              mdict["eq_Psi"][i], mdict["eq_Br"][i], \
                                              mdict["eq_Bt"][i], mdict["eq_Bz"][i], \
                                              Psi_ax=entry[4], Psi_sep=entry[5], \
                                              R_ax=entry[0], z_ax=entry[1], \
                                              rhop=mdict["eq_rhop"][i]))
                else:
                    entry = mdict["eq_special"][i]
                    slices.append(EQDataSlice(self["time"][i], \
                                              mdict["eq_R"][i], mdict["eq_z"][i], \
                                              mdict["eq_Psi"][i], mdict["eq_Br"][i], \
                                              mdict["eq_Bt"][i], mdict["eq_Bz"][i], \
                                              Psi_ax=entry[0], Psi_sep=entry[1], \
                                              rhop=mdict["eq_rhop"][i]))
                # The old .mat file store the scaled Bt not the original Bt
                # In the new netcdf files the original Bt is stored
                # The scaled Bt is only used directly in ECRad
                EQobj = EQDataExt(Ext_data=True)
                EQobj.set_slices_from_ext([self["time"][i]], [slices[-1]])
                slices[-1].R_ax, slices[-1].z_ax = EQobj.get_axis(self["time"][i])
                slices[-1].Bt = self["plasma"]["eq_data_2D"].adjust_external_Bt_vac(slices[-1].Bt, slices[-1].R, \
                                                                                    slices[-1].R_ax, 1.0/self["scaling"]["Bt_vac_scale"])
            self["plasma"]["eq_data_2D"].set_slices_from_ext(self["time"], slices)
            self["plasma"]["vessel_bd"] = mdict["vessel_bd"].T
            self["dimensions"]["N_eq_2D_R"] = len(mdict["eq_R"][0])
            self["dimensions"]["N_eq_2D_z"] = len(mdict["eq_z"][0])
            self["dimensions"]["N_vessel_bd"] = len(mdict["vessel_bd"].T)
        else:
            self["plasma"]["eq_data_3D"]["equilibrium_files"] = np.array([mdict["Use_3D_" + "equilibrium_file"]])
            self["plasma"]["eq_data_3D"]["equilibrium_type"] = mdict["Use_3D_" + "equilibrium_type"]
            self["plasma"]["eq_data_3D"]["use_mesh"] = bool(mdict["Use_3D_" + "use_mesh"])
            self["plasma"]["eq_data_3D"]["use_symmetry"] = bool(mdict["Use_3D_" + "use_symmetry"])
            self["plasma"]["eq_data_3D"]["B_ref"] = mdict["Use_3D_" + "B_ref"]
            self["plasma"]["eq_data_3D"]["s_plus"] = mdict["Use_3D_" + "s_plus"]
            self["plasma"]["eq_data_3D"]["s_max"] = mdict["Use_3D_" + "s_max"]
            self["plasma"]["eq_data_3D"]["interpolation_acc"] = mdict["Use_3D_" + "interpolation_acc"]
            self["plasma"]["eq_data_3D"]["fourier_coeff_trunc"] = mdict["Use_3D_" + "fourier_coeff_trunc"]
            self["plasma"]["eq_data_3D"]["h_mesh"] = mdict["Use_3D_" + "h_mesh"]
            self["plasma"]["eq_data_3D"]["delta_phi_mesh"] = mdict["Use_3D_" + "delta_phi_mesh"]
            self["plasma"]["eq_data_3D"]["vessel_filename"] = mdict["Use_3D_" + "vessel_filename"]
            if(self["plasma"]["rhot_prof"] is None):
                print("For 3D calculations the rho toroidal is obligatory")
                raise ValueError("Failed to load equilibrium")
            else:
                self["plasma"]["prof_reference"] = "rhot_prof"
        if("data_source" in mdict):
            self.data_source = mdict["data_source"]
        else:
            self.data_source = "Unknown"
        self.plasma_set = True
        self["dimensions"]["N_time"] = len(self["time"])
        if(not self["plasma"]["2D_prof"]):
            self["dimensions"]["N_profiles"] = len(self["plasma"]["Te"][0])
        try:        
            self.load_dist_obj_from_mat(mdict=mdict)
        except Exception:
            self["plasma"]["dist_obj"] = None
            print("No distribution data in current Scenario")

    def to_netcdf(self, filename=None, rootgrp=None):
        if(filename is not None):
            rootgrp = Dataset(filename, "w", format="NETCDF4")
        rootgrp.createGroup("Scenario")
        rootgrp["Scenario"].createDimension('str_dim', 1)
        for sub_key in self["dimensions"].keys():
            rootgrp["Scenario"].createDimension(sub_key, self["dimensions"][sub_key])
        var = rootgrp["Scenario"].createVariable("plasma" + "_" + "2D_prof", "b")
        var[...] = int(self["plasma"]["2D_prof"])
        var = rootgrp["Scenario"].createVariable("time", "f8", ("N_time",))
        var[...] = self["time"]
        var = rootgrp["Scenario"].createVariable("shot", "i8")
        var[...] = self["shot"]
        for sub_key in ["Te", "ne"]:
            if(not self["plasma"]["2D_prof"]):
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + sub_key, \
                                                         "f8", ("N_time", "N_profiles"))
            else:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + sub_key, "f8", \
                                                         ("N_time", "N_eq_2D_R", "N_eq_2D_z"))
            var[:] = self['plasma'][sub_key]
        if(not self["plasma"]["2D_prof"]):
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + "prof_reference", str, ('str_dim',))
            var[0] = self["plasma"]["prof_reference"]
            sub_key = self["plasma"]["prof_reference"]
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + sub_key, "f8", \
                                                     ("N_time", "N_profiles"))
            var[:] = self["plasma"][sub_key]
        for sub_key in self["scaling"].keys():
            var = rootgrp["Scenario"].createVariable("scaling"+ "_" + sub_key, "f8")
            var[...] = self["scaling"][sub_key]
        var = rootgrp["Scenario"].createVariable("plasma" + "_" + "eq_dim", "i8")
        var[...] = self["plasma"]["eq_dim"]

        if(self["plasma"]["eq_dim"] == 3):
            for sub_key in ["B_ref", "s_plus", "s_max", \
                           "interpolation_acc", "fourier_coeff_trunc", \
                           "h_mesh", "delta_phi_mesh"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_3D" + "_" +  sub_key, "f8")
                var[...] = self["plasma"]["eq_data_3D"][sub_key]
            for sub_key in ["use_mesh", "use_symmetry"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_3D" + "_" +  sub_key, "b")
                var[...] = self["plasma"]["eq_data_3D"][sub_key]
            for sub_key in ["equilibrium_type", "vessel_filename"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_3D" + "_" +  sub_key, str, ('str_dim',))
                var[0] = self["plasma"]["eq_data_3D"][sub_key]
            var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                     "eq_data_3D" + "_" + \
                                                     "equilibrium_files", str, ('N_time',))
            var[:] = self["plasma"]["eq_data_3D"]["equilibrium_files"]
        else:
            # print('else')
            for sub_key in ["R", "z"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key, "f8", \
                                                         ("N_time", "N_eq_2D_" + sub_key))
                var[:] = self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key)

            for sub_key in ["Psi", "rhop", "Br", "Bt", "Bz"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key, "f8", \
                                                         ("N_time", "N_eq_2D_R", "N_eq_2D_z"))
                if np.shape(self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key))[1]==0:
                    print('Array is empty: '+sub_key)
                var[:] = self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key)
                
            for sub_key in ["R_ax", "z_ax", "Psi_ax", "Psi_sep"]:
                var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key, "f8", \
                                                         ("N_time",))
                var[:] = self["plasma"]['eq_data_2D'].get_single_attribute_from_all_slices(sub_key)

            var = rootgrp["Scenario"].createVariable("plasma" + "_" + \
                                                     "vessel_bd", "f8", \
                                                     ("N_vessel_bd", "N_vessel_dim"))
            var[...,0] = self["plasma"]['vessel_bd'][...,0]

            var[...,1] = self["plasma"]['vessel_bd'][...,1]

        for sub_key in self["diagnostic"].keys():
            if(sub_key == "diag_name"):
                var = rootgrp["Scenario"].createVariable("diagnostic_" +  sub_key, str, \
                                                             ("N_time","N_ch"))
                var[:] = self["diagnostic"][sub_key]
            else:
                var = rootgrp["Scenario"].createVariable("diagnostic_" +  sub_key, "f8", \
                                                             ("N_time","N_ch"))
                var[:] = self["diagnostic"][sub_key]

        if(globalsettings.AUG):
            for sub_key in self["AUG"].keys():
                if(sub_key in ["IDA_ed", "EQ_ed"]):
                    var = rootgrp["Scenario"].createVariable("AUG_" +  sub_key, "i8", \
                                                            ("str_dim",))
                    var[0] = self["AUG"][sub_key]
                else:
                    var = rootgrp["Scenario"].createVariable("AUG_" +  sub_key, str, \
                                                            ("str_dim",))
                    var[0] = self["AUG"][sub_key]
        used_diag_dict_sub_keys = ["diags_exp", "diags_diag", \
                                   "diags_ed", "diags_Extra_arg_1",\
                                   "diags_Extra_arg_2","diags_Extra_arg_3"]
        used_diag_dict_formatted = {}

        for sub_key in used_diag_dict_sub_keys:
            used_diag_dict_formatted[sub_key] = []

        for diagname in self["used_diags_dict"]:
            cur_diag = self["used_diags_dict"][diagname]
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI")):
                used_diag_dict_formatted["diags_Extra_arg_1"].append(cur_diag.Rz_exp)
                used_diag_dict_formatted["diags_Extra_arg_2"].append(cur_diag.Rz_diag)
                used_diag_dict_formatted["diags_Extra_arg_3"].append(str(cur_diag.Rz_ed))
            elif(diagname == 'CEC'):
                used_diag_dict_formatted["diags_Extra_arg_1"].append(str(cur_diag.wg))
                used_diag_dict_formatted["diags_Extra_arg_2"].append("{0:1.7e}".format(cur_diag.dtoECESI))
                used_diag_dict_formatted["diags_Extra_arg_3"].append("{0:1.7e}".format(cur_diag.corr))
            elif("CT" in diagname or "IEC" in diagname):
                used_diag_dict_formatted["diags_Extra_arg_1"].append(str(cur_diag.beamline))
                used_diag_dict_formatted["diags_Extra_arg_2"].append("{0:1.8f}".format(cur_diag.pol_coeff_X))
                if(cur_diag.base_freq_140):
                    used_diag_dict_formatted["diags_Extra_arg_3"].append("True")
                else:
                    used_diag_dict_formatted["diags_Extra_arg_3"].append("False")
            else:
                used_diag_dict_formatted["diags_Extra_arg_1"].append("None")
                used_diag_dict_formatted["diags_Extra_arg_2"].append("None")
                used_diag_dict_formatted["diags_Extra_arg_3"].append("None")
            cur_diag = self["used_diags_dict"][diagname]
            if("EXT" == diagname.upper()):
                used_diag_dict_formatted["diags_exp"].append("None")
                used_diag_dict_formatted["diags_diag"].append("None")
                used_diag_dict_formatted["diags_ed"].append("-1")
            else:
                used_diag_dict_formatted["diags_exp"].append(cur_diag.exp)
                used_diag_dict_formatted["diags_diag"].append(cur_diag.diag)
                used_diag_dict_formatted["diags_ed"].append(str(cur_diag.ed))
        rootgrp["Scenario"].createVariable("used_diags_dict_diags", str, \
                                           ("N_used_diags",))
        for sub_key in used_diag_dict_sub_keys:
            rootgrp["Scenario"].createVariable("used_diags_dict_" +  sub_key, str, \
                                               ("N_used_diags",))
        for idiag, diagname in enumerate(self["used_diags_dict"].keys()):
            rootgrp["Scenario"]["used_diags_dict_" +  "diags"][idiag] = diagname
            for sub_key in used_diag_dict_sub_keys:
                rootgrp["Scenario"]["used_diags_dict_" + sub_key][idiag] = \
                    used_diag_dict_formatted[sub_key][idiag]
        if(self["plasma"]["dist_obj"] is not None):
            self["plasma"]["dist_obj"].to_netcdf(rootgrp)
        if(filename is not None):
            rootgrp.close()

    def from_netcdf(self, filename=None, rootgrp=None):
        if(filename is not None):
            rootgrp = Dataset(filename, "r", format="NETCDF4")
        for sub_key in rootgrp["Scenario"].dimensions.keys():
            if(sub_key == "str_dim"):
                continue
            self["dimensions"][sub_key] = rootgrp["Scenario"].dimensions[sub_key].size
        self['shot'] = rootgrp["Scenario"]["shot"][...].item()
        self['time'] = np.array(rootgrp["Scenario"]["time"])
        self["plasma"]["2D_prof"] = bool(rootgrp["Scenario"]["plasma_" + "2D_prof"][...].item())
        self["plasma"]["eq_dim"] = rootgrp["Scenario"]["plasma_" + "eq_dim"][...].item()
        for sub_key in ["Te", "ne"]:
            self['plasma'][sub_key] = np.array(rootgrp["Scenario"]["plasma_" + sub_key])
        if(not self["plasma"]["2D_prof"]):
            self["plasma"]["prof_reference"] = rootgrp["Scenario"]["plasma_prof_reference"][0]
            self["plasma"][self["plasma"]["prof_reference"]] = np.array(rootgrp["Scenario"]["plasma_" + self["plasma"]["prof_reference"]])
        for sub_key in self["scaling"].keys():
            self['scaling'][sub_key] = float(rootgrp["Scenario"]["scaling_" + sub_key][...].item())
        if(self["plasma"]["eq_dim"] == 3):
            for sub_key in ["B_ref", "s_plus", "s_max", \
                           "interpolation_acc", "fourier_coeff_trunc", \
                           "h_mesh", "delta_phi_mesh"]:
                self['plasma']["eq_data_3D"][sub_key] = float(rootgrp["Scenario"]["plasma" + "_" + \
                                                                                  "eq_data_3D" + "_"  + sub_key][...].item())
            for sub_key in ["use_mesh", "use_symmetry"]:
                self['plasma']["eq_data_3D"][sub_key] = bool(rootgrp["Scenario"]["plasma" + "_" + \
                                                                                  "eq_data_3D" + "_"  + sub_key][...].item())
            for sub_key in ["equilibrium_type", "vessel_filename"]:
                self['plasma']["eq_data_3D"][sub_key] = rootgrp["Scenario"]["plasma" + "_" + \
                                                                            "eq_data_3D" + "_"  + sub_key][0]
            self["plasma"]["eq_data_3D"]["equilibrium_files"] = np.array(rootgrp["Scenario"]["plasma" + "_" + \
                                                                                     "eq_data_3D" + "_" + \
                                                                                     "equilibrium_files"])
        else:
            self["plasma"]['eq_data_2D'] = EQDataExt(Ext_data=True)
            eq_slice_data = {}
            for sub_key in ["R", "z", "Psi", "rhop", "Br", "Bt", "Bz",\
                            "R_ax", "z_ax", "Psi_ax", "Psi_sep"]:
                eq_slice_data[sub_key] = np.array(rootgrp["Scenario"]["plasma" + "_" + \
                                                         "eq_data_2D" + "_" +  sub_key])
            self["plasma"]['eq_data_2D'].fill_with_slices_from_dict(self["time"], eq_slice_data)
            self["plasma"]['vessel_bd'] = np.array(rootgrp["Scenario"]["plasma" + "_" + \
                                                         "vessel_bd"])
        for sub_key in self["diagnostic"].keys():
            self["diagnostic"][sub_key] = np.array(rootgrp["Scenario"]["diagnostic_" + sub_key])
        if(globalsettings.AUG):
            try:
                for sub_key in self["AUG"].keys():
                    self["AUG"][sub_key] = rootgrp["Scenario"]["AUG_" +  sub_key][0]
            except IndexError:
                print("WARNING:: Failed to load AUG specific keywords from Scenarios.")
                print("WARNING:: Replacing them with default values.")
                self["AUG"]["IDA_exp"] = "AUGD"
                self["AUG"]["IDA_ed"] = 0
                self["AUG"]["EQ_exp"] = "AUGD"
                self["AUG"]["EQ_diag"] = "EQH"
                self["AUG"]["EQ_ed"] = 0
        self["dimensions"]["N_used_diags"] = len(rootgrp["Scenario"]["used_diags_dict_diags"])
        self["used_diags_dict"] = od()
        for idiag, diagname in enumerate(rootgrp["Scenario"]["used_diags_dict_diags"]):
            if((diagname == "ECN" or diagname == "ECO" or diagname == "ECI")):
                self["used_diags_dict"].update({diagname: ECI_diag(diagname, \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]), \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_1"][idiag], \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_2"][idiag], \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_3"][idiag]))})
            elif("CT" in diagname or "IEC" in diagname):
                self["used_diags_dict"].update({diagname: ECRH_diag(diagname, \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]), \
                                                                   int(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_1"][idiag]), \
                                                                   float(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_2"][idiag]), \
                                                                   rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_3"][idiag] == "True")})
            elif("CEC" == diagname.upper()):
                try:
                    self["used_diags_dict"].update({diagname: CECE_diag(diagname, \
                                                                        rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], 
                                                                        rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], 
                                                                        int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]),
                                                                        wg=int(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_1"][idiag]), 
                                                                        dtoECESI=float(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_2"][idiag]), 
                                                                        corr=float(rootgrp["Scenario"]["used_diags_dict_diags_Extra_arg_3"][idiag]))})
                except:
                    print("ERROR: Could not load CECE parameters. Setting defaults!")
                    self["used_diags_dict"].update({diagname: CECE_diag(diagname, 
                                                                        rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], 
                                                                        rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag],
                                                                        int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]))})                    

                # CECE diag expects f to be time independent -> ndim = 1
                self["used_diags_dict"][diagname].set_f_info(self["diagnostic"]['f'][0][self["diagnostic"]['diag_name'][0] == 'CEC'], 
                                                             self["diagnostic"]['df'][0][self["diagnostic"]['diag_name'][0] =='CEC'])
            elif("EXT" == diagname.upper()):
                self["used_diags_dict"].update({diagname: EXT_diag(diagname)})
                self["used_diags_dict"][diagname].set_from_scenario_diagnostic(self["diagnostic"],0)
            else:
                self["used_diags_dict"].update({diagname: \
                        Diag(diagname, \
                             rootgrp["Scenario"]["used_diags_dict_diags_exp"][idiag], \
                             rootgrp["Scenario"]["used_diags_dict_diags_diag"][idiag], \
                             int(rootgrp["Scenario"]["used_diags_dict_diags_ed"][idiag]))})
            for diag_key in self["avail_diags_dict"]:
                if(diag_key in list(self["used_diags_dict"])):
                    self["avail_diags_dict"].update({diag_key: self["used_diags_dict"][diag_key]})
        if(filename is not None):
            self.data_source = filename
        else:
            self.data_source = "Unknown"
        self.default_diag = list(self["used_diags_dict"].keys())[0]
        if("BounceDistribution" in rootgrp.groups.keys()):
            self["plasma"]["dist_obj"] = Distribution()
            self["plasma"]["dist_obj"].from_netcdf(rootgrp)
        if(filename is not None):
            rootgrp.close()
        

    def autosave(self):
        try:
            os.remove(self.scenario_file)
        except FileNotFoundError:
            pass
        self.to_netcdf(filename=self.scenario_file)


    def load_dist_obj(self, filename=None):
        self["plasma"]["dist_obj"] = Distribution()
        if(filename is not None):
            ext = os.path.splitext(filename)[1]
            if(ext == ".mat"):
                self["plasma"]["dist_obj"].from_mat(filename=filename)
            elif(ext == ".nc"):
               self["plasma"]["dist_obj"].from_netcdf(filename=filename)
            else:
                print("ERROR: Extension " + ext + " is unknown")
                raise(ValueError)

    def load_GENE_obj(self, filename, dstf):
        it = 0 # Only single time point supported
        try:
            if(dstf == "Ge"):
                self["plasma"]["GENE_obj"] = Gene(filename, self["plasma"]["time"][it], self["plasma"]["eq_data_2D"][it])
            else:
                self["plasma"]["GENE_obj"] = GeneBiMax(filename, self["plasma"]["time"][it], self["plasma"]["eq_data_2D"][it], it)
            return True
        except Exception as e:
            self["plasma"]["GENE_obj"] = None
            print("Failed to load the GENE data")
            print(e)
            return False
        
    def duplicate_time_point(self, time_index, new_times):
        # Copies the time index it len(new_times) times using the times in new_times.
        self["dimensions"]["N_time"] = len(new_times)
        self["time"] = new_times
        for key in ["rhop_prof", "Te", "ne"]:
            plasma_ref = np.copy(self["plasma"][key][time_index])
            self["plasma"][key] = np.zeros((self["dimensions"]["N_time"], plasma_ref.size))
            self["plasma"][key][:] = plasma_ref
        slice_ref = self["plasma"]["eq_data_2D"].slices[time_index]
        self["plasma"]["eq_data_2D"].slices = []
        for _ in new_times:
            self["plasma"]["eq_data_2D"].slices.append(slice_ref)
        for key in self["diagnostic"]:
            diagnsotic_ref = np.copy(self["diagnostic"][key][time_index])
            if key != "diag_name":
                self["diagnostic"][key] = np.zeros((self["dimensions"]["N_time"], diagnsotic_ref.size))
                self["diagnostic"][key][:] = diagnsotic_ref
        
    def integrate_GeneData(self, used_times):
        # We need this to hack in extra time points for the GENE computation
        if(self.GENE_obj == None):
            print("Gene object not initialized")
            return False
        new_times = []
        for used_time in used_times:
            it_gene = np.argmin(np.abs(self.GENE_obj.time - (used_time - self["time"][0])))
            new_times.append(self["time"][0] + self.GENE_obj.time[it_gene])
        new_times = np.array(new_times)
        self.duplicate_time_point(0, new_times)
        return True

if(__name__ == "__main__"):
    newScen = ECRadScenario(noLoad=True)
    newScen.load( filename="/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_ed1.nc")
    newScen.to_netcdf("/mnt/c/Users/Severin/ECRad_regression/AUGX3/ECRad_32934_EXT_Scenario.nc")
     
    
    
    
    