'''
Created on Nov 22, 2019

@author: g2sdenk
'''
from Global_Settings import globalsettings
from ECRad_Scenario import ECRadScenario
from ECRad_Config import ECRadConfig
import os
import sys
import numpy as np
sys.path.append(globalsettings.ECRadRoot)
from Plotting_Configuration import plt
from ECRad_Execution import GetECRadExec
from scipy import constants as cnst

class ECRadF2PYInterface:
    def __init__(self, Config, Scenario):
        self.diag_init = False
        self.eq_init = False
        self.N_ch = None # should stay the same once initialized
        self.update_grid = False
        self.fm_flag = None
        # Sets up the enivorment
        ecrad_exec_dummy = GetECRadExec(Config, Scenario)
        print(os.environ["LDFLAGS"])
        try:
            import ECRad_python
        except Exception as e:
            print("Failed to load ECRad_Python")
            print("Currently set ECRad dir: " + globalsettings.ECRadLibDir)
            print(e)
            raise(e)
        try:
            import ECRad_python_3D_extension
        except Exception as e:
            print("Failed to load the 3D extension of ECRad_Python")
            print("Currently set ECRad dir: " + globalsettings.ECRadLibDir)
            print(e)
            raise(e)
        self.ECRad = ECRad_python.ecrad_python
        self.ECRad_3D_extension = ECRad_python_3D_extension.ecrad_python_3d_extension

    def set_config_and_diag(self, Config, Scenario, itime):
        # This sets up the environment variables for OpenMP
        # Si units and radians
        self.N_ch = Scenario["dimensions"]["N_ch"]
        self.ECRad.pre_initialize_ecrad(Config["Execution"]["extra_output"], \
                                        Config["Physics"]["dstf"], \
                                        Config["Physics"]["raytracing"], \
                                        Config["Physics"]["ripple"], 1.20, Config["Physics"]["weak_rel"], \
                                        Config["Physics"]["ratio_for_3rd_harm"], \
                                        Config["Physics"]["considered_modes"], \
                                        Config["Physics"]["reflec_X"], Config["Physics"]["reflec_O"], False, \
                                        Config["Numerics"]["max_points_svec"], \
                                        Config["Numerics"]["N_BPD"], \
                                        Config["Physics"]["mode_conv"], \
                                        Scenario["scaling"]["Te_rhop_scale"], Scenario["scaling"]["ne_rhop_scale"], \
                                        Config["Numerics"]["large_ds"], Config["Numerics"]["small_ds"], \
                                        Config["Physics"]["R_shift"], Config["Physics"]["z_shift"], \
                                        Config["Physics"]["N_ray"], Config["Physics"]["N_freq"], True, \
                                        Scenario["dimensions"]["N_vessel_bd"], \
                                        Scenario["plasma"]["vessel_bd"].T[0], \
                                        Scenario["plasma"]["vessel_bd"].T[1], \
                                        Scenario["diagnostic"]["f"][itime], Scenario["diagnostic"]["df"][itime], \
                                        Scenario["diagnostic"]["R"][itime], np.deg2rad(Scenario["diagnostic"]["phi"][itime]), \
                                        Scenario["diagnostic"]["z"][itime], np.deg2rad(Scenario["diagnostic"]["phi_tor"][itime]),\
                                        np.deg2rad(Scenario["diagnostic"]["theta_pol"][itime]), Scenario["diagnostic"]["dist_focus"][itime], \
                                        Scenario["diagnostic"]["width"][itime])
        
    def set_equilibrium(self, Scenario, itime):
        if(Scenario["plasma"]["eq_dim"] == 3):
            rho_out = self.ECRad.initialize_ecrad_3d("init", self.N_ch, 1, 1, \
                               Scenario["plasma"]["eq_data_3D"]["equilibrium_file"], \
                               Scenario["plasma"]["eq_data_3D"]["equilibrium_type"], \
                               Scenario["plasma"]["eq_data_3D"]["use_mesh"], \
                               Scenario["plasma"]["eq_data_3D"]["use_symmetry"], \
                               Scenario["plasma"]["eq_data_3D"]["B_ref"], \
                               Scenario["plasma"]["eq_data_3D"]["s_plus"], \
                               Scenario["plasma"]["eq_data_3D"]["s_max"], \
                               Scenario["plasma"]["eq_data_3D"]["interpolation_acc"], \
                               Scenario["plasma"]["eq_data_3D"]["fourier_coeff_trunc"], \
                               Scenario["plasma"]["eq_data_3D"]["h_mesh"], \
                               Scenario["plasma"]["eq_data_3D"]["delta_phi_mesh"], \
                               Scenario["plasma"]["eq_data_3D"]["vessel_filename"])
        else:
            time = Scenario["time"][itime]
            eq_slice = Scenario["plasma"]["eq_data_2D"].GetSlice(time, Scenario["scaling"]["Bt_vac_scale"])
            if(Scenario["plasma"]["2D_prof"]):
                self.ECRad.initialize_ecrad_2D(self.N_ch, 1, 1, eq_slice.R, \
                                                       eq_slice.z, eq_slice.rhop, eq_slice.Br, \
                                                       eq_slice.Bt, eq_slice.Br, eq_slice.R_ax, eq_slice.z_ax, \
                                                       Scenario["Te"]. Scenario["ne"])
            else:
                self.ECRad.initialize_ecrad(self.N_ch, 1, 1, eq_slice.R, \
                                                       eq_slice.z, eq_slice.rhop, eq_slice.Br, \
                                                       eq_slice.Bt, eq_slice.Br, eq_slice.R_ax, \
                                                       eq_slice.z_ax)
    
    def make_rays(self, Scenario, itime):
        if(Scenario["plasma"]["2D_prof"]):
            rho_res = self.ECRad.make_rays_ecrad_2D()
        else:
            rho = Scenario["plasma"][Scenario["plasma"]["prof_reference"]][itime]
            ne = Scenario["plasma"]["ne"][itime]
            Te = Scenario["plasma"]["Te"][itime]
            rho_res = self.ECRad.make_rays_ecrad(self.N_ch,rho, ne, rho, Te)
        return rho_res
    
    def run_and_get_output(self, Result, itime):
        self.set_fm_flag
        self.ECRad.make_trad_direct()
        key = "Trad"
        if(Result.Scenario["plasma"]["eq_dim"] == 3):
            rho = "rhot"
        else:
            rho = "rhop"
        for sub_key in ["Trad", "tau", "T"]:
            Result[key][sub_key].append(np.zeros(Result.get_shape(key, start=1)))
        key = "resonance"
        for sub_key in ["s_cold", "R_cold", "z_cold", rho + "_cold"]:
            Result[key][sub_key].append(np.zeros(Result.get_shape(key, start=1)))
        for imode in range(Result["dimensions"]["N_mode_mix"]):
            Result["Trad"]["Trad"][-1][:,imode], \
                 Result["Trad"]["tau"][-1][:,imode], \
                 Result["resonance"]["s_cold"][-1][:,imode], \
                 Result["resonance"]["R_cold"][-1][:,imode], \
                 Result["resonance"]["z_cold"][-1][:,imode], \
                 Result["resonance"][rho + "_cold"][-1][:,imode] = \
                    self.ECRad.get_trad_resonances_basic(imode, Result["dimensions"]["N_ch"])
            Result["Trad"]["T"][-1][:,imode] = np.exp(-Result["Trad"]["tau"][-1][:,imode])
        if(Result.Config["Execution"]["extra_output"]):
            key = "Trad"
            for sub_key in ["Trad_second", "tau_second", "T_second"]:
                Result[key][sub_key].append(np.zeros(Result.get_shape(key, start=1)))
            key = "resonance"
            for sub_key in ["s_warm", "R_warm", "z_warm", rho + "_warm", \
                            "s_warm_second", "R_warm_second", \
                            "z_warm_second", rho + "_warm_second"]:
                Result[key][sub_key].append(np.zeros(Result.get_shape(key, start=1)))
            for imode in range(Result["dimensions"]["N_mode_mix"]):
                Result["Trad"]["Trad_second"][-1][:,imode], \
                    Result["Trad"]["tau_second"][-1][:,imode], \
                    Result["resonance"]["s_warm"][-1][:,imode], \
                    Result["resonance"][rho + "_warm"][-1][:,imode], \
                    Result["resonance"]["R_warm"][-1][:,imode], \
                    Result["resonance"]["z_warm"][-1][:,imode], \
                    Result["resonance"]["s_warm_second"][-1][:,imode], \
                    Result["resonance"][rho + "_warm_second"][-1][:,imode], \
                    Result["resonance"]["R_warm_second"][-1][:,imode], \
                    Result["resonance"]["z_warm_second"][-1][:,imode] = \
                    self.ECRad.get_trad_resonances_extra_output(imode, Result["dimensions"]["N_ch"])
                Result["Trad"]["T_second"][-1][:,imode] = np.exp(-Result["Trad"]["tau_second"][-1][:,imode])
            key = "BPD"
            for sub_key in [rho, "BPD", "BPD_second"]:
                Result[key][sub_key].append(np.zeros(Result.get_shape(key, start=1)))
            for ich in range(Result["dimensions"]["N_ch"]):
                for imode in range(Result["dimensions"]["N_mode"]):
                    Result[key][rho][-1][ich,imode,:], \
                        Result[key]["BPD"][-1][ich,imode,:], Result[key]["BPD_second"][-1][ich,imode,:] = \
                        self.ECRad.get_bpd(ich + 1, imode + 1, Result["dimensions"]["N_BPD"])
            key = "ray"
            Result["dimensions"]["N_LOS"].append(np.zeros(Result.get_shape("ray", start=1, stop=-1), dtype=np.int))
            for ich in range(Result["dimensions"]["N_ch"]):
                for imode in range(Result["dimensions"]["N_mode"]):
                    for ir in range(Result["dimensions"]["N_ray"]):
                        Result["dimensions"]["N_LOS"][-1][ich,imode, ir] = self.ECRad.get_ray_length(ich + 1, imode + 1, ir + 1)
            for sub_key in Result.sub_keys["ray"]:
                Result[key][sub_key].append(np.zeros(Result.get_shape("ray", start=1, stop=-1),dtype=np.object))
                for ich in range(Result["dimensions"]["N_ch"]):
                    for imode in range(Result["dimensions"]["N_mode"]):
                        for ir in range(Result["dimensions"]["N_ray"]):
                            Result[key][sub_key][-1][ich,imode,ir] = np.zeros(Result.get_shape(key, start=-1, i_time=itime, \
                                                                                             i_ch=ich, i_mode=imode, i_ray=ir))
            for ich in range(Result["dimensions"]["N_ch"]):
                for imode in range(Result["dimensions"]["N_mode"]):
                    for ir in range(Result["dimensions"]["N_ray"]):
                        Result[key]["s"][-1][ich,imode,ir][:], \
                            Result[key]["x"][-1][ich,imode,ir][:], \
                            Result[key]["y"][-1][ich,imode,ir][:], \
                            Result[key]["z"][-1][ich,imode,ir][:], \
                            Result[key]["Nx"][-1][ich,imode,ir][:], \
                            Result[key]["Ny"][-1][ich,imode,ir][:], \
                            Result[key]["Nz"][-1][ich,imode,ir][:], \
                            Result[key]["Bx"][-1][ich,imode,ir][:], \
                            Result[key]["By"][-1][ich,imode,ir][:], \
                            Result[key]["Bz"][-1][ich,imode,ir][:], \
                            Result[key][rho][-1][ich,imode,ir][:], \
                            Result[key]["Te"][-1][ich,imode,ir][:], \
                            Result[key]["ne"][-1][ich,imode,ir][:], \
                            Result[key]["theta"][-1][ich,imode,ir][:], \
                            Result[key]["Nc"][-1][ich,imode,ir][:], \
                            Result[key]["H"][-1][ich,imode,ir][:], \
                            Result[key]["v_g_perp"][-1][ich,imode,ir][:], \
                            Result[key]["Trad"][-1][ich,imode,ir][:], \
                            Result[key]["Trad_second"][-1][ich,imode,ir][:], \
                            Result[key]["em"][-1][ich,imode,ir][:], \
                            Result[key]["em_second"][-1][ich,imode,ir][:], \
                            Result[key]["ab"][-1][ich,imode,ir][:], \
                            Result[key]["ab_second"][-1][ich,imode,ir][:], \
                            Result[key]["T"][-1][ich,imode,ir][:], \
                            Result[key]["T_second"][-1][ich,imode,ir][:], \
                            Result[key]["BPD"][-1][ich,imode,ir][:], \
                            Result[key]["BPD_second"][-1][ich,imode,ir][:] = \
                            self.ECRad.get_ray_data(ich + 1, imode + 1, ir + 1, \
                                                    Result["dimensions"]["N_LOS"][-1][ich,imode,ir])
                        Result[key]["N"][-1][ich,imode,ir][:] = np.sqrt(Result[key]["Nx"][-1][ich,imode,ir]**2 + \
                                                                        Result[key]["Ny"][-1][ich,imode,ir]**2 + \
                                                                        Result[key]["Nz"][-1][ich,imode,ir]**2)
                        Result[key]["Y"][-1][ich,imode,ir][:] = np.sqrt(Result[key]["Bx"][-1][ich,imode,ir]**2 + \
                                                                        Result[key]["By"][-1][ich,imode,ir]**2 + \
                                                                        Result[key]["Bz"][-1][ich,imode,ir]**2)
                        f = Result.Scenario["diagnostic"]["f"][itime][ich]
                        Result[key]["Y"][-1][ich,imode,ir] *= cnst.e/(cnst.m_e* 2.0 * np.pi * f)
                        Result[key]["X"][-1][ich,imode,ir] = cnst.e**2*Result[key]["ne"][-1][ich,imode,ir] / \
                                                          (cnst.epsilon_0*cnst.m_e* (2.0 * np.pi * f)**2)
            key = "weights"
            for sub_key in Result.sub_keys[key]:
                Result[key][sub_key].append(np.zeros(Result.get_shape(sub_key, start=1)))
            for ich in range(Result["dimensions"]["N_ch"]):
                Result[key]["ray_weights"][-1][ich,:], Result[key]["freq_weights"][-1][ich,:] = \
                    self.ECRad.get_weights(Result["dimensions"]["N_ray"], Result["dimensions"]["N_freq"], ich + 1)
            if(Result["dimensions"]["N_mode"] > 1):
                for imode in range(Result["dimensions"]["N_mode"]):
                    Result["weights"]["mode_frac"][-1][:,imode], \
                    Result["weights"]["mode_frac_second"][-1][:,imode] = \
                        self.ECRad.get_mode_weights(Result["dimensions"]["N_ch"], imode+1)
            else:
                Result["weights"]["mode_frac"][-1][:,imode] = 1.0
                Result["weights"]["mode_frac_second"][-1][:,imode] = 1.0
        return Result
    
    def eval_Trad(self, Scenario, Config, itime):
        rho = Scenario.plasma_dict[Scenario.plasma_dict["prof_reference"]][itime]
        ne = Scenario.plasma_dict["ne"][itime]
        Te = Scenario.plasma_dict["Te"][itime]
        Trad, tau = self.ECRad.make_dat_model_ecrad(self.N_ch, rho, ne, rho, Te, \
                                                    1.0, Config.reflec_X, \
                                                    Config.reflec_O, self.fm_flag, 0.0, \
                                                    self.update_grid, False)
        return(Trad, tau)

    def set_fm_flag(self, fm_flag):
        self.fm_flag = np.copy(fm_flag)
        
    def set_grid_update(self, update_grid):
        self.update_grid = update_grid
        
    def reset(self):
        self.ECRad.reset_ecrad()
        
    def process_single_timepoint(self, Result, itime):
        self.reset()
        self.set_config_and_diag(Result.Config, Result.Scenario, itime)
        self.set_equilibrium(Result.Scenario, itime)
        self.make_rays(Result.Scenario, itime)
        Result = self.run_and_get_output(Result, itime)
        return Result
        

if(__name__ == "__main__"):
    ECRad_folder = "/mnt/c/Users/Severin/ECRad/"
    os.chdir(globalsettings.ECRadLibDir)
    print(os.getcwd())
    ECRad_file = os.path.join(ECRad_folder, "ECRad_35662_EXT_ed1.mat")
#     ECRad_file = "/gss_efgw_work/work/g2sdenk/ECRad_runs/ECRad_20180823016002_EXT_ed20.mat"
    Scenario = ECRadScenario(True)
    Config = ECRadConfig(True)
    Config.from_mat(path_in=ECRad_file)
    Scenario.from_mat(path_in=ECRad_file)
    Config.working_dir = ECRad_folder
    Config.scratch_dir = Config.working_dir
    Config.extra_output =False
    Config.batch = False
    ecrad_f2py_interface = ECRadF2PYInterface(Config, Scenario)
    ecrad_f2py_interface.set_config_and_diag(Config, Scenario, 0)
    rhop_out = ecrad_f2py_interface.set_equilibrium(Scenario, 0 )
    rhop = Scenario.plasma_dict["rhop_prof"][0]
    ne = Scenario.plasma_dict["ne"][0]
    Te = Scenario.plasma_dict["Te"][0]
    rhop_out = ecrad_f2py_interface.make_rays(Scenario, 0)
    fm_flag = np.zeros(ecrad_f2py_interface.N_ch, dtype=np.bool)
    fm_flag[:] = True
    ecrad_f2py_interface.set_fm_flag(fm_flag)
    Trad, tau = ecrad_f2py_interface.eval_Trad(Scenario, Config, 0)
    plt.plot(rhop_out, Trad / 1.e3, "+")
    plt.show()
    
    