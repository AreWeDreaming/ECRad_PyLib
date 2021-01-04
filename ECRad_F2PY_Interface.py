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
from ECRad_Interface import GetECRadExec


class ECRadF2PYInterface:
    def __init__(self):
        self.diag_init = False
        self.eq_init = False
        self.ECRad = None
        self.N_ch = None # should stay the same once initialized
        self.update_grid = False
        self.fm_flag = None

    def set_config_and_diag(self, Config, Scenario, itime):
        # This sets up the environment variables for OpenMP
        ecrad_exec_dummy = GetECRadExec(Config, Scenario, Scenario.plasma_dict["time"][itime])
        print(os.environ["LDFLAGS"])
        try:
            import ECRad_python
        except Exception as e:
            print("Failed to load ECRad_Python")
            print("Currently set ECRad dir: " + globalsettings.ECRadLibDir)
            print(e)
            raise(e)
        self.ECRad = ECRad_python.ecrad_python
        # Si units and radians
        self.N_ch = len(Scenario.ray_launch[itime]["f"])
        
        self.ECRad.pre_initialize_ecrad(Config.scratch_dir, "init", Config.extra_output, Config.raytracing, Config.ripple, \
                                          1.20, Config.weak_rel, \
                                          Config.ratio_for_3rd_harm, \
                                          Config.considered_modes, Config.reflec_X, Config.reflec_O, False, \
                                          Config.max_points_svec, \
                                          Config.mode_conv, \
                                          Scenario.Te_rhop_scale, Scenario.ne_rhop_scale, Scenario.bt_vac_correction, \
                                          Config.small_ds, Config.large_ds, Config.R_shift, \
                                          Config.z_shift, Config.N_ray, Config.N_freq, True, False, \
                                          Scenario.ray_launch[itime]["f"], Scenario.ray_launch[itime]["df"], \
                                          Scenario.ray_launch[itime]["R"], np.deg2rad(Scenario.ray_launch[itime]["phi"]), \
                                          Scenario.ray_launch[itime]["z"], np.deg2rad(Scenario.ray_launch[itime]["phi_tor"]),\
                                          np.deg2rad(Scenario.ray_launch[itime]["theta_pol"]), Scenario.ray_launch[itime]["dist_focus"], \
                                          Scenario.ray_launch[itime]["width"])
    def set_equilibrium(self, Scenario, itime ):
        if(Scenario.use3Dscen.used):
            rhop_out = self.ECRad.initialize_ecrad_3d("init", self.N_ch, 1, 1, \
                               Scenario.use3Dscen.equilibrium_file, Scenario.use3Dscen.equilibrium_type, \
                               Scenario.use3Dscen.use_mesh, Scenario.use3Dscen.use_symmetry, \
                               Scenario.use3Dscen.B_ref, Scenario.use3Dscen.s_plus, Scenario.use3Dscen.s_max, \
                               Scenario.use3Dscen.interpolation_acc, Scenario.use3Dscen.fourier_coeff_trunc, \
                               Scenario.use3Dscen.h_mesh, Scenario.use3Dscen.delta_phi_mesh, \
                               Scenario.use3Dscen.vessel_filename)
        else:
            rhop_out = self.ECRad.initialize_ecrad("init", self.N_ch, 1, 1, Scenario.plasma_dict["eq_data"][itime].R, \
                                        Scenario.plasma_dict["eq_data"][itime].z, \
                                        Scenario.plasma_dict["eq_data"][itime].rhop, \
                                        Scenario.plasma_dict["eq_data"][itime].Br, \
                                        Scenario.plasma_dict["eq_data"][itime].Bt, \
                                        Scenario.plasma_dict["eq_data"][itime].Br, \
                                        Scenario.plasma_dict["eq_data"][itime].R_ax, \
                                        Scenario.plasma_dict["eq_data"][itime].z_ax)
        return rhop_out
    
    def make_rays(self, Scenario, itime):
        rho = Scenario.plasma_dict[Scenario.plasma_dict["prof_reference"]][itime]
        ne = Scenario.plasma_dict["ne"][itime]
        Te = Scenario.plasma_dict["Te"][itime]
        rho_res = self.ECRad.make_rays_ecrad(self.N_ch,rho, ne, rho, Te)
        return rho_res
    
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

if(__name__ == "__main__"):
    ECRad_folder = "/mnt/c/Users/Severin/ECRad/"
    os.chdir(globalsettings.ECRadLibDir)
    print(os.getcwd())
    ECRad_file = os.path.join(ECRad_folder, "ECRad_35662_EXT_ed1.mat")
#     ECRad_file = "/gss_efgw_work/work/g2sdenk/ECRad_runs/ECRad_20180823016002_EXT_ed20.mat"
    Scenario = ECRadScenario(True)
    Config = ECRadConfig(True)
    Config.from_mat_file(path=ECRad_file)
    Scenario.from_mat(path_in=ECRad_file)
    Config.working_dir = ECRad_folder
    Config.scratch_dir = Config.working_dir
    Config.extra_output =False
    Config.batch = False
    ecrad_f2py_interface = ECRadF2PYInterface()
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
    
    