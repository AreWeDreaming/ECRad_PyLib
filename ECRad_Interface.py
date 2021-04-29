'''
Created on Dec 9, 2015

@author: sdenk
'''
from Global_Settings import globalsettings
import os
import numpy as np
import sys
sys.path.append("../ECRad_Pylib")
if(globalsettings.AUG):
    vessel_file = os.path.join(globalsettings.ECRadPylibRoot,'ASDEX_Upgrade_vessel.txt')
from shutil import copyfile, rmtree
from scipy.io import loadmat
from Basic_Methods.Equilibrium_Utils import EQDataExt
from Em_Albajar import SVec, EmAbsAlb
from Distribution_IO import export_gene_fortran_friendly, \
                            export_gene_bimax_fortran_friendly, export_fortran_friendly
from scipy.interpolate import InterpolatedUnivariateSpline
from TB_Communication import make_topfile_no_data_load, make_Te_ne_files
from netCDF4 import Dataset

def GetECRadExec(Config, Scenario, time):
    # Determine OMP stacksize
    parallel = Config.parallel
    parallel_cores = Config.parallel_cores
    ECRadVers = globalsettings.ECRadPath
    if(parallel):
        ECRadVers += "OMP"
    if(Scenario.use3Dscen.used):
        ECRadVers += "USE3D"
    if(Config.debug):
        if(Config.parallel):
            print("No parallel version with debug symbols available at the moment")
            print("Falling back to single core")
            parallel = False
        ECRadVers += "db"
    if(parallel and parallel_cores > globalsettings.max_cores):
        print("The maximum amount of cores for the current machine is: ", globalsettings.max_cores )
        print("Settings amount of cores to maximum for best performance")
        parallel_cores = globalsettings.max_cores
    if(parallel):
        stacksize = 0
        factor = 1
        for diag in Scenario.used_diags_dict:
            if(diag == "ECN" or diag == "ECO"  or diag == "ECI"):
                factor = 3
        if(Config.dstf in ["Ge", "GB", "Re", "Lu"]):
            factor *= 3
        stacksize += int(np.ceil(Config.max_points_svec * 3.125) * factor)
        os.environ['OMP_STACKSIZE'] = "{0:d}k".format(stacksize)
    else:
        parallel_cores = 1 # serial
    if(Config.batch):
        os.environ['ECRad_working_dir_1'] = Config.scratch_dir
        os.environ['ECRad'] = ECRadVers
        launch_options_dict = {}
        launch_options_dict["jobname"] = "-J " + "E{0:5d}{1:1.1f}".format(Scenario.shot, time)
        launch_options_dict["IO"] = "-o {0:s} -e {1:s} ".format(os.path.join(Config.scratch_dir, "ECRad.stdout"), \
                                                                os.path.join(Config.scratch_dir, "ECRad.stderr"))
        launch_options_dict["partition"] = globalsettings.partition_function(parallel_cores, Config.wall_time)
        launch_options_dict["qos"] = globalsettings.qos_function(parallel_cores, Config.wall_time)
        launch_options_dict["memory"] = "--mem-per-cpu={0:d}M".format(int(Config.vmem / parallel_cores))
        launch_options_dict["cpus"] = " --cpus-per-task={0:d}".format(parallel_cores)
        InvokeECRad = "sbatch"
        for key in launch_options_dict:
            InvokeECRad += " " + launch_options_dict[key]
        InvokeECRad += " " + globalsettings.ECRadPathBSUB
    else:
        os.environ['OMP_NUM_THREADS'] = "{0:d}".format(parallel_cores)
        InvokeECRad = ECRadVers + " " + Config.scratch_dir
    return InvokeECRad

def prepare_input_files(Config, Scenario, index, copy_dist=True, \
                        ext_result = None):
    working_dir = Config.scratch_dir
    # eq_exp = Config.EQ_exp always exp
    ECRad_data_path = os.path.join(working_dir, "ECRad_data", "")
    if(Config.extra_output):
        if(os.path.isdir(ECRad_data_path)):
        #Get rid of old data -> This ensures that the new data is really new
            rmtree(ECRad_data_path)
        try:
            os.mkdir(ECRad_data_path)
        except OSError:
            print("Failed to create ECRad_data folder in: ", working_dir)
            print("Please check that this folder exists and you have write permissions")
            return False
        print("Created folder " + ECRad_data_path)#
#         if(Config.dstf != "GB"):
        Ich_path = os.path.join(ECRad_data_path, "Ich" + Config.dstf)
#         else:
#             Ich_path = os.path.join(ECRad_data_path, "IchGe")
        if(not os.path.isdir(Ich_path)):
            os.mkdir(Ich_path)
            print("Created folder " + Ich_path)
        ray_folder = os.path.join(ECRad_data_path, "ray")
        if(not os.path.isdir(ray_folder)):
            os.mkdir(ray_folder)
            print("Created folder " + ray_folder)
    else:
        if(not os.path.isdir(ECRad_data_path)):
            try:
                os.mkdir(ECRad_data_path)
            except OSError:
                print("Failed to create ECRad_data folder in: ", working_dir)
                print("Please check that this folder exists and you have write permissions")
                return False
        else:
            #Get rid of old data -> This ensures that the new data is really new
            rmtree(ECRad_data_path)
            try:
                os.mkdir(ECRad_data_path)
            except OSError:
                print("Failed to create ECRad_data folder in: ", working_dir)
                print("Please check that this folder exists and you have write permissions")
                return False
    write_diag_launch(ECRad_data_path, Scenario.ray_launch[index])
    if(Scenario.Te_scale != 1.0):
        print("Te scale != 1 -> scaling Te for model")
    if(Scenario.ne_scale != 1.0):
        print("ne scale != 1 -> scaling ne for model")
    if(Scenario.bt_vac_correction != 1.0):
        print("Bt scale != 1 -> scaling Bt for model")
    success = make_ECRadInputFromPlasmaDict(ECRad_data_path, Scenario.plasma_dict, index, Scenario)
    if(not success):
        print("An error occured when creating input profiles and the topfile")
        return False
    input_file = open(os.path.join(ECRad_data_path, "ECRad.inp"), "w")
#     if(Config.dstf == "GB"):
#         input_file.write("Ge" + "\n")  # Model does not distinguish between Ge and GB
#     else:
    input_file.write(Config.dstf + "\n")
    if(Config.extra_output):
        input_file.write("T\n")
    else:
        input_file.write("F\n")
    if(Config.raytracing):
        input_file.write("F\n")
    else:
        input_file.write("T\n")
    if(Config.ripple):
        input_file.write("T\n")
    else:
        input_file.write("F\n")
    if(Config.weak_rel):
        input_file.write("T\n")
    else:
        input_file.write("F\n")
    input_file.write("{0:1.12E}".format(Config.ratio_for_3rd_harm) + "\n")
    input_file.write(str(Config.reflec_model) + "\n")
    input_file.write("{0:1.12E}".format(Config.reflec_X) + "\n")
    input_file.write("{0:1.12E}".format(Config.reflec_O) + "\n")
    input_file.write(str(Config.considered_modes) + "\n")
    input_file.write("{0:1.12E}".format(Config.mode_conv) + "\n")
    input_file.write(str(Config.N_freq) + "\n")
    input_file.write(str(Config.N_ray) + "\n")
    input_file.write("{0:1.12E}".format(Config.large_ds) + "\n")
    input_file.write("{0:1.12E}".format(Config.small_ds) + "\n")
    input_file.write("{0:1.12E}".format(Config.R_shift) + "\n")
    input_file.write("{0:1.12E}".format(Config.z_shift) + "\n")
    input_file.write("{0:10d}".format(Config.max_points_svec) + "\n")
    if(Config.use_ext_rays):
        input_file.write("T\n")
    else:
        input_file.write("F\n")
    if(Scenario.use3Dscen.used):
        input_file.write("3D\n")
    elif(Scenario.plasma_dict["Te"][0].ndim == 2):
        input_file.write("2D\n")
    else:
        input_file.write("1D\n")
    input_file.flush()
    input_file.close()
    if(Config.use_ext_rays):
        if(ext_result is None):
            raise ValueError("Told to write external rays. but no external result provided")
        write_ext_ray_input(ECRad_data_path, Config, Scenario, index, ext_result)
    if(Scenario.use3Dscen.used):
        try:
            use3dconfigfile = open(os.path.join(ECRad_data_path, "equ3D_info"), "w")
            use3dconfigfile.write(os.path.basename(Scenario.use3Dscen.equilibrium_file) + "\n")
            use3dconfigfile.write(Scenario.use3Dscen.equilibrium_type + "\n")
            if(Scenario.use3Dscen.use_mesh):
                use3dconfigfile.write("T\n")
            else:
                use3dconfigfile.write("F\n")
            if(Scenario.use3Dscen.use_symmetry):
                use3dconfigfile.write("T\n")
            else:
                use3dconfigfile.write("F\n")
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.B_ref))
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.s_plus))
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.s_max))
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.interpolation_acc))
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.fourier_coeff_trunc))
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.h_mesh))
            use3dconfigfile.write("{0:1.7e}\n".format(Scenario.use3Dscen.delta_phi_mesh))
            use3dconfigfile.flush()
            use3dconfigfile.close()
            copyfile(os.path.join(Scenario.use3Dscen.equilibrium_file), \
                     os.path.join(ECRad_data_path,os.path.basename(Scenario.use3Dscen.equilibrium_file)))
            copyfile(os.path.join(Scenario.use3Dscen.vessel_filename), \
                     os.path.join(ECRad_data_path, "vessel_bd.txt"))
        except OSError as e:
            print("Did not find an equilibrium file at " + Scenario.use3Dscen.equilibrium_file)
            print(e)
            return False
    else:
        fvessel = open(os.path.join(ECRad_data_path, "vessel_bd.txt"), "w")
        fvessel.write("{0: 7d}".format(len(Scenario.plasma_dict["vessel_bd"][0])) + "\n")
        for i in range(len(Scenario.plasma_dict["vessel_bd"][0])):
            fvessel.write("{0: 1.12E} {1: 1.12E}".format(Scenario.plasma_dict["vessel_bd"][0][i], Scenario.plasma_dict["vessel_bd"][1][i]) + "\n")
        fvessel.flush()
        fvessel.close()
    if(Config.dstf == "Re" and copy_dist):
        fRe_dir = os.path.join(ECRad_data_path, "fRe")
        os.mkdir(fRe_dir)
        export_fortran_friendly([Scenario.dist_obj, fRe_dir])
    if(Config.dstf == "Ge" and copy_dist):
        wpath = os.path.join(os.path.join(ECRad_data_path, "fGe"))
        if os.path.exists(wpath):
            rmtree(wpath)  # Removing the old files first is faster than overwriting them
        os.mkdir(wpath)
        export_gene_fortran_friendly(wpath, Scenario.GENE_obj.rhop, Scenario.GENE_obj.beta_par, \
                                     Scenario.GENE_obj.mu_norm, Scenario.GENE_obj.ne, \
                                     Scenario.GENE_obj.f[index], Scenario.GENE_obj.f0, \
                                     Scenario.GENE_obj.B0)
    if(Config.dstf == "GB" and copy_dist):
        wpath = os.path.join(os.path.join(ECRad_data_path, "fGB"))
        if os.path.exists(wpath):
            rmtree(wpath)  # Removing the old files first is faster than overwriting them
        os.mkdir(wpath)
        Te_perp, Te_par = Scenario.GENE_obj.make_bi_max_single_timepoint(index)
        export_gene_bimax_fortran_friendly(wpath, Scenario.GENE_obj.rhop, Scenario.GENE_obj.beta_par, \
                                           Scenario.GENE_obj.mu_norm, Scenario.GENE_obj.Te, Scenario.GENE_obj.ne, \
                                           Te_perp, Te_par, Scenario.GENE_obj.B0)
        Te_perp_file = open(os.path.join(ECRad_data_path, "Te_perp.dat"), "w")
        Te_perp_file.write("{0: 5d}\n".format(len(Te_perp)))
        for i in range(len(Te_perp)):
            Te_perp_file.write("{0: 1.12E} {1: 1.12E}\n".format(Scenario.GENE_obj.rhop[i], Te_perp[i]))
        Te_perp_file.close()
        Te_par_file = open(os.path.join(ECRad_data_path, "Te_par.dat"), "w")
        Te_par_file.write("{0: 5d}\n".format(len(Te_par)))
        for i in range(len(Te_perp)):
            Te_par_file.write("{0: 1.12E} {1: 1.12E}\n".format(Scenario.GENE_obj.rhop[i], Te_par[i]))
        Te_par_file.close()
    return True

def write_ext_ray_input(ECRad_data_path, Config, Scenario, index, ext_results):
    ext_ray_path = os.path.join(ECRad_data_path, "ext_rays")
    os.mkdir(ext_ray_path)
    if(len(Scenario.ray_launch[index]["f"]) != len(ext_results.Scenario.ray_launch[index]["f"]) or \
       Config.N_ray != ext_results.Config.N_ray or Config.considered_modes != \
       ext_results.Config.considered_modes):
        print("Number of channels, rays and mode selection must remain unchanged when using external rays")
        raise ValueError("Forbidden modification of the ECRad configuration or diagnostic selection")
    for ich in range(len(ext_results.Scenario.ray_launch[index]["f"])):
        for imode, mode in enumerate(ext_results.modes):
            if(ext_results.Config.N_ray > 1):
                for iray in range(len(ext_results.ray["s" + mode][index][ich])):
                    ray_file_name = "raydata_{0:03d}_{1:03d}_{2:03d}.dat".format(ich+1,imode+1,iray+1)
                    ext_ray_file = open(os.path.join(ext_ray_path, ray_file_name),"w")
                    ext_ray_file.write("{0: 6d}".format(len(ext_results.ray["s" + mode][index][ich][iray])) + "\n")
                    for i in range(len(ext_results.ray["s" + mode][index][ich][iray])):
                        line = ""
                        line += "{0: 1.12E}".format(ext_results.ray["s" + mode][index][ich][iray][i])
                        for key in ["x","y","z", "Nx", "Ny", "Nz", "Bx", "By", "Bz", "rhop", "ne", "Te", "Nc", "v_g_perp"]:
                            line += " {0: 1.12E}".format(ext_results.ray[key + mode][index][ich][iray][i])
                        ext_ray_file.write(line + "\n")
            else:
                iray = 0
                ray_file_name = "raydata_{0:03d}_{1:03d}_{2:03d}.dat".format(ich+1,imode+1,iray+1)
                ext_ray_file = open(os.path.join(ext_ray_path, ray_file_name),"w")
                ext_ray_file.write("{0: 6d}".format(len(ext_results.ray["s" + mode][index][ich])) + "\n")
                for i in range(len(ext_results.ray["s" + mode][index][ich])):
                    line = ""
                    line += "{0: 1.12E}".format(ext_results.ray["s" + mode][index][ich][i])
                    for key in ["x","y","z", "Nx", "Ny", "Nz", "Bx", "By", "Bz", "rhop", "ne", "Te", "Nc", "v_g_perp"]:
                        line += " {0: 1.12E}".format(ext_results.ray[key + mode][index][ich][i])
                    ext_ray_file.write(line + "\n")
            ext_ray_file.flush()
            ext_ray_file.close()
                    
                    

def get_ECE_launch_info(shot, diag):
    from Shotfile_Handling_AUG import get_ECE_launch_params
    import Geo_Los
    N = 200
    ECE_launch = get_ECE_launch_params(shot, diag)
    ECE_launch["phi"] = np.zeros(len(ECE_launch["f"]))
    ECE_launch["phi_tor"] = np.zeros(len(ECE_launch["f"]))
    ECE_launch["theta_pol"] = np.zeros(len(ECE_launch["f"]))
    ECE_launch["width"] = np.zeros(len(ECE_launch["f"]))
    ECE_launch["dist_focus"] = np.zeros(len(ECE_launch["f"]))
    ECE_launch["R"] = np.zeros(len(ECE_launch["f"]))
    ECE_launch["R"][:] = 3.90
    ECE_launch["z"] = np.zeros(len(ECE_launch["f"]))
    wg_last = 0
    R = np.zeros(N)
    z = np.zeros(N)
    for ich in range(len(ECE_launch["f"])):
        if(shot <= 24202):
            print("LOS geometry of discharges with shotno. <= 24202 is not yet implemented")
            print("WARNING - LOS geometry of fwd. model by S. Rathgeber. inconsistent with new LOS geometry.")
            raise ValueError
        elif(shot > 24202 and shot <= 33724):
            if (ECE_launch["waveguide"][ich] == 4 or ECE_launch["waveguide"][ich] == 12):
                ECE_launch["phi_tor"][ich] = -2.1798
                ECE_launch["phi"][ich] = -0.28
            elif(ECE_launch["waveguide"][ich] == 9):
                ECE_launch["phi_tor"][ich] = +2.1798
                ECE_launch["phi"][ich] = 0.28
            elif (ECE_launch["waveguide"][ich] == 10):
                ECE_launch["phi_tor"][ich] = +0.7265
                ECE_launch["phi"][ich] = 0.04
            else:
                print("subroutine make_theta_los: something wrong with wg(ich) for 24202 < shotno. <= 33724!")
                print("wg", ECE_launch["waveguide"][ich])
                raise ValueError
        else:
            if (ECE_launch["waveguide"][ich] == 4 or ECE_launch["waveguide"][ich] == 12):
                ECE_launch["phi_tor"][ich] = -2.1798
                ECE_launch["phi"][ich] = -0.28
            elif(ECE_launch["waveguide"][ich] == 10):
                ECE_launch["phi_tor"][ich] = +0.7265
                ECE_launch["phi"][ich] = -0.04
            elif (ECE_launch["waveguide"][ich] == 11 or ECE_launch["waveguide"][ich] == 3):
                ECE_launch["phi_tor"][ich] = -0.7265
                ECE_launch["phi"][ich] = 0.04
            else:
                print("subroutine make_theta_los: something wrong with wg(ich) for shotno. > 33724!")
                print("wg", ECE_launch["waveguide"][ich])
                raise ValueError
        if(ECE_launch["waveguide"][ich] != wg_last):
            R, z = Geo_Los.geo_los(shot, ECE_launch["waveguide"][ich], ECE_launch["z_lens"], R, z)
            R1 = R[0]
            R2 = R[-1]
            z1 = z[0]
            z2 = z[-1]
            dRds = (R1 - R2) / np.sqrt((R1 - R2) ** 2 + (z1 - z2) ** 2)
            dzds = (z1 - z2) / np.sqrt((R1 - R2) ** 2 + (z1 - z2) ** 2)
            wg_last = ECE_launch["waveguide"][ich]
        ECE_launch["z"][ich] = (ECE_launch["R"][ich] - R1) / dRds * dzds + z1
        ECE_launch["theta_pol"][ich] = np.rad2deg(np.arccos((z2 - ECE_launch["z"][ich]) / \
                                                            np.sqrt((R2 - ECE_launch["R"][ich]) ** 2 + \
                                                                    (z2 - ECE_launch["z"][ich]) ** 2)) - np.pi / 2.e0)
    ECE_launch["phi"][:] += (8.5e0) * 22.5
    ECE_launch["dist_focus"][:] = 2.131
    ECE_launch["width"][:] = 17.17e-2
    del(Geo_Los) # Delete to avoid problems with conflicting libraries
    return ECE_launch

def get_diag_launch(shot, time, used_diag_dict, gy_dict=None, ECI_dict=None):
    print("ECRad data will be written for diags ", used_diag_dict)
    launch_array = []
    for diag in used_diag_dict:
        launch = {}
        if(used_diag_dict[diag].name == "ECE"):
            launch = get_ECE_launch_info(shot, used_diag_dict[diag])
        elif(used_diag_dict[diag].name == "IEC"):
            launch["f"] = []
            dfreq_IEC = 3.0e9
            for i in range(6):
                launch["f"].append(132.5e9 + i * dfreq_IEC)
            launch["f"] = np.array(launch["f"])
            launch["df"] = np.zeros(len(launch["f"]))
            launch["R"] = np.zeros(len(launch["f"]))
            launch["phi"] = np.zeros(len(launch["f"]))
            launch["z"] = np.zeros(len(launch["f"]))
            launch["theta_pol"] = np.zeros(len(launch["f"]))
            launch["phi_tor"] = np.zeros(len(launch["f"]))
            launch["dist_focus"] = np.zeros(len(launch["f"]))
            launch["width"] = np.zeros(len(launch["f"]))
            launch["pol_coeff_X"] = np.zeros(len(launch["f"]))
            launch["df"][:] = dfreq_IEC
            launch["R"][:] = np.sqrt(gy_dict[str(used_diag_dict[diag].beamline)].x ** 2 + gy_dict[str(used_diag_dict[diag].beamline)].y ** 2)
            launch["phi"][:] = gy_dict[str(used_diag_dict[diag].beamline)].phi
            launch["z"][:] = gy_dict[str(used_diag_dict[diag].beamline)].z
            launch["phi_tor"][:] = -gy_dict[str(used_diag_dict[diag].beamline)].phi_tor[np.argmin(np.abs(gy_dict[str(used_diag_dict[diag].beamline)].time - time))]  # TORBEAM convention
            launch["theta_pol"][:] = -gy_dict[str(used_diag_dict[diag].beamline)].theta_pol[np.argmin(np.abs(gy_dict[str(used_diag_dict[diag].beamline)].time - time))]  # TORBEAM convention
            launch["pol_coeff_X"][:] = used_diag_dict[diag].pol_coeff_X
            launch["width"][:] = gy_dict[str(used_diag_dict[diag].beamline)].width_y
            launch["dist_focus"][:] = gy_dict[str(used_diag_dict[diag].beamline)].curv_y
#            R_curv = gy_dict[str(used_diag_dict[diag].beamline)].curv_y
#            dist_foc = (launch["f"] ** 2 * np.pi ** 2 * R_curv * launch["width"] ** 4) / (cnst.c ** 2 * R_curv ** 2 + launch["f"] ** 2 * np.pi ** 2 * launch["width"] ** 4)
#            launch["dist_focus"][:] = dist_foc
        if(used_diag_dict[diag].name == "CTC" or used_diag_dict[diag].name == "CTA"):
            if(type(time) != float):
                time = float(time)
            if(used_diag_dict[diag].name == "CTC"):
                if(used_diag_dict[diag].base_freq_140):
                    f_CTC = np.array([137.0000, 137.6500, 138.0750, 138.3750, 138.5700, \
                         138.6600, 138.7400, 138.8200, 138.9000, 138.9800, \
                         139.0600, 139.1400, 139.2200, 139.3000, 139.3800, \
                         139.4600, 139.5400, 139.6200, 139.7000, 139.7800, \
                         139.8600, 139.9400, 140.0200, 140.1000, 140.1800, \
                         140.2600, 140.3400, 140.4200, 140.5000, 140.5800, \
                         140.6600, 140.7400, 140.8200, 140.9000, 140.9800, \
                         141.0600, 141.1400, 141.2800, 141.5300, 141.8800, 142.3550, 143.0000])
                else:
                    f_CTC = [102., 102.65 , 103.075, 103.375, 103.57 , 103.66, \
                         103.74 , 103.82 , 103.9  , 103.98 , 104.06 , 104.14 , \
                         104.22 , 104.3  , 104.38 , 104.46 , 104.54 , 104.62 , \
                         104.7  , 104.78 , 104.86 , 104.94 , 105.02 , 105.1  , \
                         105.18 , 105.26 , 105.34 , 105.42 , 105.5  , 105.58 , \
                         105.66 , 105.74 , 105.82 , 105.9  , 105.98 , 106.06 , \
                         106.14 , 106.28 , 106.53 , 106.88 , 107.355, 108.   ]
                dfreq_CTC = np.array([0.7500, 0.5000, 0.3500, 0.2500, \
                             0.1400, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.0800, 0.0800, 0.0800, \
                             0.0800, 0.2000, 0.3000, 0.4000, 0.5500, 0.7500])
            else:
                if(used_diag_dict[diag].base_freq_140):
                    f_CTC = np.array([ 135.57, 136.32, 136.82, 137.32, 137.82, 138.12, 138.22, \
                         138.32, 138.42, 138.52, 138.62, 138.72, 138.82, 138.92, \
                         139.02, 139.12, 139.22, 139.32, 139.42, 139.52, 139.62, \
                         139.72, 139.82, 139.92, 140.02, 140.12, 140.22, 140.32, \
                         140.42, 140.52, 140.62, 140.72, 140.82, 140.92, 141.02,
                         141.12, 141.22, 141.32, 141.42, 141.52, 141.62, 141.72, \
                         141.82, 141.92, 142.02, 142.32, 142.82, 143.32, 143.82, \
                         144.57])
                else:
                    f_CTC = np.array([ 100.5 , 101.25, 101.75, 102.25, 102.75, 103.05, 103.15,
                                       103.25, 103.35, 103.45, 103.55, 103.65, 103.75, 103.85,
                                       103.95, 104.05, 104.15, 104.25, 104.35, 104.45, 104.55,
                                       104.65, 104.75, 104.85, 104.95, 105.05, 105.15, 105.25,
                                       105.35, 105.45, 105.55, 105.65, 105.75, 105.85, 105.95,
                                       106.05, 106.15, 106.25, 106.35, 106.45, 106.55, 106.65,
                                       106.75, 106.85, 106.95, 107.25, 107.75, 108.25, 108.75,
                                       109.5 ])
                dfreq_CTC = np.array([ 1. , 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                       0.1, 0.5, 0.5, 0.5, 0.5, 0.5])
            f_CTC = np.array(f_CTC) * 1.e9
            dfreq_CTC = np.array(dfreq_CTC) * 1.e9
            launch["f"] = np.array(f_CTC)
            launch["df"] = np.array(dfreq_CTC)
            launch["R"] = np.zeros(len(launch["f"]))
            launch["phi"] = np.zeros(len(launch["f"]))
            launch["z"] = np.zeros(len(launch["f"]))
            launch["theta_pol"] = np.zeros(len(launch["f"]))
            launch["phi_tor"] = np.zeros(len(launch["f"]))
            launch["dist_focus"] = np.zeros(len(launch["f"]))
            launch["width"] = np.zeros(len(launch["f"]))
            launch["pol_coeff_X"] = np.zeros(len(launch["f"]))
            launch["df"][:] = dfreq_CTC
            launch["R"][:] = np.sqrt(gy_dict[str(used_diag_dict[diag].beamline)].x ** 2 + gy_dict[str(used_diag_dict[diag].beamline)].y ** 2)
            launch["phi"][:] = gy_dict[str(used_diag_dict[diag].beamline)].phi
            launch["z"][:] = gy_dict[str(used_diag_dict[diag].beamline)].z
            launch["pol_coeff_X"][:] = used_diag_dict[diag].pol_coeff_X
            if(np.isscalar(gy_dict[str(used_diag_dict[diag].beamline)].phi_tor)):
                launch["phi_tor"][:] = -gy_dict[str(used_diag_dict[diag].beamline)].phi_tor  # TORBEAM convention
            else:
                launch["phi_tor"][:] = -gy_dict[str(used_diag_dict[diag].beamline)].phi_tor[np.argmin(np.abs(gy_dict[str(used_diag_dict[diag].beamline)].time - time))]  # TORBEAM convention
            if(np.isscalar(gy_dict[str(used_diag_dict[diag].beamline)].theta_pol)):
                launch["theta_pol"][:] = -gy_dict[str(used_diag_dict[diag].beamline)].theta_pol  # TORBEAM convention
            else:
                launch["theta_pol"][:] = -gy_dict[str(used_diag_dict[diag].beamline)].theta_pol[np.argmin(np.abs(gy_dict[str(used_diag_dict[diag].beamline)].time - time))]  # TORBEAM convention
            launch["theta_pol"][:] += used_diag_dict[diag].pol_angle_cor
            launch["phi_tor"][:] += used_diag_dict[diag].tor_angle_cor
            launch["width"][:] = gy_dict[str(used_diag_dict[diag].beamline)].width_y
            launch["dist_focus"][:] = gy_dict[str(used_diag_dict[diag].beamline)].curv_y
        if(diag == "ECN" or diag == "ECO"):
            if(ECI_dict is None):
                raise ValueError("ECI_dict has to present if diag.name is ECN or ECO!")
            phi_ECE = (8.5e0) * 22.5 / 180.0 * np.pi
            launch["f"] = np.copy(ECI_dict["freq"])
            launch["df"] = np.zeros(len(launch["f"]))
            launch["R"] = np.copy(np.sqrt(ECI_dict["x"] ** 2 + ECI_dict["y"] ** 2))
            launch["phi"] = np.copy(np.rad2deg(np.arctan2(ECI_dict["y"], ECI_dict["x"]) + phi_ECE))
            launch["z"] = np.copy(ECI_dict["z"])
            launch["theta_pol"] = np.copy(ECI_dict["pol_ang"])
            launch["phi_tor"] = np.copy(ECI_dict["tor_ang"])
            launch["width"] = np.copy(ECI_dict["w"])
            launch["dist_focus"] = np.copy(ECI_dict["dist_foc"])
#            R_curv = np.copy(ECI_dict["dist_foc"])  # This is a curvature radius !
#            dist_foc = (launch["f"] ** 2 * np.pi ** 2 * R_curv * launch["width"] ** 4) / (cnst.c ** 2 * R_curv ** 2 + launch["f"] ** 2 * np.pi ** 2 * launch["width"] ** 4)
#            launch["dist_focus"][:] = dist_foc
            if(diag == "ECO"):
                launch["df"][:] = 0.7e9
            else:
                launch["df"][:] = 0.39e9
        if(used_diag_dict[diag].name == "EXT"):
            launch["f"] = np.copy(used_diag_dict[diag].f)
            launch["df"] = np.copy(used_diag_dict[diag].df)
            launch["R"] = np.copy(used_diag_dict[diag].R)
            launch["phi"] = np.copy(used_diag_dict[diag].phi)
            launch["z"] = np.copy(used_diag_dict[diag].z)
            launch["theta_pol"] = np.copy(used_diag_dict[diag].theta_pol)
            launch["phi_tor"] = np.copy(used_diag_dict[diag].phi_tor)
            launch["dist_focus"] = np.copy(used_diag_dict[diag].dist_focus)
            launch["width"] = np.copy(used_diag_dict[diag].width)
            launch["pol_coeff_X"] = np.copy(used_diag_dict[diag].pol_coeff_X)
        if("pol_coeff_X" not in launch):
            launch["pol_coeff_X"] = np.zeros(len(launch["f"]))
            launch["pol_coeff_X"][:] = -1  # Means that ECRad will compute the X-mode fraction
        launch["diag_name"] = np.zeros(len(launch["f"]), dtype="|U3")
        launch["diag_name"][:] = used_diag_dict[diag].name
        launch_array.append(launch)
    # The subdivision into individual diagnostics is only relevant for the GUI for ECRad all diagnostics are equal
    # -> Only one launch file for all diagnostics
    if(len(used_diag_dict) > 1):
        flat_launch = dict(launch_array[0])
        for i in range(1, len(used_diag_dict)):
            for key in launch_array[i]:
                flat_launch[key] = np.concatenate([flat_launch[key], launch_array[i][key]])
    else:
        flat_launch = launch_array[0]
    return flat_launch



def write_diag_launch(path, diag_launch):
    launch_file = open(os.path.join(path, 'ray_launch.dat'), 'w')
    launch_file.write("{0: 5d}\n".format(len(diag_launch["f"])))
    for i in range(len(diag_launch["f"])):
        launch_file.write("{0: 1.10E} ".format(diag_launch["f"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["df"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["R"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["z"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi_tor"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["theta_pol"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["width"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["dist_focus"][i]))
        launch_file.write("{0: 1.10E}\n".format(diag_launch["pol_coeff_X"][i]))
    launch_file.flush()
    launch_file.close()
    return diag_launch

def make_reflec_launch(working_dir, shot, time, used_diag_dict, gy_dict=None, ECI_dict=None):
    # This routine prepares the ECRad calculation of the isotropic background radiation temperature
    # At the moment this considers just the normal LOS, hence the approach is very much simplified
    # TO BE IMPROVED
    print("ECRad reflec data will be written for diags ", used_diag_dict)
    if(len(used_diag_dict) > 1):
        raise ValueError("Currently only one concurrent diagnostic supported for reflec_model = 1")
    diag_launch_array = get_diag_launch(working_dir, shot, time, used_diag_dict, gy_dict=gy_dict, ECI_dict=ECI_dict)
    for diag_launch in diag_launch_array:
        launch_file = open(os.path.join(working_dir, "REF_launch.dat"), 'w')
        launch_file.write("{0: 5d}\n".format(len(diag_launch["f"]) + 2))
        i_min_freq = np.argmin(diag_launch["f"] - diag_launch["df"])
        launch_file.write("{0: 1.10E} ".format(diag_launch["f"][i_min_freq] - diag_launch["df"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["df"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["R"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["z"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi_tor"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["theta_pol"][i_min_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["width"][i_min_freq]))
        launch_file.write("{0: 1.10E}\n".format(diag_launch["dist_focus"][i_min_freq]))
        f_sort = np.argsort(diag_launch["f"])
        for i in range(len(diag_launch["f"])):
            launch_file.write("{0: 1.10E} ".format(diag_launch["f"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["df"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["R"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["phi"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["z"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["phi_tor"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["theta_pol"][f_sort][i]))
            launch_file.write("{0: 1.10E} ".format(diag_launch["width"][f_sort][i]))
            launch_file.write("{0: 1.10E}\n".format(diag_launch["dist_focus"][f_sort][i]))
        i_max_freq = np.argmax(diag_launch["f"] + diag_launch["df"])
        launch_file.write("{0: 1.10E} ".format(diag_launch["f"][i_max_freq] + diag_launch["df"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["df"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["R"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["z"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi_tor"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["theta_pol"][i_max_freq]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["width"][i_max_freq]))
        launch_file.write("{0: 1.10E}\n".format(diag_launch["dist_focus"][i_max_freq]))
        launch_file.flush()
        launch_file.close()
    return True

def make_hedgehog_launch(working_dir, f, df, R, phi, z):
    N = 15
    theta_pols = np.linspace(-70, 70, N)
    phi_tors = np.linspace(-70, 70, N)
    diag_launch = {}
    diag_launch["f"] = np.zeros(N * N)
    diag_launch["df"] = np.zeros(len(diag_launch["f"]))
    diag_launch["R"] = np.zeros(len(diag_launch["f"]))
    diag_launch["phi"] = np.zeros(len(diag_launch["f"]))
    diag_launch["z"] = np.zeros(len(diag_launch["f"]))
    diag_launch["theta_pol"] = np.zeros(len(diag_launch["f"]))
    diag_launch["phi_tor"] = np.zeros(len(diag_launch["f"]))
    diag_launch["dist_focus"] = np.zeros(len(diag_launch["f"]))
    diag_launch["width"] = np.zeros(len(diag_launch["f"]))
    diag_launch["f"][:] = f
    diag_launch["df"][:] = df
    diag_launch["R"][:] = R
    diag_launch["phi"][:] = phi
    diag_launch["z"][:] = z
    diag_launch["dist_focus"][:] = 0.e0  # No multiple rays!
    diag_launch["width"][:] = 0.e0  # No multiple rays
    i = 0
    for theta_pol in theta_pols:
        for phi_tor in phi_tors:
            diag_launch["theta_pol"][i] = theta_pol
            diag_launch["phi_tor"][i] = phi_tor
            i += 1
    launch_file = open(os.path.join(working_dir, 'Ext_launch.dat'), 'w')
    launch_file.write("{0: 5d}\n".format(len(diag_launch["f"])))
    for i in range(len(diag_launch["f"])):
        launch_file.write("{0: 1.10E} ".format(diag_launch["f"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["df"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["R"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["z"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["phi_tor"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["theta_pol"][i]))
        launch_file.write("{0: 1.10E} ".format(diag_launch["width"][i]))
        launch_file.write("{0: 1.10E}\n".format(diag_launch["dist_focus"][i]))
    launch_file.flush()
    launch_file.close()

def load_plasma_from_mat(path):
    try:
        plasma_dict = {}
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["time", "R", "z", "eq_R", "eq_z", "Psi_sep", "Psi_ax"]
        at_least_2d_keys = ["rhop_prof", "rhot_prof", "rhop", "Te", "ne", "eq_special", "eq_R", "eq_z", "eq_special"]
        at_least_3d_keys = ["Psi", "Br", "Bt", "Bz"]
        at_least_3d_keys += ["eq_Psi", "eq_Br", "eq_Bt", "eq_Bz"]
        variable_names = at_least_1d_keys + at_least_2d_keys + at_least_3d_keys + ["shot"] + ["vessel_bd"] + ["bt_vac_correction"]
        # print(variable_names)
        try:
            mdict = loadmat(path, chars_as_strings=True, squeeze_me=True, variable_names=variable_names)
        except IOError:
            print("Error: " + path + " does not exist")
            raise IOError
        try:
            plasma_dict["shot"] = mdict["shot"]
        except:
            plasma_dict["shot"] = -1
        increase_diag_dim = False
        increase_time_dim = False
        if(np.isscalar(mdict["time"])):
            plasma_dict["time"] = np.array([mdict["time"]])
            increase_time_dim = True
        else:
            plasma_dict["time"] = mdict["time"]
        for key in mdict:
            if(not key.startswith("_")):  # throw out the .mat specific information
                try:
                    if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                        mdict[key] = np.array([mdict[key]])
                    elif(key in at_least_2d_keys):
                        if(increase_time_dim):
                            mdict[key] = np.array([mdict[key]])
                        if(increase_diag_dim):
                            mdict[key] = np.array([mdict[key]])
                    elif(key in at_least_3d_keys):
                        if(increase_time_dim):
                            mdict[key] = np.array([mdict[key]])
                except Exception as e:
                    print(key)
                    print(e)
        plasma_dict["Te"] = mdict["Te"]
        if(np.min(mdict["Te"]) < 2.e-2):
            print("WARNING: The electron temperatures you provided contain values below room temperature (20 meV)")
            print("INFO: Smallest value: {0:1.3e}".format(np.min(mdict["ne"])))
            print("INFO: ECRad uses logarithmic interpolation for Te")
            print("INFO: All values below 20 meV have been set to 20 meV")
            plasma_dict["Te"][plasma_dict["Te"] < 2.e-2] = 2.e-2
        if(np.min(mdict["ne"]) < 1.e14):
            print("WARNING: The electron densities you provided contain values below 1.e14 m^(-3)")
            print("INFO: Smallest value: {0:1.3e}".format(np.min(mdict["ne"])))
            print("INFO: ECRad uses logarithmic interpolation for ne")
            print("INFO: All values below 1.e14 m^(-3) have been set to 1.e14 m^(-3)")
            plasma_dict["ne"][plasma_dict["ne"] < 1.e14] = 1.e14
        plasma_dict["ne"] = mdict["ne"]
        if(len(plasma_dict["Te"][0].shape) == 1):
            if("rhop_prof" in mdict):
                plasma_dict["rhop_prof"] = mdict["rhop_prof"]
                plasma_dict["prof_reference"] = "rhop_prof"
            else:
                plasma_dict["rhop_prof"] = mdict["rhop"]
            if("rhot_prof" in mdict):
                plasma_dict["rhot_prof"] = mdict["rhot_prof"]
        else:
            plasma_dict["prof_reference"] = "2D"
        # External data should be delivered without additional scaling
        # Otherwise it is not clear whether this means that the data should be scaled or is already scaled by this factor
        plasma_dict["ECE_rhop"] = []
        plasma_dict["ECE_dat"] = []
        plasma_dict["eq_data"] = []
        # TODO remove this place holder by a routine that does this for external equilibriae
        plasma_dict["ECE_mod"] = []  
        EQ_obj = EQDataExt(mdict["shot"], external_folder=os.path.dirname(path), Ext_data=True)
        if("Bt" in mdict):
            EQ_obj.load_slices_from_mat(plasma_dict["time"], mdict)
        else:
            EQ_obj.load_slices_from_mat(plasma_dict["time"], mdict,eq_prefix=True)
        plasma_dict["eq_data_2D"] = EQ_obj
        if("vessel_bd" not in mdict):
            try:
                vessel_bd = np.loadtxt(os.path.join(os.path.dirname(path), "vessel_bd"), skiprows=1)
                plasma_dict["vessel_bd"] = []
                plasma_dict["vessel_bd"].append(vessel_bd.T[0])
                plasma_dict["vessel_bd"].append(vessel_bd.T[1])
                plasma_dict["vessel_bd"] = np.array(plasma_dict["vessel_bd"])
            except IOError as e:
                print("If the vessel boundary is not in the .mat file there should be a file named vessel_bd " + \
                      "in the same folder as the .mat file")
                print("Format: First line - amount of contour points")
                print("Then the R z contour points in two columns [m]")
                raise e
        else:
            plasma_dict["vessel_bd"] = mdict["vessel_bd"]
            if(len(plasma_dict["vessel_bd"]) == 2):
                plasma_dict["vessel_bd"] = plasma_dict["vessel_bd"].T
        plasma_dict["Rwall"] = 0.9  # default in IDA -> make this an input quantity
        plasma_dict["EQ_exp"] = "EXT"
        plasma_dict["EQ_diag"] = "EXT"
        plasma_dict["EQ_ed"] = 0
        return plasma_dict
    except IOError as e:
        print(e)
        print("Could not read external data")
        return None
    except ValueError as e:
        print(e)
        print("Could not read external data")
        return None

def load_from_plasma(filename):
    rootgrp = Dataset(filename, "r", format="NETCDF4")
    plasma_dict = {}
    plasma_dict['shot'] = rootgrp["Plasma"]["shot"][...].item()
    plasma_dict['time'] = np.array(rootgrp["Plasma"]["time"])
    plasma_dict["2D_prof"] = bool(rootgrp["Plasma"]["2D_prof"][...].item())
    plasma_dict["eq_dim"] = rootgrp["Plasma"]["eq_dim"][...].item()
    for sub_key in ["Te", "ne"]:
        plasma_dict[sub_key] = np.array(rootgrp["Plasma"][sub_key])
    if(not plasma_dict["2D_prof"]):
        plasma_dict["prof_reference"] = rootgrp["Plasma"]["prof_reference"][0]
        plasma_dict[plasma_dict["prof_reference"]] = np.array(rootgrp["Plasma"][plasma_dict["prof_reference"]])
    if(plasma_dict["eq_dim"] == 3):
        for sub_key in ["B_ref", "s_plus", "s_max", \
                        "interpolation_acc", "fourier_coeff_trunc", \
                        "h_mesh", "delta_phi_mesh"]:
            plasma_dict["eq_data_3D"][sub_key] = float(rootgrp["Plasma"]["eq_data_3D" + "_"  + sub_key][...].item())
        for sub_key in ["use_mesh", "use_symmetry"]:
            plasma_dict["eq_data_3D"][sub_key] = bool(rootgrp["Plasma"]["eq_data_3D" + "_"  + sub_key][...].item())
        for sub_key in ["equilibrium_type", "vessel_filename"]:
            plasma_dict["eq_data_3D"][sub_key] = rootgrp["Plasma"]["eq_data_3D" + "_"  + sub_key][0]
        plasma_dict["eq_data_3D"]["equilibrium_files"] = np.array(rootgrp["Plasma"]["eq_data_3D" + "_" + \
                                                                                    "equilibrium_files"])
    else:
        plasma_dict['eq_data_2D'] = EQDataExt(plasma_dict["shot"], Ext_data=True)
        eq_slice_data = {}
        for sub_key in ["R", "z", "Psi", "rhop", "Br", "Bt", "Bz",\
                        "R_ax", "z_ax", "R_sep", "z_sep", "Psi_ax", "Psi_sep"]:
            eq_slice_data[sub_key] = np.array(rootgrp["Plasma"]["eq_data_2D" + "_" +  sub_key])
        plasma_dict['eq_data_2D'].fill_with_slices_from_dict(plasma_dict["time"], eq_slice_data)
        plasma_dict['vessel_bd'] = np.array(rootgrp["Plasma"]["vessel_bd"])
    rootgrp.close()
    return plasma_dict

def make_ECRadInputFromPlasmaDict(working_dir, plasma_dict, index, Scenario):
    # In the topfile the dimensions of the matrices are z,R unlike in the GUI where it is R,z -> transpose the matrices here
    if(not Scenario.use3Dscen.used):
        EQ = plasma_dict["eq_data"][index]
        if(plasma_dict["prof_reference"] != "rhop_prof"):
            raise ValueError("Profiles must be given in rho poloidal in case of axissymmetric equlibria")
        error = make_topfile_no_data_load(working_dir, Scenario.shot, plasma_dict["time"][index], EQ.R, EQ.z, EQ.Psi, EQ.Br, \
                                      EQ.Bt, EQ.Bz, EQ.Psi_ax, EQ.Psi_sep) # Routine does the transposing!
        if(error != 0):
            return False
        if(plasma_dict["Te"][index].ndim == 1):
            return make_Te_ne_data(working_dir, Scenario.shot, plasma_dict["time"][index], \
                                   plasma_dict[plasma_dict["prof_reference"]][index] * Scenario.Te_rhop_scale, \
                                   plasma_dict["Te"][index] * Scenario.Te_scale, \
                                   plasma_dict[plasma_dict["prof_reference"]][index] * Scenario.ne_rhop_scale, \
                                   plasma_dict["ne"][index] * Scenario.ne_scale, \
                                   grid=False, EQ=EQ)
        else:
            return make_Te_ne_data(working_dir, Scenario.shot, plasma_dict["time"][index], None, \
                                   plasma_dict["Te"][index] * Scenario.Te_scale, \
                                   plasma_dict["ne"][index]* Scenario.ne_scale, \
                                   grid=True, EQ=EQ)
        return make_Te_ne_data(working_dir, Scenario.shot, plasma_dict["time"][index], \
                               plasma_dict[plasma_dict["prof_reference"]][index], \
                               plasma_dict["Te"][index] * Scenario.Te_scale, \
                               plasma_dict[plasma_dict["prof_reference"]][index] * Scenario.ne_rhop_scale, \
                               plasma_dict["ne"][index] * Scenario.ne_scale, \
                               grid=False)

def make_Te_ne_data(working_dir, shot, time, rho_Te, Te, rho_ne, ne, grid=False, EQ=None):
    if(grid==False):
        make_Te_ne_files(working_dir, rho_Te, Te, rho_ne, ne)
    else:
        print("Creating Te and ne matrix")
        Te_ne_matfile = open(os.path.join(working_dir, "Te_ne_matfile"), "w")
        Te_ne_matfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5d}: {1:1.4f}\n'.format(shot, time))
        Te_ne_matfile.write('   {0: 8d} {1: 8d}\n'.format(len(EQ.R), len(EQ.z)))
        Te_ne_matfile.write('Radial grid coordinates\n')
        cnt = 0
        columns = 8  # number of coloumns
        columns -= 1 # to actually get x columns the variable has to have the value X - 1
        for i in range(len(EQ.R)):
            Te_ne_matfile.write("  {0: 1.8E}".format(EQ.R[i]))
            if(cnt == columns):
                Te_ne_matfile.write("\n")
                cnt = 0
            else:
                cnt += 1
        if(cnt != 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('Vertical grid coordinates\n')
        cnt = 0
        for j in range(len(EQ.z)):
            Te_ne_matfile.write("  {0: 1.8E}".format(EQ.z[j]))
            if(cnt == columns):
                Te_ne_matfile.write("\n")
                cnt = 0
            else:
                cnt += 1
        if(cnt != 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('Te on grid\n')
        cnt = 0
        print("EQ.Bz shape", EQ.Bz.shape)
        print("Te shape", Te.shape)
        for i in range(len(Te)):
            for j in range(len(Te[i])):
                Te_ne_matfile.write("  {0: 1.8E}".format(max(Te.T[i][j], 2.e-2))) # also transpose here
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
        if(cnt != 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('ne on grid\n')
        cnt = 0
        for i in range(len(ne)):
            for j in range(len(ne[i])):
                Te_ne_matfile.write("  {0: 1.8E}".format(max(ne.T[i][j], 1.e12))) # also transpose here
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
    return True

def read_svec_dict_from_file(folder, ich, mode="X"):  # ch no. starts from 1
    # ich is here channel nummer - i.e. channel 1 is the first channel -> add + 1 if ich comes from loop
    if(mode == "O"):
        ch_filename = os.path.join(folder, "chOdata{0:0>3}.dat".format(ich))
        mode = -1
    else:
        ch_filename = os.path.join(folder, "chdata{0:0>3}.dat".format(ich))
        mode = +1
    try:
        svec_block = np.loadtxt(ch_filename)
    except ValueError as e:
        print(e)
        print("Channel ", ich)
        return
    freqs = np.loadtxt(os.path.join(folder, "f_ECE.dat"))
    svec = {}
    svec["s"] = svec_block.T[0][svec_block.T[3] != -1.0]
    svec["R"] = svec_block.T[1][svec_block.T[3] != -1.0]
    svec["z"] = svec_block.T[2][svec_block.T[3] != -1.0]
    svec["rhop"] = svec_block.T[3][svec_block.T[3] != -1.0]
    svec["ne"] = svec_block.T[4][svec_block.T[3] != -1.0]
    svec["Te"] = svec_block.T[5][svec_block.T[3] != -1.0]
    svec["theta"] = svec_block.T[6][svec_block.T[3] != -1.0]
    Abs_obj = EmAbsAlb()
    svec["freq_2X"] = svec_block.T[-1][svec_block.T[3] != -1.0]
    svec["N_abs"] = []
    for i in range(len(svec["s"])):
        svec_cur = SVec(svec["rhop"][i], svec["Te"][i], svec["ne"][i], svec["freq_2X"][i], svec["theta"][i])
        N = Abs_obj.refr_index(svec_cur, freqs[ich - 1] * 2.0 * np.pi, mode)
        svec["N_abs"].append(N)
    svec["N_abs"] = np.array(svec["N_abs"])
    return svec, freqs[ich - 1]

def read_ray_dict_from_file(folder, dist, ich, mode="X", iray=1):
    if(mode == "O"):
        ray_filename = os.path.join(folder, "Ich" + dist, "BPD_ray{0:03d}ch{1:03d}_O.dat".format(iray, ich))
    else:
        ray_filename = os.path.join(folder, "Ich" + dist, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(iray, ich))
    ray_data = np.loadtxt(ray_filename)
    ray_dict = {}
    ray_dict["s"] = ray_data.T[0][ray_data.T[4] != -1.0]
    ray_dict["x"] = ray_data.T[1][ray_data.T[4] != -1.0]
    ray_dict["y"] = ray_data.T[2][ray_data.T[4] != -1.0]
    ray_dict["z"] = ray_data.T[3][ray_data.T[4] != -1.0]
    ray_dict["rhop"] = ray_data.T[4][ray_data.T[4] != -1.0]
    ray_dict["BPD"] = ray_data.T[5][ray_data.T[4] != -1.0]
    ray_dict["BPD_second"] = ray_data.T[6][ray_data.T[4] != -1.0]
    ray_dict["N_ray"] = ray_data.T[9][ray_data.T[4] != -1.0]
    ray_dict["N_cold"] = ray_data.T[10][ray_data.T[4] != -1.0]
    ray_dict["theta"] = ray_data.T[11][ray_data.T[4] != -1.0]
    if(len(ray_data[0]) <= 12):
        spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
        ray_dict["Nx"] = spl.derivative(1)(ray_dict["s"])
        spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
        ray_dict["Ny"] = spl.derivative(1)(ray_dict["s"])
        spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
        ray_dict["Nz"] = spl.derivative(1)(ray_dict["s"])
        norm = ray_dict["N_ray"] / np.sqrt(ray_dict["Nx"] ** 2 + ray_dict["Ny"] ** 2 + ray_dict["Nz"] ** 2)
        ray_dict["Nx"] *= norm
        ray_dict["Ny"] *= norm
        ray_dict["Nz"] *= norm
        ray_dict["Bx"] = None
        ray_dict["By"] = None
        ray_dict["Bz"] = None
    else:
        ray_dict["Nx"] = ray_data.T[12][ray_data.T[4] != -1.0]
        ray_dict["Ny"] = ray_data.T[13][ray_data.T[4] != -1.0]
        ray_dict["Nz"] = ray_data.T[14][ray_data.T[4] != -1.0]
        ray_dict["Bx"] = ray_data.T[15][ray_data.T[4] != -1.0]
        ray_dict["By"] = ray_data.T[16][ray_data.T[4] != -1.0]
        ray_dict["Bz"] = ray_data.T[17][ray_data.T[4] != -1.0]
#    plt.plot(ray_dict["s"], np.sqrt(ray_dict["Nx"] ** 2 + ray_dict["Ny"] ** 2 + ray_dict["Nz"] ** 2), "-")
#    plt.plot(ray_dict["s"], ray_dict["N_ray"], "--")
#    plt.show()
    return ray_dict


if(__name__ == "__main__"):
    # print(get_ECE_launch_info(31539, Diag("ECE", "AUGD", "RMD", 0)))
    make_hedgehog_launch("/tokp/work/sdenk/ECRad2/", 104.e9, 0.2e9, 2.206323000000e+00, 104.e0, 1.531469000000e-01)
