'''
Created on Dec 9, 2015

@author: sdenk
'''
import os
import numpy as np
import sys
sys.path.append("../ECRad_Pylib")
from GlobalSettings import AUG, TCV, itm, ECRadDevPath, ECRadPath, ECRadPathBSUB
from Diags import Diag
if(AUG):
    vessel_file = '../ECRad_Pylib/ASDEX_Upgrade_vessel.txt'
if(TCV):
    vessel_file = '../ECRad_Pylib/TCV_vessel.txt'
from shutil import copy, copyfile, rmtree
from scipy.io import loadmat
import scipy.constants as cnst
if(itm):
    tb_path = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/torbeam"
else:
    tb_path = "/marconi_work/eufus_gw/work/g2sdenk/torbeam/lib-OUT"
from equilibrium_utils import EQDataExt
from electron_distribution_utils import export_gene_fortran_friendly, \
                                        export_gene_bimax_fortran_friendly, \
                                        load_f_from_mat, export_fortran_friendly
from scipy.interpolate import InterpolatedUnivariateSpline
from Geometry_utils import get_Surface_area_of_torus
from shutil import rmtree

def GetECRadExec(Config, Scenario, time):
    # Determine OMP stacksize
    parallel = Config.parallel
    if(Config.debug):
        if(Config.parallel):
            print("No parallel version with debug symbols available at the moment")
            print("Falling back to single core")
            parallel = False
        ECRadVers = ECRadDevPath
    else:
        ECRadVers = ECRadPath
    if(parallel and Config.parallel_cores > 16):
        print("The maximum amount of cores for tokp submission is 16")
        return
    if(parallel):
        stacksize = 0
        cores  = Config.parallel_cores
        for diag in Scenario.used_diags_dict:
            if(diag == "ECN" or diag == "ECO"  or diag == "ECI"):
                stacksize += int(np.ceil(Config.max_points_svec * 3.125) * 3)
            else:
                stacksize += int(np.ceil(Config.max_points_svec * 3.125))
    else:
        cores = 1 # serial
    if(Config.batch):
        os.environ['ECRad_working_dir_1'] = Config.working_dir
        os.environ['ECRad'] = ECRadVers
        launch_options_dict = {}
        launch_options_dict["jobname"] = "-J " + "E{0:5d}{1:1.1f}".format(Scenario.shot, time)
        os.environ['OMP_STACKSIZE'] = "{0:d}k".format(stacksize)
        if(parallel):
            launch_options_dict["partition"] = "--partition=p.tok.openmp"
            launch_options_dict["qos"] = "--qos p.tok.2h"
            if(Config.wall_time > 2):
                launch_options_dict["qos"] = "p.tok.48h"
            launch_options_dict["memory"] = "--mem-per-cpu={0:d}M".format(int(Config.vmem / Config.parallel_cores))
            launch_options_dict["cpus"] = " --cpus-per-task={0:d}".format(cores)
        else:
            launch_options_dict["partition"] = "--partition=s.tok"
            launch_options_dict["qos"] = "--qos s.tok.short"
            if(Config.wall_time > 4):
                launch_options_dict["qos"] = "--qos s.tok.standard"
            if(Config.wall_time > 36):
                launch_options_dict["qos"] = "--qos s.tok.long"
            launch_options_dict["memory"] = "--mem-per-cpu={0:d}M".format(Config.vmem)
            launch_options_dict["cpus"] = " --cpus-per-task={0:d}".format(cores)
        InvokeECRad = "sbatch"
        for key in launch_options_dict:
            InvokeECRad += " " + launch_options_dict[key]
        InvokeECRad += " " + ECRadPathBSUB
    else:
        os.environ['OMP_NUM_THREADS'] = "{0:d}".format(cores)
        InvokeECRad = ECRadVers + " " + Config.working_dir
    return InvokeECRad

def prepare_input_files(Config, Scenario, index, copy_dist=True):
    working_dir = Config.working_dir
    # eq_exp = Config.EQ_exp always exp
    ECRad_data_path = os.path.join(working_dir, "ECRad_data", "")
    if(os.path.isdir(ECRad_data_path)):
    #Get rid of old data -> This ensures that the new data is really new
        rmtree(ECRad_data_path)
    try:
        os.mkdir(ECRad_data_path)
    except OSError:
        print("Failed to create ECRad_data folder in: ", working_dir)
        print("Please check that this folder exists and you have write permissions")
        return False
    print("Created folder " + ECRad_data_path)
    if(Config.dstf != "GB"):
        Ich_path = os.path.join(ECRad_data_path, "Ich" + Config.dstf)
    else:
        Ich_path = os.path.join(ECRad_data_path, "IchGe")
    if(not os.path.isdir(Ich_path)):
        os.mkdir(Ich_path)
        print("Created folder " + Ich_path)
    ray_folder = os.path.join(ECRad_data_path, "ray")
    if(not os.path.isdir(ray_folder)):
        os.mkdir(ray_folder)
        print("Created folder " + ray_folder)
    write_diag_launch(ECRad_data_path, Scenario.ray_launch[index])
    input_file = open(os.path.join(ECRad_data_path, "ECRad.inp"), "w")
    if(Config.dstf == "GB"):
        input_file.write("Ge" + "\n")  # Model does not distinguish between Ge and GB
    else:
        input_file.write(Config.dstf + "\n")
    if(Config.extra_output):
        input_file.write("T\n")
    else:
        input_file.write("F\n")
    if(Config.Te_scale != 1.0):
        print("Te scale != 1 -> scaling Te for model")
    if(Config.ne_scale != 1.0):
        print("ne scale != 1 -> scaling ne for model")
    if((Config.dstf != "Ge" and Config.dstf != "GB")):
        success = make_ECRadInputFromPlasmaDict(ECRad_data_path, Scenario.plasma_dict, index)
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
    input_file.write("{0:1.12E}".format(Config.bt_vac_correction) + "\n")
    input_file.write(str(Config.considered_modes) + "\n")
    input_file.write("{0:1.12E}".format(Config.mode_conv) + "\n")
    input_file.write("{0:1.12E}".format(Config.Te_rhop_scale) + "\n")
    input_file.write("{0:1.12E}".format(Config.ne_rhop_scale) + "\n")
    input_file.write(str(Config.N_freq) + "\n")
    input_file.write(str(Config.N_ray) + "\n")
    input_file.write("{0:1.12E}".format(Config.large_ds) + "\n")
    input_file.write("{0:1.12E}".format(Config.small_ds) + "\n")
    input_file.write("{0:1.12E}".format(Config.R_shift) + "\n")
    input_file.write("{0:1.12E}".format(Config.z_shift) + "\n")
    input_file.write("{0:10d}".format(Config.max_points_svec) + "\n")
    input_file.flush()
    input_file.close()
    fvessel = open(os.path.join(ECRad_data_path, "vessel_bd.txt"), "w")
    fvessel.write("{0: 7d}".format(len(Scenario.plasma_dict["vessel_bd"][0])) + "\n")
    for i in range(len(Scenario.plasma_dict["vessel_bd"][0])):
        fvessel.write("{0: 1.12E} {1: 1.12E}".format(Scenario.plasma_dict["vessel_bd"][0][i], Scenario.plasma_dict["vessel_bd"][1][i]) + "\n")
    fvessel.flush()
    fvessel.close()
    if(Config.dstf == "Re" and copy_dist):
        dist_filename = os.path.join(Config.Relax_dir, "Dist_{0:d}_{1:1.2f}.mat".format(Scenario.shot, Scenario.plasma_dict["time"][index]))
        if(os.path.isfile(dist_filename)):
            dist_obj = load_f_from_mat(dist_filename)
            fRe_dir = os.path.join(ECRad_data_path, "fRe")
            os.mkdir(fRe_dir)
            export_fortran_friendly([dist_obj, fRe_dir])
        else:
            print(dist_filename + " does not exist!")
            return False
    if(Config.dstf == "Ge" and copy_dist):
        wpath = os.path.join(os.path.join(Config.working_dir, "ECRad_data", "fGe"))
        if os.path.exists(wpath):
            rmtree(wpath)  # Removing the old files first is faster than overwriting them
        os.mkdir(wpath)
        export_gene_fortran_friendly(wpath, Config.gene_obj[index].rhop, Config.gene_obj[index].beta_par, \
                                     Config.gene_obj[index].mu_norm, Config.gene_obj[index].ne, \
                                     Config.gene_obj[index].f, Config.gene_obj[index].f0, \
                                     Config.gene_obj[index].B0)
    if(Config.dstf == "GB" and copy_dist):
        wpath = os.path.join(os.path.join(Config.working_dir, "ECRad_data", "fGe"))
        if os.path.exists(wpath):
            rmtree(wpath)  # Removing the old files first is faster than overwriting them
        os.mkdir(wpath)
        Config.gene_obj[index].make_bi_max()
        export_gene_bimax_fortran_friendly(wpath, Config.gene_obj[index].rhop, Config.gene_obj[index].beta_par, \
                                     Config.gene_obj[index].mu_norm, Config.gene_obj[index].Te, Config.gene_obj[index].ne, \
                                     Config.gene_obj[index].Te_perp, Config.gene_obj[index].Te_par, Config.gene_obj[index].B0)

    return True


def get_ECE_launch_info(shot, diag):
    from shotfile_handling_AUG import get_ECE_launch_params
    from equilibrium_utils_AUG import EQData
    from geo_los import geo_los
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
                ECE_launch["phi"][ich] = 0.04
            elif (ECE_launch["waveguide"][ich] == 11 or ECE_launch["waveguide"][ich] == 3):
                ECE_launch["phi_tor"][ich] = -0.7265
                ECE_launch["phi"][ich] = -0.04
            else:
                print("subroutine make_theta_los: something wrong with wg(ich) for shotno. > 33724!")
                print("wg", ECE_launch["waveguide"][ich])
                raise ValueError
        if(ECE_launch["waveguide"][ich] != wg_last):
            R, z = geo_los(shot, ECE_launch["waveguide"][ich], ECE_launch["z_lens"], R, z)
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
    return ECE_launch

def get_diag_launch(shot, time, used_diag_dict, gy_dict=None, ECI_dict=None):
    print("ECRad data will be written for diags ", used_diag_dict.keys())
    launch_array = []
    for diag in used_diag_dict.keys():
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
            launch["width"][:] = gy_dict[str(used_diag_dict[diag].beamline)].width_y
            launch["dist_focus"][:] = gy_dict[str(used_diag_dict[diag].beamline)].curv_y
#            R_curv = gy_dict[str(used_diag_dict[diag].beamline)].curv_y
#            dist_foc = (launch["f"] ** 2 * np.pi ** 2 * R_curv * launch["width"] ** 4) / (cnst.c ** 2 * R_curv ** 2 + launch["f"] ** 2 * np.pi ** 2 * launch["width"] ** 4)
#            launch["dist_focus"][:] = dist_foc
#        if(used_diag_dict[diag].name == "VCE"):
#            launch["f"] = np.array([104.9, 106.6, 108.1, 109.8, 111.3, 112.9, \
#                              133.5, 136.4, 139.3, 141.9, 144.9, 147.5]) * 1.e9
#            launch["df"] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, \
#                                  0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]) * 1.e9
#            launch["R"] = np.zeros(len(launch["f"]))
#            launch["phi"] = np.zeros(len(launch["f"]))
#            launch["z"] = np.zeros(len(launch["f"]))
#            launch["theta_pol"] = np.zeros(len(launch["f"]))
#            launch["phi_tor"] = np.zeros(len(launch["f"]))
#            launch["dist_focus"] = np.zeros(len(launch["f"]))
#            launch["width"] = np.zeros(len(launch["f"]))
#            if(AUG):
#                launch["R"][:] = 0.88 * used_diag_dict[diag].R_scale
#                launch["phi"][:] = 0.0
#                launch["z"][:] = 0.99 * used_diag_dict[diag].z_scale
#            else:
#                launch["R"][:] = 0.88
#                launch["phi"][:] = 0.0
#                launch["z"][:] = 0.99
#            launch["phi_tor"][:] = 1.e-1
#            launch["theta_pol"][:] = 90.e0
#            launch["width"][:] = 0.03e0  # assuming a very parallel beam
#            launch["dist_focus"][:] = 10.0
#        if(used_diag_dict[diag].diag == "LCE" or used_diag_dict[diag].diag == "UCE"):
#            launch["f"] = np.array([67.6, 69.1, 70.5, 72. , 73.4, 74.9, 76.3, 77.8,
#                              79.2, 80.7, 82.2, 85.5, 86.9, 88.4, 89.8, 91.3,
#                              92.7, 94.2, 95.7, 97.1, 98.6, 100. , 101.5]) * 1.e9
#            launch["df"] = np.zeros(len(launch["f"]))
#            launch["R"] = np.zeros(len(launch["f"]))
#            launch["phi"] = np.zeros(len(launch["f"]))
#            launch["z"] = np.zeros(len(launch["f"]))
#            launch["theta_pol"] = np.zeros(len(launch["f"]))
#            launch["phi_tor"] = np.zeros(len(launch["f"]))
#            launch["dist_focus"] = np.zeros(len(launch["f"]))
#            launch["width"] = np.zeros(len(launch["f"]))
#            launch["df"][:] = 0.75e9
#            launch["R"][:] = 1.15
#            launch["phi"][:] = 0.0
#            if(used_diag_dict[diag].diag == "LCE"):
#                launch["z"][:] = 0.e0
#                launch["width"][:] = 0.0351e0
#                launch["dist_focus"][:] = -0.33e0  # Beam is not focused
#            else:
#                launch["z"][:] = 0.21e0
#                launch["width"][:] = 0.0211e0
#                launch["dist_focus"][:] = -0.257e0  # Beam is not focused
#            launch["phi_tor"][:] = 1.e-1
#            launch["theta_pol"][:] = 0.e0
        if(diag == "ECN" or diag == "ECO"):
            if(ECI_dict is None):
                raise ValueError("ECI_dict has to present if diag.name is ECN or ECO!")
            phi_ECE = (8.5e0) * 22.5 / 180.0 * np.pi
            launch["f"] = np.copy(ECI_dict["freq_ECI_in"])
            launch["df"] = np.zeros(len(launch["f"]))
            launch["R"] = np.copy(np.sqrt(ECI_dict["x"] ** 2 + ECI_dict["y"] ** 2))
            launch["phi"] = np.copy(np.rad2deg(np.arctan2(ECI_dict["y"], ECI_dict["x"]) + phi_ECE))
            launch["z"] = np.copy(ECI_dict["z"])
            launch["theta_pol"] = np.copy(ECI_dict["pol_ang"])
            launch["phi_tor"] = np.copy(ECI_dict["tor_ang"])
            launch["width"][:] = np.copy(ECI_dict["w"])
            launch["dist_focus"][:] = np.copy(ECI_dict["dist_foc"])
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
#        if(used_diag_dict[diag].diag == "CCE"):
#            if(type(time) != float):
#                time = float(time)
#            launch_geo = used_diag_dict[diag].get_launch_geo(time)
#            launch["f"] = np.copy(launch_geo[0])
#            launch["df"] = np.copy(launch_geo[1])
#            launch["R"] = np.copy(launch_geo[2])
#            launch["phi"] = np.copy(launch_geo[3])
#            launch["z"] = np.copy(launch_geo[4])
#            launch["phi_tor"] = np.copy(launch_geo[5])
#            launch["theta_pol"] = np.copy(launch_geo[6])
#            launch["width"] = np.copy(launch_geo[7])
#            launch["dist_focus"] = np.copy(launch_geo[8])
        if("pol_coeff_X" not in launch.keys()):
            launch["pol_coeff_X"] = np.zeros(len(launch["f"]))
            launch["pol_coeff_X"][:] = -1  # Means that ECRad will compute the X-mode fraction
        launch["diag_name"] = np.zeros(len(launch["f"]), dtype="|S3")
        launch["diag_name"][:] = used_diag_dict[diag].name
        launch_array.append(launch)
    # The subdivision into individual diagnostics is only relevant for the GUI for ECRad all diagnostics are equal
    # -> Only one launch file for all diagnostics
    if(len(used_diag_dict.keys()) > 1):
        flat_launch = dict(launch_array[0])
        for i in range(1, len(used_diag_dict.keys())):
            for key in launch_array[i].keys():
                flat_launch[key] = np.concatenate([flat_launch[key], launch_array[i][key]])
    else:
        flat_launch = launch_array[0]
    return flat_launch

def write_diag_launch(path, diag_launch):
    print("ECRad data will be written for diags ", list(set(diag_launch["diag_name"])))
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
    print("ECRad reflec data will be written for diags ", used_diag_dict.keys())
    if(len(used_diag_dict.keys()) > 1):
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

def make_vessel_plasma_ratio(index, Config):
    if(Config.Ext_plasma):
        EQ_obj = EQData(Config.shot, external_folder=None, bt_vac_correction=Config.bt_vac_correction)
        EQ_obj.insert_slices_from_ext(Config.time[index], Config.plasma_dict["eq_data"][index])
        R_vessel = Config.plasma_dict["vessel_bd"][0]
        z_vessel = Config.plasma_dict["vessel_bd"][0]
        raise ValueError("Not implemented yet for ExtPlasmas")
    else:
        EQ_obj = EQData(Config.shot, EQ_exp=Config.EQ_exp, EQ_diag=Config.EQ_diag, EQ_ed=Config.EQ_ed, bt_vac_correction=Config.bt_vac_correction)
        vessel_data = np.loadtxt(vessel_file, skiprows=1)
        R_vessel = vessel_data.T[0]
        z_vessel = vessel_data.T[1]
    A_torus = get_Surface_area_of_torus(R_vessel, z_vessel)
    A_plasma = EQ_obj.get_surface_area(Config.time[index], 0.99)
    print("Wall surface area", A_torus)
    print("Plasma surface area", A_plasma)
    return A_torus / A_plasma

def make_reflec_Trad(index, Config, f_reflec, Trad_X_reflec, tau_X_reflec, Trad_O_reflec, tau_O_reflec):
    # Currently broken  DO NOT USE !!!
    # Simplified improved wall reflection model - no mode conversion (adapted from W. H. M. Clark, 1983, Plasma Phys. 23 1501
    ECRad_data_path = os.path.join(Config.working_dir, "ECRad_data", "")
    vessel_plasma_ratio = make_vessel_plasma_ratio(index, Config)
    if(Config.considered_modes == 1):
        Trad_X_reflected = Trad_X_reflec / (vessel_plasma_ratio * (1.e0 - Config.reflec_X) + (1.0 - np.exp(-tau_X_reflec)))
        Trad_X_reflec_file = open(os.path.join(ECRad_data_path, "X_reflec_Trad.dat"), "w")
        Trad_X_reflec_file.write("{0: 5d}\n".format(len(Trad_X_reflected)))
        for i in range(len(Trad_X_reflec)):
            print("tau_reflect - T_rad X reflected / Trad_X {0:1.3f} - {1:1.3f}".format(tau_X_reflec[i], Trad_X_reflected[i] / Trad_X_reflec[i]))
            Trad_X_reflec_file.write("{0: 1.10E} {1: 1.10E}\n".format(f_reflec[i], Trad_X_reflected[i] * 1.e3))  # keV -> eV
        Trad_X_reflec_file.flush()
        Trad_X_reflec_file.close()
    elif(Config.considered_modes == 2):
        Trad_O_reflected = Trad_O_reflec / (vessel_plasma_ratio * (1.e0 - Config.reflec_O) + (1.0 - np.exp(-tau_O_reflec)))
        Trad_O_reflec_file = open(os.path.join(ECRad_data_path, "O_reflec_Trad.dat"), "w")
        Trad_O_reflec_file.write("{0: 5d}\n".format(len(Trad_O_reflected)))
        for i in range(len(Trad_O_reflec)):
            Trad_O_reflec_file.write("{0: 1.10E} {1: 1.10E}\n".format(f_reflec[i], Trad_O_reflected[i] * 1.e3))  # keV -> eV
        Trad_O_reflec_file.flush()
        Trad_O_reflec_file.close()
    else:
        Trad_X_reflected = Trad_X_reflec / (vessel_plasma_ratio * (1.e0 - Config.reflec_X) + (1.0 - np.exp(-tau_X_reflec)))
        Trad_O_reflected = Trad_O_reflec / (vessel_plasma_ratio * (1.e0 - Config.reflec_O) + (1.0 - np.exp(-tau_O_reflec)))
        Trad_X_mix = (1.0 - Config.mode_conv) * Trad_X_reflected + Config.mode_conv * Trad_O_reflected
        Trad_O_mix = (1.0 - Config.mode_conv) * Trad_O_reflected + Config.mode_conv * Trad_X_reflected
        Trad_X_reflec_file = open(os.path.join(ECRad_data_path, "X_reflec_Trad.dat"), "w")
        Trad_X_reflec_file.write("{0: 5d}\n".format(len(Trad_X_mix)))
        for i in range(len(Trad_X_mix)):
            print("tau_reflect - T_rad X reflected / Trad_X {0:1.3f} - {1:1.3f}".format(tau_X_reflec[i], Trad_X_mix[i] / Trad_X_reflec[i]))
            Trad_X_reflec_file.write("{0: 1.10E} {1: 1.10E}\n".format(f_reflec[i], Trad_X_mix[i] * 1.e3))  # keV -> eV
        Trad_X_reflec_file.flush()
        Trad_X_reflec_file.close()
        Trad_O_reflec_file = open(os.path.join(ECRad_data_path, "O_reflec_Trad.dat"), "w")
        Trad_O_reflec_file.write("{0: 5d}\n".format(len(Trad_O_mix)))
        for i in range(len(Trad_O_mix)):
            Trad_O_reflec_file.write("{0: 1.10E} {1: 1.10E}\n".format(f_reflec[i], Trad_O_mix[i] * 1.e3))  # keV -> eV
        Trad_O_reflec_file.flush()
        Trad_O_reflec_file.close()

def load_and_validate_external_plasma(ECRadConfig):
    try:
        plasma_dict = {}
        ext_data_folder = os.path.join(ECRadConfig.working_dir, "Ext_data")
        time = np.loadtxt(os.path.join(ext_data_folder, "t"), dtype=np.double, ndmin=1)
        plasma_dict["Te"] = []
        plasma_dict["ne"] = []
        plasma_dict["rhop"] = []
        plasma_dict["ne_rhop_scale"] = np.zeros(len(time))
        plasma_dict["ne_rhop_scale"][:] = 1.0
        plasma_dict["ECE_rhop"] = []
        plasma_dict["ECE_dat"] = []
        plasma_dict["eq_data"] = []
        # TODO remove this place holder by a routine that does this for external equilibriae
        index = 0
        plasma_dict["ECE_mod"] = []
        EQ_obj = EQDataExt(ECRadConfig.shot, external_folder=ext_data_folder, bt_vac_correction=ECRadConfig.bt_vac_correction)
        for t in time:
            plasma_dict["eq_data"].append(EQ_obj.read_EQ_from_Ext_single_slice(t, index))
            Te_data = np.loadtxt(os.path.join(ext_data_folder, "Te{0:d}".format(index)))
            ne_data = np.loadtxt(os.path.join(ext_data_folder, "ne{0:d}".format(index)))
            if(not ECRadConfig.Ext_Grid):
                plasma_dict["Te"].append(Te_data.T[1])
                plasma_dict["rhop"].append(Te_data.T[0])
                plasma_dict["ne"].append(ne_data.T[1])
                if(not np.all(ne_data.T[0] == Te_data.T[0])):
                    print("ERROR: The rhoploidal axis for both Te and ne have to be identical")
                    raise(IOError)
            else:
                plasma_dict["rhop"].append([])
                plasma_dict["Te"].append(Te_data.T)  # Right alignement for topfile
                plasma_dict["ne"].append(ne_data.T)
            plasma_dict["ECE_mod"].append([])
            index += 1
        plasma_dict["Te"] = np.array(plasma_dict["Te"])
        plasma_dict["rhop"] = np.array(plasma_dict["rhop"])
        plasma_dict["eq_data"] = np.array(plasma_dict["eq_data"])
        plasma_dict["rhop"] = np.array(plasma_dict["rhop"])
        plasma_dict["ne"] = np.array(plasma_dict["ne"])
        plasma_dict["RwallX"] = ECRadConfig.reflec_X  # default in IDA -> make this an input quantity
        plasma_dict["RwallO"] = ECRadConfig.reflec_O  # default in IDA -> make this an input quantity
        try:
            plasma_dict["vessel_bd"] = np.loadtxt(os.path.join(ext_data_folder, "Ext_vessel.bd"), skiprows=1).T
        except IOError:
            print("External vessel file not found - falling back to default file located at: ", vessel_file)
            plasma_dict["vessel_bd"] = np.loadtxt(os.path.join(vessel_file), skiprows=1).T
        if(len(plasma_dict["vessel_bd"][0]) < 40 or len(plasma_dict["vessel_bd"][0]) > 80):
            s = np.linspace(0.0, 1.0, len(plasma_dict["vessel_bd"][0]))
            R_spl = InterpolatedUnivariateSpline(s, plasma_dict["vessel_bd"][0])
            z_spl = InterpolatedUnivariateSpline(s, plasma_dict["vessel_bd"][1])
            s_short = np.linspace(0.0, 1.0, 60)
            plasma_dict["vessel_bd"] = np.array([R_spl(s_short), z_spl(s_short)])
        if(not ECRadConfig.Ext_Grid):
            plasma_dict["eq_diag"] = "Ext"
            ECRadConfig.EQ_diag = "Ext"
        else:
            plasma_dict["eq_diag"] = "E2D"
            ECRadConfig.EQ_diag = "E2D"
        plasma_dict["eq_ed"] = 0
        return time, plasma_dict
    except IOError as e:
        print(e)
        print("Could not read external data")
        return None, None

def load_plasma_from_mat(path):
    try:
        plasma_dict = {}
        # Loading from .mat sometimes adds single entry arrays that we don't want
        at_least_1d_keys = ["t", "R", "z", "Psi_sep", "Psi_ax"]
        at_least_2d_keys = ["rhop_prof", "rhop", "Te", "ne"]
        at_least_3d_keys = ["Psi", "Br", "Bt", "Bz"]
        variable_names = at_least_1d_keys + at_least_2d_keys + at_least_3d_keys + ["shotnum"]
        # print(variable_names)
        try:
            mdict = loadmat(path, chars_as_strings=True, squeeze_me=True, variable_names=variable_names)
        except IOError:
            print("Error: " + path + " does not exist")
            raise IOError
        print(mdict.keys())
        plasma_dict["shot"] = mdict["shotnum"]
        increase_diag_dim = False
        increase_time_dim = False
        if(np.isscalar(mdict["t"])):
            plasma_dict["time"] = np.array([mdict["t"]])
            increase_time_dim = True
        else:
            plasma_dict["time"] = mdict["t"]
        for key in mdict.keys():
            if(not key.startswith("_")):  # throw out the .mat specific information
                try:
                    if(key in at_least_1d_keys and np.isscalar(mdict[key])):
                        mdict[key] = np.array([mdict[key]])
                    elif(key in at_least_2d_keys):
                        if(increase_time_dim):
                            mdict[key] = np.array([mdict[key]])
                        elif(increase_time_dim):
                            for i in range(len(mdict[key])):
                                mdict[key][i] = np.array([mdict[key][i]])
                        if(increase_diag_dim):
                            mdict[key] = np.array([mdict[key]])
                    elif(key in at_least_3d_keys):
                        if(increase_time_dim):
                            mdict[key] = np.array([mdict[key]])
                except Exception as e:
                    print(key)
                    print(e)
        for key in at_least_3d_keys:
            mdict[key] = np.swapaxes(mdict[key], 2, 0)
            mdict[key] = np.swapaxes(mdict[key], 1, 2)
        for key in at_least_2d_keys:
            mdict[key] = np.swapaxes(mdict[key], 0, 1)
        plasma_dict["Te"] = mdict["Te"] * 1.e3
        plasma_dict["ne"] = mdict["ne"]
        if(len(plasma_dict["Te"][0].shape) == 1):
            if("rhop_prof" in mdict.keys()):
                plasma_dict["rhop_prof"] = mdict["rhop_prof"]
            else:
                plasma_dict["rhop_prof"] = mdict["rhop"]
        plasma_dict["ne_rhop_scale"] = np.zeros(len(plasma_dict["time"]))
        plasma_dict["ne_rhop_scale"][:] = 1.0
        plasma_dict["ECE_rhop"] = []
        plasma_dict["ECE_dat"] = []
        plasma_dict["eq_data"] = []
        # TODO remove this place holder by a routine that does this for external equilibriae
        plasma_dict["ECE_mod"] = []
        EQ_obj = EQDataExt(mdict["shotnum"], external_folder=os.path.dirname(path), bt_vac_correction=1.0, Ext_data=True)
        EQ_obj.load_slices_from_mat(plasma_dict["time"], mdict)
        plasma_dict["eq_data"] = EQ_obj.slices
        if("vessel_bd" not in mdict.keys()):
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
        plasma_dict["Rwall"] = 0.9  # default in IDA -> make this an input quantity
        plasma_dict["eq_exp"] = "EXT"
        plasma_dict["eq_diag"] = "EXT"
        plasma_dict["eq_ed"] = 0
        return plasma_dict
    except IOError as e:
        print(e)
        print("Could not read external data")
        return None
    except ValueError as e:
        return None

def make_topfile(working_dir, shot, time, EQ_t):
    # Note this routine uses MBI-BTFABB for a correction of the toroidal magnetic field
    # Furthermore, an empirical vacuum BTF correction factor of bt_vac_correction is applied
    # This creates a topfile that is consistent with the magnetic field used in OERT
    print("Creating topfile for #{0:d} t = {1:1.2f}".format(shot, time))
    columns = 8  # number of coloumns
    columns -= 1
    print("Magnetic axis position: ", "{0:1.3f}".format(EQ_t.R_ax))
    topfile = open(os.path.join(working_dir, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5n}: {1:1.4f}\n'.format(shot, time))
    topfile.write('   {0: 8n} {1: 8n}\n'.format(len(EQ_t.R), len(EQ_t.z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.8E}  {1: 1.8E}  {2: 1.8E}'.format(EQ_t.R[0], EQ_t.R[-1], \
        EQ_t.Psi_sep))  # 1.0000
    # Normalize PSI
    topfile.write('\n')
    topfile.write('Radial grid coordinates\n')
    cnt = 0
    for i in range(len(EQ_t.R)):
        topfile.write("  {0: 1.8E}".format(EQ_t.R[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    topfile.write('\n')
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for i in range(len(EQ_t.z)):
        topfile.write("  {0: 1.8E}".format(EQ_t.z[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    # ivR = np.argmin(np.abs(pfm_dict["Ri"] - rv))
    # jvz = np.argmin(np.abs(pfm_dict["zj"] - vz))
    # plt.plot(pfm_dict["Ri"],B_t[0], "^", label = "EQH B")
    # print("BTFABB correction",Btf0, Btf0_eq )
    # print("R,z",pfm_dict["Ri"][ivR],pfm_dict["zj"][jvz])
    B_r = EQ_t.Br.T  # in topfile R is the small index (i.e. second index in C) and z the large index (i.e. first index in C)
    B_t = EQ_t.Bt.T  # in topfile z comes first regardless of the arrangement
    B_z = EQ_t.Bz.T  # in topfile z comes first regardless of the arrangement
    Psi = EQ_t.Psi.T  # in topfile z comes first regardless of the arrangement
    if(cnt != columns):
        topfile.write('\n')
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(B_r)):
        for j in range(len(B_r[i])):
            topfile.write("  {0: 1.8E}".format(B_r[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != columns):
        topfile.write('\n')
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(B_t)):
        for j in range(len(B_t[i])):
            topfile.write("  {0: 1.8E}".format(B_t[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != columns):
        topfile.write('\n')
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(B_z)):
        for j in range(len(B_z[i])):
            topfile.write("  {0: 1.8E}".format(B_z[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != columns):
        topfile.write('\n')
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(Psi)):
        for j in range(len(Psi[i])):
            topfile.write("  {0: 1.8E}".format(Psi[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.flush()
    topfile.close()
    print("topfile successfully written to", os.path.join(working_dir, "topfile"))
    return 0

def make_ECRadInputFromPlasmaDict(working_dir, plasma_dict, index):
    # In the topfile the dimensions of the matrices are z,R unlike in the GUI where it is R,z -> transpose the matrices here
    columns = 5  # number of coloumns
    columns -= 1
    EQ = plasma_dict["eq_data"][index]
    print("Magnetic axis position: ", "{0:1.3f}".format(EQ.special[0]))
    topfile = open(os.path.join(working_dir, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points:\n')
    topfile.write('   {0: 8d} {1: 8d}\n'.format(len(EQ.R), len(EQ.z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.8E}  {1: 1.8E}  {2: 1.8E}'.format(EQ.R[0], EQ.R[-1], \
        EQ.special[1]))
    topfile.write('\n')
    topfile.write('Radial grid coordinates\n')
    cnt = 0
    for i in range(len(EQ.R)):
        topfile.write("  {0: 1.8E}".format(EQ.R[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for j in range(len(EQ.z)):
        topfile.write("  {0: 1.8E}".format(EQ.z[j]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(EQ.Br.T)):
        for j in range(len(EQ.Br.T[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Br.T[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(EQ.Bt.T)):
        for j in range(len(EQ.Bt.T[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Bt.T[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(EQ.Bz.T)):
        for j in range(len(EQ.Bz.T[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Bz.T[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(EQ.Psi.T)):
        for j in range(len(EQ.Psi.T[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Psi.T[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.flush()
    topfile.close()
    ne = plasma_dict["ne"][index]
    Te = plasma_dict["Te"][index]
    if(len(Te.shape) == 1):
        rhop = plasma_dict["rhop_prof"][index]
        Te_file = open(os.path.join(working_dir, "Te_file.dat"), "w")
        Te_tb_file = open(os.path.join(working_dir, "Te.dat"), "w")
        lines = 150
        Te_file.write("{0: 7d}".format(lines) + "\n")
        Te_tb_file.write("{0: 7d}".format(lines) + "\n")
        Te_spline = InterpolatedUnivariateSpline(rhop, Te, k=1)
        rhop_short = np.linspace(np.min(rhop), np.max(rhop), lines)
        for i in range(len(rhop_short)):
            try:
                Te_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], Te_spline(rhop_short[i]).item()) + "\n")
                Te_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], Te_spline(rhop_short[i]).item() / 1.e03) + "\n")
            except ValueError:
                print(rhop_short[i], Te_spline(rhop_short[i]))
                raise(ValueError)
        Te_file.flush()
        Te_file.close()
        Te_tb_file.flush()
        Te_tb_file.close()
        ne_file = open(os.path.join(working_dir, "ne_file.dat"), "w")
        ne_tb_file = open(os.path.join(working_dir, "ne.dat"), "w")
        lines = 150
        ne_file.write("{0: 7d}".format(lines) + "\n")
        ne_tb_file.write("{0: 7d}".format(lines) + "\n")
        ne_spline = InterpolatedUnivariateSpline(rhop, ne, k=1)
        rhop_short = np.linspace(np.min(rhop), np.max(rhop), lines)
        for i in range(len(rhop_short)):
            ne_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], ne_spline(rhop_short[i]).item()) + "\n")
            ne_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], ne_spline(rhop_short[i]).item() / 1.e19) + "\n")
        ne_file.flush()
        ne_file.close()
        ne_tb_file.flush()
        ne_tb_file.close()
    else:
        Te_ne_matfile = open(os.path.join(working_dir, "Te_ne_matfile"), "w")
        Te_ne_matfile.write('Number of radial and vertical grid points: \n')
        Te_ne_matfile.write('   {0: 8d} {1: 8d}\n'.format(len(EQ.R), len(EQ.z)))
        Te_ne_matfile.write('Radial grid coordinates\n')
        cnt = 0
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
        for i in range(len(Te.T)):
            for j in range(len(Te.T[i])):
                Te_ne_matfile.write("  {0: 1.8E}".format(Te.T[i][j]))
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
        if(cnt != 0):
            Te_ne_matfile.write('\n')
        Te_ne_matfile.write('ne on grid\n')
        cnt = 0
        for i in range(len(ne.T)):
            for j in range(len(ne.T[i])):
                Te_ne_matfile.write("  {0: 1.8E}".format(ne.T[i][j]))
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
    return True

def make_topfile_from_ext_data(working_dir, shot, time, EQ, rhop, Te, ne, grid=False):
    columns = 5  # number of coloumns
    columns -= 1
    print("Magnetic axis position: ", "{0:1.3f}".format(EQ.special[0]))
    topfile = open(os.path.join(working_dir, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5d}: {1:1.4f}\n'.format(shot, time))
    topfile.write('   {0: 8d} {1: 8d}\n'.format(len(EQ.R), len(EQ.z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.8E}  {1: 1.8E}  {2: 1.8E}'.format(EQ.R[0], EQ.R[-1], \
        EQ.special[1]))
    topfile.write('\n')
    topfile.write('Radial grid coordinates\n')
    cnt = 0
    for i in range(len(EQ.R)):
        topfile.write("  {0: 1.8E}".format(EQ.R[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for j in range(len(EQ.z)):
        topfile.write("  {0: 1.8E}".format(EQ.z[j]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(EQ.Br)):
        for j in range(len(EQ.Br[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Br[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(EQ.Bt)):
        for j in range(len(EQ.Bt[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Bt[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(EQ.Bz)):
        for j in range(len(EQ.Bz[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Bz[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    if(cnt != 0):
        topfile.write('\n')
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(EQ.Psi)):
        for j in range(len(EQ.Psi[i])):
            topfile.write("  {0: 1.8E}".format(EQ.Psi[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.flush()
    topfile.close()
    print("topfile successfully written to", os.path.join(working_dir, "topfile"))
    if(not grid):
        print("Copying Te and ne profile")
        Te_file = open(os.path.join(working_dir, "Te_file.dat"), "w")
        Te_tb_file = open(os.path.join(working_dir, "Te.dat"), "w")
        lines = 150
        Te_file.write("{0: 7d}".format(lines) + "\n")
        Te_tb_file.write("{0: 7d}".format(lines) + "\n")
        Te_spline = InterpolatedUnivariateSpline(rhop, Te, k=1)
        rhop_short = np.linspace(np.min(rhop), np.max(rhop), lines)
        for i in range(len(rhop_short)):
            try:
                Te_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], Te_spline(rhop_short[i]).item()) + "\n")
                Te_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], Te_spline(rhop_short[i]).item() / 1.e03) + "\n")
            except ValueError:
                print(rhop_short[i], Te_spline(rhop_short[i]))
                raise(ValueError)
        Te_file.flush()
        Te_file.close()
        Te_tb_file.flush()
        Te_tb_file.close()
        ne_file = open(os.path.join(working_dir, "ne_file.dat"), "w")
        ne_tb_file = open(os.path.join(working_dir, "ne.dat"), "w")
        lines = 150
        ne_file.write("{0: 7d}".format(lines) + "\n")
        ne_tb_file.write("{0: 7d}".format(lines) + "\n")
        ne_spline = InterpolatedUnivariateSpline(rhop, ne, k=1)
        rhop_short = np.linspace(np.min(rhop), np.max(rhop), lines)
        for i in range(len(rhop_short)):
            ne_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], ne_spline(rhop_short[i]).item()) + "\n")
            ne_tb_file.write("{0: 1.12E} {1: 1.12E}".format(rhop_short[i], ne_spline(rhop_short[i]).item() / 1.e19) + "\n")
        ne_file.flush()
        ne_file.close()
        ne_tb_file.flush()
        ne_tb_file.close()
    else:
        print("Copying Te and ne matrix")
        Te_ne_matfile = open(os.path.join(working_dir, "Te_ne_matfile"), "w")
        Te_ne_matfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5d}: {1:1.4f}\n'.format(shot, time))
        Te_ne_matfile.write('   {0: 8d} {1: 8d}\n'.format(len(EQ.R), len(EQ.z)))
        Te_ne_matfile.write('Radial grid coordinates\n')
        cnt = 0
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
                Te_ne_matfile.write("  {0: 1.8E}".format(Te[i][j]))
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
                Te_ne_matfile.write("  {0: 1.8E}".format(ne[i][j]))
                if(cnt == columns):
                    Te_ne_matfile.write("\n")
                    cnt = 0
                else:
                    cnt += 1
    return 0

if(__name__ == "__main__"):
    # print(get_ECE_launch_info(31539, Diag("ECE", "AUGD", "RMD", 0)))
    make_hedgehog_launch("/tokp/work/sdenk/ECRad2/", 104.e9, 0.2e9, 2.206323000000e+00, 104.e0, 1.531469000000e-01)
