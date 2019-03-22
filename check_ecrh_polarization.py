'''
Created on Dec 6, 2017

@author: sdenk
'''
import subprocess
import os
import numpy as np
from plotting_configuration import *
from get_ECRH_config import libECRH_wrapper
from em_Albajar import em_abs_Alb, rotate_vec_around_axis
import scipy.constants as cnst
linear_p08 = np.zeros(8, dtype=np.int)
linear_p04 = np.zeros(8, dtype=np.int)
perp_a = np.zeros(8, dtype=np.int)
perp_beta = np.zeros(8, dtype=np.int)
# 99.981005999999994 # best we can do
linear_p08[4] = 826
linear_p04[4] = 165
perp_a[4] = 320
perp_beta[4] = 17

class ECRH_polarization:
    def __init__(self, folder, infile_name, outfile_name, gy_index, ref_shot):
        self.folder = folder
        self.infile_name = os.path.join(folder, infile_name)
        self.outfile_name = os.path.join(folder, outfile_name)
        self.gyindex = gy_index
        self.ECRH_wrapper = libECRH_wrapper(ref_shot)
        infile = open(self.infile_name)
        infile_lines = infile.readlines()
        infile.close()
        self.gy_freq = 140.e9
        self.mode = 1
        for i in range(len(infile_lines)):
            if("Ipmax" in infile_lines[i]):
                # Make sure IP is zero
                infile_lines[i] = "setenv AEPCHK_Ipmax 0.0e+06\n"
            if("PO8" in infile_lines[i]):
                p08_str = infile_lines[i].split("\"")[1]
                self.p08_init = np.fromstring(p08_str, sep=" ", dtype=np.int)
                self.p08_init[linear_p08 != 0] = linear_p08[linear_p08 != 0]
            if("PO4" in infile_lines[i]):
                p04_str = infile_lines[i].split("\"")[1]
                self.p04_init = np.fromstring(p04_str, sep=" ", dtype=np.int)
                self.p04_init[linear_p04 != 0] = linear_p04[linear_p04 != 0]
            if("POLPos" in infile_lines[i]):
                POLPos = infile_lines[i].split("\"")[1]
                self.a_init = np.fromstring(POLPos, sep=" ", dtype=np.int)
                self.a_init[perp_a != 0] = perp_a[perp_a != 0]
            if("TORPos" in infile_lines[i]):
                TORPos = infile_lines[i].split("\"")[1]
                self.beta_init = np.fromstring(TORPos, sep=" ", dtype=np.int)
                self.beta_init[perp_beta != 0] = perp_beta[perp_beta != 0]

    def find_best_polarization_for_no_Ip(self):
        N = 20
#        p_range = np.linspace(0, N - 1, N, dtype=np.int)
        p_range = np.linspace(-10 * (N - 1) / 2, 10 * (N - 1) / 2, N, dtype=np.int)
        p08_dir = +1
        p04_dir = +1
        pol_mat = np.zeros((N, N))
        self.p08 = np.copy(self.p08_init)
        self.p04 = np.copy(self.p04_init)
        self.a = self.a_init
        self.beta = self.beta_init
        self.Ip = 0.0
        self.Bt = -2.5
        for i in range(N):
            print("+++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++")
            print("Now working on i", i)
            print("+++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++")
            self.p08[self.gyindex] = self.p08_init[self.gyindex] + p_range[i] * p08_dir
            for j in range(N):
                self.p04[self.gyindex] = self.p04_init[self.gyindex] + p_range[j] * p04_dir
                output_dict = self.launch_pol_check()
                pol_mat[i, j] = output_dict["pol"]
        i, j = np.unravel_index(pol_mat.argmax(), pol_mat.shape)
        print("Best p08 and p04 value", self.p08_init[self.gyindex] + p_range[i], self.p04_init[self.gyindex] + p_range[j])
        print("Polarizer efficiency", pol_mat[i, j])
        plt.contourf(self.p08_init[self.gyindex] + p_range, self.p04_init[self.gyindex] + p_range, pol_mat.T)
        plt.plot(self.p08_init[self.gyindex] + p_range[i], self.p04_init[self.gyindex] + p_range[j], "+")
        plt.show()

    def launch_pol_check(self):
        infile = open(self.infile_name)
        infile_lines = infile.readlines()
        infile.close()
        infile = open(self.infile_name, "w")
        for k in range(len(infile_lines)):
            if("AEPCHK_Bt" in infile_lines[k]):
                infile_lines[k] = infile_lines[k].split("AEPCHK_Bt")[0] + "AEPCHK_Bt " + "{0: 1.2e}\n".format(self.Bt)
            elif("AEPCHK_Ipmax" in infile_lines[k]):
                infile_lines[k] = infile_lines[k].split("AEPCHK_Ipmax")[0] + "AEPCHK_Ipmax" + "{0: 1.1e}\n".format(self.Ip)
            elif("AEPCHK_SIMAT_POLPos" in infile_lines[k]):
                a_str = infile_lines[k].split("\"")[1]
                a_str = "\""
                for val in self.a:
                    a_str += " {0:d}".format(val)
                a_str += "\""
                infile_lines[k] = infile_lines[k].split("\"")[0] + \
                                          a_str + infile_lines[k].split("\"")[2]
            elif("AEPCHK_SIMAT_TORPos" in infile_lines[k]):
                beta_str = infile_lines[k].split("\"")[1]
                beta_str = "\""
                for val in self.beta:
                    beta_str += " {0:d}".format(val)
                beta_str += "\""
                infile_lines[k] = infile_lines[k].split("\"")[0] + \
                                          a_str + infile_lines[k].split("\"")[2]
                infile_lines[k] = infile_lines[k].split("\"")[0] + \
                                          beta_str + infile_lines[k].split("\"")[2]
            elif("PO8" in infile_lines[k]):
                p08_str = "\""
                for val in self.p08:
                    p08_str += " {0:d}".format(val)
                p08_str += "\""
                infile_lines[k] = infile_lines[k].split("\"")[0] + \
                                          p08_str + infile_lines[k].split("\"")[2]
            elif("PO4" in infile_lines[k]):
                p04_str = "\""
                for val in self.p04:
                    p04_str += " {0:d}".format(val)
                p04_str += "\""
                infile_lines[k] = infile_lines[k].split("\"")[0] + \
                                          p04_str + infile_lines[k].split("\"")[2]
            infile.write(infile_lines[k])
        infile.close()
        output_dict = {}
        print("Now calling", self.infile_name)
        ls_proc = subprocess.Popen([self.infile_name], stdout=subprocess.PIPE)
        ls_proc.wait()
        # check return code
        output_lines = ls_proc.stdout.readlines()
#                try:
#                    output_lines = check_output([self.infile_name]).splitlines()
#                except CalledProcessError:
#                    pass
#                outfile = open(self.outfile_name, "w")
#                outfile_lines = outfile.readlines()
#                outfile.close()
        for k in range(len(output_lines)):
            if("FAST (X) mode content" in output_lines[k]):
                output_dict["pol"] = float(output_lines[k].replace("FAST (X) mode content", "").replace("%", ""))
            elif("Magnetic field vector at intersection point" in output_lines[k]):
                output_dict["B_vec"] = np.fromstring(output_lines[k + 1], sep=" ")
            elif("intersection coordinates" in output_lines[k]):
                output_dict["x_vec"] = np.fromstring(output_lines[k + 1], sep=" ")
            elif("Cosine and sine of the angle between beam vector and magnetic field vector" in output_lines[k]):
                theta = np.fromstring(output_lines[k + 1], sep=" ")
                output_dict["cos_theta"] = theta[0]
                output_dict["sin_theta"] = theta[1]
            elif("using value [m-3] :" in output_lines[k]):
                output_dict["ne"] = float(output_lines[k].split(":")[1])
            elif("theta" in output_lines[k]):
                output_dict["theta_launch"] = float(output_lines[k].replace("theta", ""))
            elif("phi" in output_lines[k] and "->" not in output_lines[k]):
                output_dict["phi_launch"] = float(output_lines[k].replace("phi", "").split("+")[0])
            elif("S5 sys angle" in output_lines[k]):
                output_dict["phi_offset"] = float(output_lines[k].replace("= S5 sys angle", ""))
        print(output_dict)
        return output_dict

    def compare_pol_single(self, Bt, Ip, theta_launch, phi_launch):
        self.Bt = Bt
        self.Ip = Ip
        if(self.gyindex < 4):
            gynum = 100 + self.gyindex
        else:
            gynum = 200 + self.gyindex
        error, a, beta = self.ECRH_wrapper.tp2setval(gynum, theta_launch, phi_launch)
        a *= 10
        beta *= 10
        self.a = np.copy(self.a_init)
        self.beta = np.copy(self.beta_init)
        self.a[self.gyindex] = a
        self.beta[self.gyindex] = beta
        self.p08 = self.p08_init
        self.p04 = self.p04_init
        output_dict = self.launch_pol_check()
        abs_obj = em_abs_Alb()
        omega = self.gy_freq * np.pi * 2.0
        x_vec = output_dict["x_vec"]
        B_vec = output_dict["B_vec"]
#        phi_rot = np.deg2rad(output_dict["phi_offset"]) + 5.0 / 8.0 * np.pi  # Sector 5
#        x_vec = rotate_vec_around_axis(x_vec, z_axis, -phi_rot)
#        B_vec = rotate_vec_around_axis(B_vec, z_axis, -phi_rot)
        N_vec = np.zeros(3)
        N_vec[0] = -np.cos(np.deg2rad(output_dict["theta_launch"])) * np.cos(np.deg2rad(output_dict["phi_offset"]))
        N_vec[1] = np.cos(np.deg2rad(output_dict["theta_launch"])) * np.sin(np.deg2rad(output_dict["phi_offset"]))
        N_vec[2] = np.sin(np.deg2rad(output_dict["theta_launch"]))
#        phi = np.arctan2(x_vec[1], x_vec[0])
#        R_axis = np.zeros(3)
#        R_axis[0] = np.cos(phi)
#        R_axis[1] = np.sin(phi)
#        phi_axis = np.zeros(3)
#        phi_axis[0] = -np.sin(phi)
#        phi_axis[1] = np.cos(phi)
#        z_axis = np.zeros(3)
#        z_axis[2] = 1.0
#        N_vec = rotate_vec_around_axis(N_vec, phi_axis, )
#        N_vec = rotate_vec_around_axis(N_vec, z_axis, )
        print("theta_pol, phi_tor from input", theta_launch, phi_launch)
        print("theta_pol, phi_tor from script", output_dict["theta_launch"], output_dict["phi_launch"])
        output_dict["cos_theta_py"] = np.dot(N_vec / np.linalg.norm(N_vec), B_vec / np.linalg.norm(B_vec))
        print("cos_theta - py vs. idl:", output_dict["cos_theta_py"], output_dict["cos_theta"])
        X = output_dict["ne"] * cnst.e ** 2 / (cnst.epsilon_0 * cnst.m_e * omega ** 2)
        Y = cnst.e * np.linalg.norm(output_dict["B_vec"]) / (omega * cnst.m_e)
        output_dict["transmittance"] = abs_obj.get_filter_transmittance_correct_filter(omega, X, Y, self.mode, x_vec, -N_vec, B_vec, x_vec)  # _reverse
#        output_dict["transmittance"] = abs_obj.get_filter_transmittance(omega, X, Y, self.mode, x_vec, -N_vec, B_vec)
        print(output_dict["pol"], output_dict["transmittance"] * 100.0)
        output_dict["a"] = a
        output_dict["beta"] = beta
        return output_dict

    def compare_polarizers(self):
        a = []
        beta = []
        pol_ang = 0.0
        tor_ang = []
        cos_theta_py = []
        cos_theta_idl = []
        pol_py = []
        pol_idl = []
        fig1 = plt.figure()
        fig2 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)
        Bt = -1.8
        Ip = 800.e3
        # np.linspace(-10.0, 10.0, 31) [0.0]
        for tor_ang_init in np.linspace(-11.0, 11.0, 31):
            output_dict = self.compare_pol_single(Bt, Ip, pol_ang, tor_ang_init)
            tor_ang.append(output_dict["phi_launch"])
            beta.append(output_dict["beta"])
            cos_theta_py.append(output_dict["cos_theta_py"])
            cos_theta_idl.append(output_dict["cos_theta"])
            pol_idl.append(output_dict["pol"])
            pol_py.append(output_dict["transmittance"] * 100.0)
        ax1.plot(tor_ang, np.rad2deg(np.arccos(cos_theta_idl)), label=r"$\theta$ AUGEPCHK")
        ax1.plot(tor_ang, np.rad2deg(np.arccos(cos_theta_py)), "--", label=r"$\theta$ ECRad")
        ax1.set_xlabel(r"$\phi_\mathrm{tor}$ $[^\circ]$")
        ax1.set_ylabel(r"$\theta$ $[^\circ]$")
        ax1.legend()
        ax2.plot(np.rad2deg(np.arccos(cos_theta_idl)), np.array(pol_idl), label=r"$X_\mathrm{fraction}$ AUGEPCHK")  # rad2deg(np.arccos(cos_theta_idl))
        ax2.plot(np.rad2deg(np.arccos(cos_theta_py)), pol_py, "--", label=r"$X_\mathrm{fraction}$ ECRad")  # np.rad2deg(np.arccos(cos_theta_py))
        ax2.legend()
#        ax2.set_xlabel(r"$\phi_\mathrm{tor}$ $[^\circ]$")
        ax2.set_xlabel(r"$\theta$ $[^\circ]$")
        ax2.set_ylabel(r"$X_\mathrm{fraction}$ $[\%]$")
        plt.show()

if(__name__ == "__main__"):
    EC_pol = ECRH_polarization("/afs/ipp-garching.mpg.de/home/s/sdenk/ECRH_stuff", "prep_man_env_call", "test", 4, 33697)
#    EC_pol.find_best_polarization_for_no_Ip()
    EC_pol.compare_polarizers()

