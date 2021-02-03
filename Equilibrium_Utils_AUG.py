'''
Created on Jan 29, 2017

@author: sdenk
'''
import sys
import os
sys.path.append('/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib')
import dd
from map_equ import equ_map
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from Basic_Methods.Equilibrium_Utils import EQDataExt, EQDataSlice, special_points
from scipy import __version__ as scivers
from scipy import constants as cnst

vessel_bd_file = "ASDEX_Upgrade_vessel.txt"
from datetime import datetime

def eval_R(x):
    return -x[0] ** 3

def eval_rhop(x, spl, rhop_target):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1]) - rhop_target) ** 2
    else:
        return (spl(x[0], x[1], grid=False) - rhop_target) ** 2

def check_Bt_vac_source(shot):
    try:
        MBI_shot = dd.shotfile('MBI', int(shot))
    except:
        print("No MBI shotfile. No Bt source!")
        return False, 1.0
    try:
        MBI_shot.getSignal("BTFABB")
        return True, 1.005
    except:
        return True, 1.01

def make_rhop_signed_axis(shot, time, R, rhop, f, f2=None, eq_exp='AUGD', eq_diag='EQH', eq_ed=0, external_folder=''):
    eq_obj = EQData(shot, external_folder=external_folder, eq_exp=eq_exp, eq_diag=eq_diag, eq_ed=eq_ed)
    R_ax = eq_obj.get_axis(time)[0]
    HFS = R < R_ax
    last_HFS = HFS[0]
    profile_cnt = 0
    rhop_out = []
    rhop_out.append([])
    f_out = []
    f_out.append([])
    if(f2 is not None):
        f2_out = []
        f2_out.append([])
    for i in range(len(HFS)):
        if(last_HFS != HFS[i]):
            last_HFS = HFS[i]
            rhop_out.append([])
            f_out.append([])
            if(f2 is not None):
                f2_out.append([])
            profile_cnt += 1
        if(HFS[i]):
            rhop_out[profile_cnt].append(-rhop[i])
        else:
            rhop_out[profile_cnt].append(rhop[i])
        f_out[profile_cnt].append(f[i])
        if(f2 is not None):
            f2_out[profile_cnt].append(f2[i])
    for profile_cnt in range(len(rhop_out)):
        rhop_out[profile_cnt] = np.array(rhop_out[profile_cnt])
        f_out[profile_cnt] = np.array(f_out[profile_cnt])
        if(f2 is not None):
            f2_out[profile_cnt] = np.array(f2_out[profile_cnt])
    if(f2 is not None):
        return rhop_out, f_out, f2_out, R_ax
    else:
        return rhop_out, f_out, R_ax


class EQData(EQDataExt):
    def __init__(self, shot, external_folder='', EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, Ext_data=False):
        EQDataExt.__init__(self, shot, external_folder, EQ_exp, EQ_diag, EQ_ed, Ext_data)        

    def init_read_from_shotfile(self):
        self.equ = equ_map()
        self.state = 0
        if(not self.equ.Open(self.shot, diag=self.EQ_diag, exp=self.EQ_exp, ed=self.EQ_ed)):
            print("Failed to open shotfile")
            self.state = -1
            return
        self.EQ_ed = self.equ.ed
        if(self.EQ_diag == "EQH"):
            self.GQH = dd.shotfile("GQH", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
            self.FPC = dd.shotfile("FPC", int(self.shot))
        elif(self.EQ_diag == "IDE"):
            self.GQH = dd.shotfile("IDG", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
            self.IDF = dd.shotfile("IDF", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
            self.FPC = None
        else:
            print("EQ diagnostic {0:s} not supported - only EQH and IDE are currently supported!".format(self.EQ_diag))
        self.MBI_shot = dd.shotfile('MBI', int(self.shot))
        self.equ.read_scalars()
        self.shotfile_ready = True

    def ApplyBVacCorrectionToSlice(self, EQSlice):
        self.R0 = 1.65  # Point for which BTFABB is defined
        # Adapted from mod_eqi.f90 by R. Fischer
        rv = 2.40
        vz = 0.e0
        Bt_out = self.equ.rz2brzt(np.array([rv]), np.array([vz]), \
                                  EQSlice.time)[2]
        Bt_out = np.asscalar(Bt_out)
        Btf0_eq = Bt_out
        Btf0_eq = Btf0_eq * rv / self.R0
        try:
            signal = self.MBI_shot.getSignal("BTFABB", \
                          tBegin=EQSlice.time - 5.e-5, tEnd=EQSlice.time + 5.e-5)
            if(not np.isscalar(signal)):
                signal = np.mean(signal)
            Btf0 = signal
            Btok = Btf0 *  self.R0 / EQSlice.R
        except Exception as e:
            print(e)
            print("Could not find MBI data")
            Btok = Btf0_eq *  self.R0 / EQSlice.R
            Btf0 = Btf0_eq
            for j in range(len(EQSlice.z)):
                # plt.plot(pfm_dict["Ri"],B_t[j], label = "EQH B")
                Btok_eq = Btf0_eq * self.R0 / EQSlice.R  # vacuum toroidal field from EQH
                Bdia = EQSlice.B_t.T[j] - Btok_eq  # subtract vacuum toroidal field from equilibrium to obtain diamagnetic field
                EQSlice.B_t.T[j] = Btok + Bdia  # add corrected vacuum toroidal field to be used
    # #         print(Btf0)
    # #         print("Original magnetic field: {0:2.3f}".format(Btf0))
    # #         print("New magnetic field: {0:2.3f}".format(Btf0 * self.bt_vac_correction))
        return EQSlice

    def GetSlice(self, time, bt_vac_correction = 1.0, B_vac_correction=True):
        if(not self.shotfile_ready):
            self.init_read_from_shotfile()
        R = self.equ.Rmesh
        z = self.equ.Zmesh
        dummy, time_index = self.equ._get_nearest_index(time)
        time_index = time_index[0]
        special = special_points(self.equ.ssq['Rmag'][time_index], self.equ.ssq['Zmag'][time_index], self.equ.psi0[time_index], self.equ.ssq['Raus'][time_index], self.equ.ssq['Zsquad'][time_index], self.equ.psix[time_index])
        self.equ.read_pfm()
        Psi = self.equ.pfm[:, :, time_index]
        rhop = np.sqrt((Psi - special.psiaxis) / (special.psispx - special.psiaxis))
        B_r, B_z, B_t = self.equ.Bmesh(time) 
        if(B_vac_correction):
            EQ_slice = self.ApplyBVacCorrectionToSlice( EQDataSlice(time, R, z, \
                                                                Psi, B_r, B_t, \
                                                                B_z, special=special, \
                                                                rhop=rhop ))
        else:
            EQ_slice = EQDataSlice(time, R, z, Psi, B_r, B_t, B_z, special=special, rhop=rhop )
        if(bt_vac_correction != 1.0):
            if(EQ_slice.R_ax is None):
                R_ax, z_ax = self.get_axis(time)
            else:
                R_ax = EQ_slice.R_ax
            EQ_slice.Bt = self.adjust_external_Bt_vac(EQ_slice.B_t, EQ_slice.R, R_ax, bt_vac_correction)
        self.eq_shape = (len(self.slices[0].R),len(self.slices[0].z))
        return EQ_slice

    def map_Rz_to_rhot(self, time, R, z):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            return self.equ.rz2rho(R, z, t_in=time, coord_out="rho_tor")


    def rhop_to_rhot(self, time, rhop):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            rhot = self.equ.rho2rho(rhop, t_in=time, coord_out="rho_tor")
            try:
                return rhot 
            except TypeError:
                return rhot[0] # equ routines always return arrays
            
    def rhot_to_rhop(self, time, rhot):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            rhop = self.equ.rho2rho(rhot, t_in=time, coord_in='rho_tor', coord_out="rho_pol")
            try:
                return rhop 
            except TypeError:
                return rhop[0] # equ routines always return arrays

    def rhop_to_Psi(self, time, rhop):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            Psi = self.equ.rho2rho(rhop, t_in=time, coord_out="Psi")
            try:
                return Psi 
            except TypeError:
                return Psi[0] # equ routines always return arrays
        
        
    def getQuantity(self, rhop, quant_name, time):
        dummy, time_index = self.equ._get_nearest_index(time)
        time_index=time_index[0]
        pfl = self.equ.get_profile("PFL")[time_index]
        psi_in = self.rhop_to_Psi(time, rhop)
        if(quant_name not in ['Vol', 'Area', 'Pres', 'Jpol', 'dVol', 'dArea', 'dPres', 'dJpol']):
            quant= self.equ.get_profile(quant_name)
        elif(quant_name in ['Vol', 'Area', 'Pres', 'Jpol']):
            quant, dummy = self.equ.get_mixed(quant_name)
        elif(quant_name in ['dVol', 'dArea', 'dPres', 'dJpol']):
            dummy, quant = self.equ.get_mixed(quant_name[1:])
        else:
            print(quant_name + " not supported in get Quantity") 
        quant = quant[time_index]
        if(pfl[0] > pfl[-1]):
            pfl = pfl[::-1]
            quant = quant[::-1]
        quantspl = InterpolatedUnivariateSpline(pfl, quant)
        return quantspl(psi_in)

    def add_ripple_to_slice(self, time, EQSlice):
        R = self.get_axis(time)[0]
        B_axis = self.get_B_on_axis(time)
        B_sign = np.sign(np.mean(EQSlice.Bt))
        EQSlice.inject_ripple(aug_bt_ripple(R, B_axis * B_sign))
        return EQSlice
    
    def make_E_Fit_EQDSK(self, working_dir, shot, time, I_p):
        EQDSK_file = open(os.path.join(working_dir, "g{0:d}_{1:05d}".format(self.shot, int(time*1.e3))), "w")
        dummy = 0.0
        EQ_t = self.GetSlice(time)
        today = datetime.date.today()
        # First line
        EQDSK_file.write("{0:50s}".format("EFIT " + str(today.month) + "/" + str(today.day) + "/" + str(today.year) + " #" + str(shot) + \
                                               " " + str(int(time*1.e3)) + "ms"))
        # Second line
        EQDSK_file.write("{0: 2d}{1: 4d}{2: 4d}\n".format(self.EQ_ed, len(EQ_t.R), len(EQ_t.z)))
        # Third line
        EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(EQ_t.R[-1] - EQ_t.R[0], EQ_t.z[-1] - EQ_t.z[0], \
                                                                                      1.65, EQ_t.R[0], np.mean(EQ_t.z)))
        B_axis = np.double(self.get_B_on_axis(time)) * np.sign(np.mean(EQ_t.Bt.flatten()))
        # Fourth line
        EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(EQ_t.R_ax, EQ_t.z_ax, \
                                                                                      EQ_t.Psi_ax, EQ_t.Psi_sep, np.double(B_axis)))
        
        # Fifth line
        EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(I_p, EQ_t.Psi_ax, 0.0, EQ_t.R_ax, 0.0))
        # Sixth line
        EQDSK_file.write("{0: 1.9e}{1: 1.9e}{2: 1.9e}{3: 1.9e}{4: 1.9e}\n".format(EQ_t.z_ax, 0.0, EQ_t.Psi_sep, 0.0, 0.0))
        N = len(EQ_t.R)
        Psi = np.linspace(EQ_t.Psi_ax, EQ_t.Psi_sep, N)
        rhop = np.sqrt((Psi - EQ_t.Psi_ax)/(EQ_t.Psi_sep - EQ_t.Psi_ax))
        quant_dict = {}
        quant_dict["q"] = self.getQuantity(rhop, "Qpsi", time)
        quant_dict["pres"] = self.getQuantity(rhop, "Pres", time)
        quant_dict["pprime"] = self.getQuantity(rhop, "dPres", time)
        quant_dict["ffprime"] = self.getQuantity(rhop, "FFP", time)
        ffp_spl = InterpolatedUnivariateSpline(rhop, quant_dict["ffprime"] )
        f_sq_spl = ffp_spl.antiderivative(1)
        f_spl = InterpolatedUnivariateSpline(rhop, np.sign(B_axis) * \
                                             (np.sqrt(2.0 * f_sq_spl(rhop) +  \
                                                      (EQ_t.R_ax * B_axis)**2)))
        # Get the correct sign back since it is not included in ffprime
        quant_dict["fdia"] = f_spl(rhop)
        N_max = 5
        format_str = " {0: 1.9e}"
        for key in ["fdia", "pres", "ffprime", "pprime"]:
            i = 0
            while i < N:
                EQDSK_file.write(format_str.format(quant_dict[key][i]))
                i += 1
                if(i %  N_max == 0):
                    EQDSK_file.write("\n")
        if(i % N_max != 0):
            EQDSK_file.write("\n")
        N_new = 0
        for i in range(len(EQ_t.R)):
            for j in range(len(EQ_t.z)):
                EQDSK_file.write(format_str.format(EQ_t.Psi[i,j]))
                N_new += 1
                if(N_new == N_max):
                    EQDSK_file.write("\n")
                    N_new = 0
        if(N_new != 0):
            EQDSK_file.write("\n")
        for key in ["q"]:
            i = 0
            while i < N:
                EQDSK_file.write(format_str.format(quant_dict[key][i]))
                i += 1
                if(i %  N_max == 0):
                    EQDSK_file.write("\n")
        EQDSK_file.flush()
        EQDSK_file.close()

class aug_bt_ripple:
    def __init__(self, R0, Btf0):
        # Btf0 vacuum toroidal magnetic field at R0
        if(np.abs(Btf0) > 20.e0 or np.abs(Btf0) < 20.e-3 or R0 <= 0.e0):
            print("Nonsensical Btf0 or R0 in init_ripple in mod_ripple3d.f90")
            print("R0 [m], Btf0 [T]", R0, Btf0)
            raise ValueError("Input error in init_ripple in mod_ripple3d.f90")
        self.R0 = R0
        self.Btf0 = Btf0
        self.A0 = -21.959; self.A1 = 4.653; self.A2 = 1.747
        self.B0 = 7.891; self.B1 = -1.070; self.B2 = -0.860
        self.C0 = -23.315; self.C1 = 5.024; self.C2 = 0.699
        self.D0 = 8.196; self.D1 = -1.362; self.D2 = -0.367
        self.E0 = -19.539; self.E1 = 3.905;
        self.K0 = 6.199; self.K1 = -0.767

    def get_ripple(self, R_vec):
        Btf0 = self.Btf0 / R_vec[0] * self.R0
        # 16 coils at ASDEX Upgrade, hence we have also 16 periods in the ripple function when going once around the torus
        psi = R_vec[1] * 16.e0
        B_ripple = np.zeros(3)
        B_ripple[0] = Btf0 * np.exp(self.C0 + self.C1 * R_vec[2] ** 2 + self.C2 * R_vec[2] ** 4 + \
          R_vec[0] * (self.D0 + self.D1 * R_vec[2] ** 2 + self.D2 * R_vec[2] ** 4)) * np.sin(psi)
        B_ripple[1] = Btf0 * np.exp(self.A0 + self.A1 * R_vec[2] ** 2 + self.A2 * R_vec[2] ** 4 + \
                        R_vec[0] * (self.B0 + self.B1 * R_vec[2] ** 2 + self.B2 * R_vec[2] ** 4)) * np.cos(psi)
        B_ripple[2] = Btf0 * np.exp(self.E0 + self.E1 * R_vec[2] ** 2 + R_vec[0] * (self.K0 + \
                      self.K1 * R_vec[2] ** 2)) * R_vec[2] * np.sin(psi)

        return B_ripple
    
def compare_f_dia(shot, time, EQ_exp, EQ_diag, EQ_ed):
    from Plotting_Configuration import plt
    EQObj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    EQ_t = EQObj.GetSlice(time)
    rho = np.linspace(0.0, 1.0, 100)
    ffp = EQObj.getQuantity(rho, "FFP", time)
    ffp_spl = InterpolatedUnivariateSpline(rho, ffp)
    f_sq_spl = ffp_spl.antiderivative(1)
    magn_field_axis = EQObj.MBI_shot.getSignal("BTFABB", \
                                              tBegin=time - 5.e-5, tEnd=time + 5.e-5)
    f_spl = InterpolatedUnivariateSpline(rho, np.sign(magn_field_axis) * \
                                                               (np.sqrt(2.0 * f_sq_spl(rho) + \
                                                                        (EQ_t.R_ax * magn_field_axis)**2)))
    psi_prof = EQObj.rhop_to_Psi(time, rho)
    plt.plot(psi_prof, f_spl(rho))
    gpol = EQObj.getQuantity(rho, "Jpol", time) * cnst.mu_0 / 2.0 / np.pi
    plt.plot(psi_prof, gpol, "--")
    plt.show()
        

if(__name__ == "__main__"):
#     compare_f_dia(35662, 3.84, EQ_exp="AUGD", EQ_diag="IDE", EQ_ed=2)
#     from plotting_configuration import *
    EQ_obj = EQData(35662, EQ_diag="IDE")
    EQSlice = EQ_obj.GetSlice(3.84)
    np.savetxt(os.path.join(os.path.expanduser("~"), "Documentation", "Data", "35662_R.dat"), EQSlice.R)
    np.savetxt(os.path.join(os.path.expanduser("~"), "Documentation", "Data", "35662_z.dat"), EQSlice.z)
    np.savetxt(os.path.join(os.path.expanduser("~"), "Documentation", "Data", "35662_Br.dat"), EQSlice.Br)
    np.savetxt(os.path.join(os.path.expanduser("~"), "Documentation", "Data", "35662_Bt.dat"), EQSlice.Bt)
    np.savetxt(os.path.join(os.path.expanduser("~"), "Documentation", "Data", "35662_Bz.dat"), EQSlice.Bz)
#     time = 4.80
#     rhop = np.linspace(0.025, 0.99, 10)
#     EQSlice = EQ_obj.GetSlice(time)
#     plt.contour(EQSlice.R, EQSlice.z, EQSlice.rhop.T, levels=rhop)
# #    print("R_aus", "z_aus", EQ_obj.get_R_aus(time, rhop))
# #    EQ_obj = EQData(33697, EQ_diag="IDE")
# #    R_av = EQ_obj.get_mean_r(time, [0.08])
# #    plt.figure()
# #    plt.plot(rhop, R_av)
#     plt.show()
