'''
Created on 29.04.2019

@author: sdenk
'''
import sys
import ctypes as ct
import os
sys.path.append("/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib")
import dd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from equilibrium_utils import EQDataExt, EQDataSlice, eval_spline, special_points
from Geometry_utils import get_contour, get_Surface_area_of_torus, get_arclength, get_av_radius
from scipy import __version__ as scivers
import scipy.optimize as scopt
from kk_local import KK
vessel_bd_file = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECFM_Pylib/ASDEX_Upgrade_vessel.txt"
from scipy.signal import medfilt
from map_equ import equ_map
def eval_R(x):
    return -x[0] ** 3

def eval_rhop(x, spl, rhop_target):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1]) - rhop_target) ** 2
    else:
        return (spl(x[0], x[1], grid=False) - rhop_target) ** 2

def make_rhop_signed_axis(shot, time, R, rhop, f, f2=None, eq_exp='AUGD', eq_diag='EQH', eq_ed=0, external_folder=''):
    eq_obj = EQData(shot, external_folder=external_folder, eq_exp=eq_exp, eq_diag=eq_diag, eq_ed=eq_ed)
    R_ax, z_ax = eq_obj.get_axis(time)
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
    def __init__(self, shot, external_folder='', EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005, Ext_data=False):
        EQDataExt.__init__(self, shot, external_folder, EQ_exp, EQ_diag, EQ_ed, bt_vac_correction, Ext_data)

    def init_read_from_shotfile(self):
        self.equ = equ_map()
        self.R0 = 1.65  # Point for which BTFABB is defined
        if(not self.equ.Open(self.shot, diag=self.EQ_diag, exp=self.EQ_exp, ed=self.EQ_ed)):
            self.state = -1
            return
        self.KKobj = KK()
        self.KKobj.kkeqints(10)# No interpolation!
        self.state = 0
#         if(not self.EQH.Load(self.shot, Experiment=self.EQ_exp, Diagnostic=self.EQ_diag, Edition=self.EQ_ed)):
#             self.state = -1
#             return
        if(self.EQ_diag == "EQH"):
            self.GQH = dd.shotfile("GQH", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
        elif(self.EQ_diag == "IDE"):
            self.GQH = dd.shotfile("IDG", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
        else:
            print("EQ diagnostic {0:s} not supported - only EQH and IDE are currently supported!".format(self.EQ_diag))
        self.MBI = dd.shotfile('MBI', int(self.shot))
        self.shotfile_ready = True

    def GetSlice(self, time, B_vac_correction=True):
        if(not self.shotfile_ready):
            self.init_read_from_shotfile()
        output = self.KKobj.kkeqpfm(self.shot, time, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed,  m=65, n=129)
        R = output.R
        z = output.z
        Psi = output.Psi
        
        B_r = np.zeros((len(R), len(z)))
        B_t = np.zeros((len(R), len(z)))
        B_z = np.zeros((len(R), len(z)))
        rhop = np.zeros((len(R), len(z)))
        # Adapted from mod_eqi.f90 by R. Fischer
        rv = 2.40
        vz = 0.e0
        magn_field_outside = self.KKobj.kkrzBrzt(self.shot, time, np.array([rv]), np.array([vz]), exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
        self.EQ_ed = magn_field_outside.ed
        Btf0_eq = magn_field_outside.bt[0]
        Btf0_eq = Btf0_eq * rv / self.R0
        special_points = self.KKobj.kkeqpfx(self.shot, time, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
        if(special_points.Rspx == 0.0):
            print("Limiter plasma detected overwriting psi_spx, R_spx and z_spx with psi_lim, R_lim and z_lim")
            special_points.psispx = special_points.psilim
            special_points.Rspx = special_points.Rlim
            special_points.zspx = special_points.zlim
        try:
            MBI_signal = self.MBI_shot.getSignal("BTFABB")
            MBI_time = self.MBI_shot.getTimeBase("BTFABB")
            itime_MBI = np.argmin(np.abs(time - MBI_time))
            if(itime_MBI >= 99 and itime_MBI <= len(MBI_signal) - 99):
                Btf0 = np.median(MBI_signal[itime_MBI-99: itime_MBI+99]) # analogous to IDA
            else:
                Btf0 = MBI_signal[itime_MBI]
            Btok = Btf0 * self.R0 / R
            print("Btf0 vs. Btf0_eq", Btf0, Btf0_eq)
        except Exception as e:
            print(e)
            print("Could not find MBI data")
            Btok = Btf0_eq * self.R0 / R
            Btf0 = Btf0_eq
        R_temp = np.zeros(len(z))
        for i in range(len(R)):
            R_temp[:] = R[i]
            rhop_out = self.KKobj.kkrzptfn(self.shot, time, R_temp, z, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            rhop[i] = rhop_out.rho_p
            magn_field = self.KKobj.kkrzBrzt(self.shot, time, R_temp, z, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            B_r[i] = magn_field.br
            B_t[i] = magn_field.bt
            B_z[i] = magn_field.bz
        # plt.plot(pfm_dict["Ri"],B_t[0], "^", label = "EQH B")
        # print("BTFABB correction",Btf0, Btf0_eq )
        # print("R,z",pfm_dict["Ri"][ivR],pfm_dict["zj"][jvz])
        # print("Time of B was: ", magn_field.time)
        # print("WARNING DIAMAGNETIC FIELD HAS OPPOSITE SIGN!!!")
        if(B_vac_correction):
            for j in range(len(z)):
                # plt.plot(pfm_dict["Ri"],B_t[j], label = "EQH B")
                Btok_eq = Btf0_eq * self.R0 / R  # vacuum toroidal field from EQH
                Bdia = B_t.T[j] - Btok_eq  # subtract vacuum toroidal field from equilibrium to obtain diamagnetic field
                B_t.T[j] = (Btok * self.bt_vac_correction) + Bdia  # add corrected vacuum toroidal field to be used
#         print(Btf0)
#         print("Original magnetic field: {0:2.3f}".format(Btf0))
#         print("New magnetic field: {0:2.3f}".format(Btf0 * self.bt_vac_correction))
        return EQDataSlice(time, R, z, Psi, B_r, B_t, B_z, special=special_points, rhop=rhop)

    def get_axis(self, time):
        if(self.external_folder != '' or self.Ext_data):
            if(self.loaded == False):
                self.read_EQ_from_Ext()
            index = np.argmin(self.times - time)
            R = self.slices[index].R
            z = self.slices[index].z
            Psi = self.slices[index].Psi
            special = self.slices[index].special
            if(Psi[len(R) / 2][len(z) / 2] > special[1]):
            # We want a minimum in the flux at the magn. axis
                Psi *= -1.0
                special[1] *= -1.0
            psi_spl = RectBivariateSpline(R, z, Psi)
            indicies = np.unravel_index(np.argmin(Psi), Psi.shape)
            R_init = np.array([R[indicies[0]], z[indicies[1]]])
            print(R_init)
            opt = minimize(eval_spline, R_init, args=[psi_spl], \
                     bounds=[[np.min(R), np.max(R)], [np.min(z), np.max(z)]])
            print("Magnetic axis position: ", opt.x[0], opt.x[1])
            return opt.x[0], opt.x[1]
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            if(not self.shotfile_ready):
                return
            special_points = self.KKobj.kkeqpfx(self.shot, time, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            return special_points.Raxis, special_points.zaxis

    def get_B_on_axis(self, time):
        R_ax, z_ax = self.get_axis(time)
        if(self.external_folder != '' or self.Ext_data):
            index = np.argmin(self.times - time)
            R = self.slices[index].R
            z = self.slices[index].z
            B_tot = np.sqrt(self.slices[index].Br ** 2 + self.slices[index].Br ** 2 + self.slices[index].Bz ** 2)
            B_tot_spl = RectBivariateSpline(R, z, B_tot)
            return B_tot_spl(R_ax, z_ax)
        else:
            magn_field = self.KKobj.kkrzBrzt(self.shot, time, np.array([R_ax]), np.array([z_ax]), exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            return np.sqrt(magn_field.br[0] ** 2 + magn_field.bt[0] ** 2 + magn_field.bz[0] ** 2)

    def map_Rz_to_rhop(self, time, R, z):
        if(self.external_folder != '' or self.Ext_data):
            if(self.loaded == False):
                self.read_EQ_from_Ext()
            index = np.argmin(self.times - time)
            R = self.slices[index].R
            z = self.slices[index].z
            Psi = self.slices[index].Psi
            special = self.slices[index].special
            if(Psi[len(R) / 2][len(z) / 2] > special[1]):
            # We want a minimum in the flux at the magn. axis
                Psi *= -1.0
                special[1] *= -1.0
            psi_spl = RectBivariateSpline(R, z, Psi)
            indicies = np.unravel_index(np.argmin(Psi), Psi.shape)
            R_init = np.array([R[indicies[0]], z[indicies[1]]])
            print(R_init)
            opt = minimize(eval_spline, R_init, args=[psi_spl], \
                     bounds=[[np.min(R), np.max(R)], [np.min(z), np.max(z)]])
            print("Magnetic axis position: ", opt.x[0], opt.x[1])
            psi_ax = psi_spl(opt.x[0], opt.x[1])
            rhop = np.sqrt((Psi - psi_ax) / (special[1] - psi_ax))
            rhop_spl = RectBivariateSpline(R, z, rhop)
            return rhop_spl(R, z, grid=False)
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            output = self.KKobj.kkrzptfn(self.shot, time, R, z, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            return output.rho_p

    def map_Rz_to_rhot(self, time, R, z):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            output = self.KKobj.kkrzptfn(self.shot, time, R, z, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            return output.rho_t

    def rhop_to_rhot(self, time, rhop):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            output = self.KKobj.kkrhopto(self.shot, time, rhop, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            return output.rho_t

    def rhop_to_Psi(self, time, rhop):
        if(self.external_folder != '' or self.Ext_data):
            print("Not yet implemented")
            raise ValueError("Not yet implemented")
        else:
            if(not self.shotfile_ready):
                self.init_read_from_shotfile()
            output = self.KKobj.kkrhopto(self.shot, time, rhop, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            return output.pf

    def get_Rz_contour(self, time, rhop_in, only_single_closed=False):
        if(np.isscalar(rhop_in)):
            rhops = [rhop_in]
        else:
            rhops = rhop_in
        EQSlice = self.read_EQ_from_shotfile(time)
#        special_points = self.KKobj.kkeqpfx(self.shot, time, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
#        z_points = np.array([special_points.zspx, special_points.zspx_2])
#        point_sort = np.argsort(z_points)
#        z_lower = z_points[point_sort][0] - 0.001
#        z_upper = z_points[point_sort][1] + 0.001
#        rhop_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, EQSlice.rhop)
#        R_points = np.linspace(np.min(EQSlice.R), np.max(EQSlice.R), 151)
#        z_points = np.linspace(z_lower, z_upper, 151)
#        rhop = rhop_spl(R_points, z_points, grid=True)
        R_conts = []
        z_conts = []
        for rhop_cur in rhops:
            closed_info, conts = get_contour(EQSlice.R, EQSlice.z, EQSlice.rhop, rhop_cur)
            if(only_single_closed):
                for closed, cont in zip(closed_info, conts):
                    # Return first closed -> should be the nested flux surface
                    if(closed):
                        R_conts.append(cont.T[0])
                        z_conts.append(cont.T[1])
                        break;
            else:
                R_conts.append(np.concatenate(cont.T[0]))
                z_conts.append(np.concatenate(cont.T[1]))
        if(np.isscalar(rhop_in)):
            return R_conts[-1], z_conts[-1]
        else:
            return R_conts, z_conts


    def add_ripple_to_slice(self, time, EQSlice):
        R, z = self.get_axis(time)
        B_axis = self.get_B_on_axis(time)
        B_sign = np.sign(np.mean(EQSlice.Bt))
        EQSlice.inject_ripple(aug_bt_ripple(R, B_axis * B_sign))
        return EQSlice

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

if(__name__ == "__main__"):
    EQ_obj = EQData(33697)
    time = 4.80
#    rhop = np.linspace(0.025, 0.99, 10)
#    EQSlice = EQ_obj.read_EQ_from_shotfile(time)
#    plt.contour(EQSlice.R, EQSlice.z, EQSlice.rhop.T, levels=[rhop])
#    print("R_aus", "z_aus", EQ_obj.get_R_aus(time, rhop))
    EQ_obj = EQData(33697, EQ_diag="IDE")
    R_av = EQ_obj.get_mean_r(time, [0.08])
#    plt.figure()
#    plt.plot(rhop, R_av)
#    plt.show()
