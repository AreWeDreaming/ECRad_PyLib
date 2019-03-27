'''
Created on Jan 29, 2017

@author: sdenk
'''
import sys
import os
sys.path.append("/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib")
import dd
import numpy as np
libkk = np.ctypeslib.load_library("libkk8x", "/afs/ipp-garching.mpg.de/aug/ads/lib64/amd64_sles11/")
from EQU import EQU
from kk_mwillens import KK
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
import copy
import matplotlib.pyplot as plt

def eval_spline(x_vec, spl):
    return np.array([spl[0](x_vec[0], x_vec[1])])

def eval_spline_grad(x_vec, spl):
    return np.array([[spl[0].ev(x_vec[0], x_vec[1], dx=1, dy=0), spl[0].ev(x_vec[0], x_vec[1], dx=0, dy=1)]])

class special_points:
    def __init__(self, R_ax, z_ax, psi_ax, R_sep, z_sep, psi_sep):
        self.Raxis = R_ax
        self.zaxis = z_ax
        self.Rspx = R_sep
        self.zspx = z_sep
        self.psiaxis = psi_ax
        self.psispx = psi_sep

def adjust_external_Bt_vac(B_t, R, R_axis, bt_vac_correction):
    jvz = int(len(B_t[0]) / 2.0)
    Rv = R[-1]
    # jvz = np.argmin(np.abs(pfm_dict["zj"] - vz))
    Btf0 = B_t[-1, jvz]
    Btf0 = Btf0 * Rv / R_axis
    for i in range(len(B_t.T)):
        Btok = Btf0 * R_axis / R  # vacuum toroidal field from EQH
        Bdia = B_t.T[i] - Btok  # subtract vacuum toroidal field from equilibrium to obtain diamagnetic field
        B_t.T[i] = (Btok * bt_vac_correction) + Bdia  # add corrected vacuum toroidal field to be used
    return B_t

class EQDataSlice:
    def __init__(self, time, R, z, Psi, Br, Bt, Bz, special_points, rhop=None):
        self.time = time
        self.R = R
        self.z = z
        self.Psi = Psi
        self.rhop = rhop
        self.Br = Br
        self.Bt = Bt
        self.Bz = Bz
        self.R_ax = special_points.Raxis
        self.z_ax = special_points.zaxis
        self.R_sep = special_points.Rspx
        self.z_sep = special_points.zspx
        self.Psi_ax = special_points.psiaxis
        self.Psi_sep = special_points.psispx
        self.special = np.array([self.R_ax, self.Psi_sep])

    def transpose_matrices(self):
        self.Br = self.Br.T
        self.Bt = self.Bt.T
        self.Bz = self.Bz.T
        self.Psi = self.Psi.T
        if(self.rhop is not None):
            self.rhop = self.rhop.T

class EQData:
    def __init__(self, shot, external_folder='', EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005, Ext_data=False):
        self.shot = shot
        self.EQ_exp = EQ_exp
        self.EQ_diag = EQ_diag
        self.EQ_ed = EQ_ed
        self.bt_vac_correction = bt_vac_correction
        self.shotfile_ready = False
        self.loaded = False
        self.external_folder = external_folder
        self.Ext_data = Ext_data
        self.slices = []
        self.times = []

    def insert_slices_from_ext(self, times, slices, transpose=False):
        self.slices = copy.deepcopy(slices)
        self.times = copy.deepcopy(times)
        self.Ext_data = True
        self.loaded = True
        if(transpose):
            for eq_slice in self.slices:
                eq_slice.transpose_matrices()

    def read_EQ_from_Ext(self):
        t = np.loadtxt(os.path.join(self.external_folder, "t"))
        for it in range(len(t)):
            self.slices.append(self.read_EQ_from_Ext_single_slice(self, t[it], it))
            self.times.append(t[it])
        self.loaded = True

    def read_EQ_from_Ext_single_slice(self, time, index):
        R = np.loadtxt(os.path.join(self.external_folder, "R{0:d}".format(index)))
        z = np.loadtxt(os.path.join(self.external_folder, "z{0:d}".format(index)))
        # Rows are z, coloumns are R, like R, z in the poloidal crosssection
        Psi = np.loadtxt(os.path.join(self.external_folder, "Psi{0:d}".format(index)))
        B_r = np.loadtxt(os.path.join(self.external_folder, "Br{0:d}".format(index)))
        B_t = np.loadtxt(os.path.join(self.external_folder, "Bt{0:d}".format(index)))
        B_z = np.loadtxt(os.path.join(self.external_folder, "Bz{0:d}".format(index)))
        special = np.loadtxt(os.path.join(self.external_folder, "special_points{0:d}".format(index)))
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
        adjust_external_Bt_vac(B_t, R, opt.x[0], self.bt_vac_correction)
        rhop = np.sqrt((Psi - psi_ax) / (special[1] - psi_ax))
        print("WARNING!: R_sep and z_sep hard coded")
        special_pnts = special_points(opt.x[0], opt.x[1], psi_ax, 2.17, 0.0, special[1])
        return EQDataSlice(time, R, z, Psi.T, B_r.T, B_t.T, B_z.T, special_pnts, rhop.T)

    def init_read_from_shotfile(self):
        self.EQH = EQU()
        self.state = 0
        if(not self.EQH.Load(self.shot, Experiment=self.EQ_exp, Diagnostic=self.EQ_diag, Edition=self.EQ_ed)):
            self.state = -1
            return
        self.KKobj = KK()
        if(self.EQ_diag == "EQH"):
            self.GQH = dd.shotfile("GQH", int(self.shot))
        elif(self.EQ_diag == "IDE"):
            self.GQH = dd.shotfile("IDG", int(self.shot))
            self.IDF = dd.shotfile("IDF", int(self.shot))
        else:
            print("EQ diagnostic {0:s} not supported - only EQH and IDE are currently supported!".format(self.EQ_diag))
        self.MBI_shot = dd.shotfile('MBI', int(self.shot))
        self.shotfile_ready = True

    def GetSlice(self, time):
        if(not self.shotfile_ready):
            self.init_read_from_shotfile()
        R = self.EQH.getR(time)
        z = self.EQH.getz(time)
        Psi = self.EQH.getPsi(time)
        R0 = 1.65  # Point for which BTFABB is defined
        B_r = np.zeros((len(R), len(z)))
        B_t = np.zeros((len(R), len(z)))
        B_z = np.zeros((len(R), len(z)))
        # Adapted from mod_eqi.f90 by R. Fischer
        rv = 2.40
        vz = 0.e0
        magn_field_outside = self.KKobj.kkrzBrzt(self.shot, time, np.array([rv]), np.array([vz]), exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
        self.EQ_ed = magn_field_outside.ed
        Btf0_eq = magn_field_outside.bt[0]
        Btf0_eq = Btf0_eq * rv / R0
        special_points = self.KKobj.kkeqpfx(self.shot, time, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
        rhop = np.sqrt((Psi - special_points.psiaxis) / (special_points.psispx - special_points.psiaxis))
        if(self.EQ_diag == "IDE"):
            signal = self.IDF.getSignal("Btor", \
                              tBegin=time - 5.e-5, tEnd=time + 5.e-5)
            if(not np.isscalar(signal)):
                signal = np.mean(signal)
            Btok = signal * R0 / R
            Btf0 = signal
        else:
            try:
                signal = self.MBI_shot.getSignal("BTFABB", \
                              tBegin=time - 5.e-5, tEnd=time + 5.e-5)
                if(not np.isscalar(signal)):
                    signal = np.mean(signal)
                Btf0 = signal
                Btok = Btf0 * R0 / R
            except Exception as e:
                print(e)
                print("Could not find MBI data")
                Btok = Btf0_eq * R0 / R
                Btf0 = Btf0_eq
        R_temp = np.zeros(len(z))
        for i in range(len(R)):
            R_temp[:] = R[i]
            magn_field = self.KKobj.kkrzBrzt(self.shot, time, R_temp, z, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
            B_r[i] = magn_field.br
            B_t[i] = magn_field.bt
            B_z[i] = magn_field.bz
        # plt.plot(pfm_dict["Ri"],B_t[0], "^", label = "EQH B")
        # print("BTFABB correction",Btf0, Btf0_eq )
        # print("R,z",pfm_dict["Ri"][ivR],pfm_dict["zj"][jvz])
        # print("Time of B was: ", magn_field.time)
        # print("WARNING DIAMAGNETIC FIELD HAS OPPOSITE SIGN!!!")
        for j in range(len(z)):
            # plt.plot(pfm_dict["Ri"],B_t[j], label = "EQH B")
            Btok_eq = Btf0_eq * R0 / R  # vacuum toroidal field from EQH
            Bdia = B_t.T[j] - Btok_eq  # subtract vacuum toroidal field from equilibrium to obtain diamagnetic field
            B_t.T[j] = (Btok * self.bt_vac_correction) + Bdia  # add corrected vacuum toroidal field to be used
        print(Btf0)
        print("Original magnetic field: {0:2.3f}".format(Btf0))
        print("New magnetic field: {0:2.3f}".format(Btf0 * self.bt_vac_correction))
        return EQDataSlice(time, R, z, Psi, B_r, B_t, B_z, special_points, rhop)

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

    def get_surface_area(self, time, rho):
        self.eqm = map_equ.equ_map()
        self.eqm.Open(self.shot, exp=self.EQ_exp, diag=self.EQ_diag, ed=self.EQ_ed)
        return self.eqm.getQuantity(rho, "Area", t_in=time, coord_in='rho_pol')

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

if(__name__ == "__main__"):
    EQobj = EQData(33896, EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005, Ext_data=False)
    EQ_t = EQobj.GetSlice(3.0)
    plt.contour(EQ_t.R, EQ_t.z, (EQ_t.Psi.T - EQ_t.Psi_ax) / (EQ_t.Psi_sep - EQ_t.Psi_ax), levels=np.linspace(0.1, 1.2, 12))
    plt.contour(EQ_t.R, EQ_t.z, EQ_t.rhop.T, levels=np.linspace(0.1, 1.2, 12))
    plt.show()

