'''
Created on Jan 29, 2017

@author: sdenk
'''
from GlobalSettings import AUG, TCV, itm, SLES12
if(not AUG or TCV):
    raise(ValueError('Using TCV equilibrium module even though AUG False and/or TCV True'))
import sys
import ctypes as ct
import os
if(not itm):
    sys.path.append("/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib")
import dd
import numpy as np
if(SLES12):
    libkk = np.ctypeslib.load_library("libkk8x", "/afs/ipp-garching.mpg.de/aug/ads/lib64/amd64_sles12/")
else:
    libkk = np.ctypeslib.load_library("libkk8x", "/afs/ipp-garching.mpg.de/aug/ads/lib64/amd64_sles11/")
from EQU import EQU
from kk_mwillens import KK
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from equilibrium_utils import EQDataExt, EQDataSlice, eval_spline, special_points
from Geometry_utils import get_contour, get_Surface_area_of_torus, get_arclength, get_av_radius
from scipy import __version__ as scivers
import scipy.optimize as scopt
vessel_bd_file = "/afs/ipp-garching.mpg.de/home/s/sdenk/F90/ECRad_Pylib/ASDEX_Upgrade_vessel.txt"
try:
    from kk_abock import kk as KKeqi
except ImportError as e:
    print(e)
import matplotlib.pyplot as plt

def eval_R(x):
    return -x[0] ** 3

def eval_rhop(x, spl, rhop_target):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1]) - rhop_target) ** 2
    else:
        return (spl(x[0], x[1], grid=False) - rhop_target) ** 2


def get_current(nshot, tshot, npts):

# Input
    nr = npts
    ed = 0
    exp = "AUGD"
    diag = "EQH"
    c_nr = ct.c_long(nr - 1)
    _nr = ct.byref(c_nr)
    c_nr_sp = ct.c_long(4)
    _nr_sp = ct.byref(c_nr_sp)
    c_ed = ct.c_long(ed)
    _ed = ct.byref(c_ed)
    c_err = ct.c_long(0)
    _err = ct.byref(c_err)
    c_shot = ct.c_long(nshot)
    _shot = ct.byref(c_shot)
    c_exp = ct.c_char_p(exp)
    c_dia = ct.c_char_p(diag)
    c_tshot = ct.c_float(tshot)
    _tshot = ct.byref(c_tshot)
    # c_typ=ct.c_long(typ)
    # _typ=ct.byref(c_typ)
    c_rhop = (ct.c_float * nr)()
    _rhop = ct.byref(c_rhop)
    c_pfx = (ct.c_float * 5)()
    _pfx = ct.byref(c_pfx)
    c_pfr = (ct.c_float * 5)()
    _pfr = ct.byref(c_pfr)
    c_pfz = (ct.c_float * 5)()
    _pfz = ct.byref(c_pfz)
    c_jdotb = (ct.c_float * nr)()
    _jdotb = ct.byref(c_jdotb)
    c_jpar = (ct.c_float * nr)()
    _jpar = ct.byref(c_jpar)
    libkk.kkeqpfx(_err, c_exp, c_dia, c_shot, _ed, _tshot, \
               _nr_sp, _pfx, _pfr, _pfz)
    psi_axis = c_pfx[0]
    psi_sep = c_pfx[1]
    libkk.kkeqjpar(_err, c_exp, c_dia, c_shot, _ed, _tshot, \
               11, _nr, _rhop, _jdotb, _jpar)
# Numpy output`
    time = c_tshot.value
    rhop1 = np.zeros(nr)
    R = np.zeros(nr)
    rhop = np.zeros(nr)
    jpar = np.zeros(nr)
    for jr in range (nr):
        rhop[jr] = c_rhop[jr]
        jpar[jr] = c_jpar[jr]
    jr = 0
    while jr < nr:
        if(rhop[jr] == 0.0):
            break
        jr += 1
    rhop = rhop[0:jr]
    jpar = jpar[0:jr]
    # R =  R[0:jr]

    # max_rhop = max(rhop2)
    # min_rhop = min(rhop2)
    rhop = np.sqrt((rhop - psi_axis) / (psi_sep - psi_axis))
    rhop_sorted = np.zeros(len(rhop))
    jpar_sorted = np.zeros(len(rhop))
    rhop_sorted[-1] = rhop[0]
    jpar_sorted[-1] = jpar[0]
    for i in range(0, len(rhop) - 1):
        rhop_sorted[i] = rhop[-i - 1]
        jpar_sorted[i] = jpar[-i - 1]
    # plt.plot(rhop, jpar)
    # print rhop_sorted
    # output = kk.KK().kkeqpsp(nshot,tshot,1.0,"AUGD","EQH",0)
    # r = output.r_surf
    # z = output.z_surf
    # A = 0.0
    # for i in range(len(z) - 1):
    #    for j in range(len(r) - 1):
    #        A += abs(z[i + 1] - z[i])*abs(r[i+1]-r[i])
    # print A
    if(np.nan in jpar_sorted):
        print("Nan in current")
        c_err.value = -1
    if(np.Inf in np.abs(jpar_sorted)):
        print("Warning inf encountered in the current")
        for i in range(len(rhop_sorted)):
            if(abs(jpar_sorted[i]) == np.inf):
                if(i > 0 and i < len(jpar_sorted) - 1):
                    if(abs(jpar_sorted[i + 1]) != np.inf and \
                        abs(jpar_sorted[i - 1]) != np.inf):
                        jpar_sorted[i] = jpar_sorted[i - 1] + \
                            (jpar_sorted[i + 1] - jpar_sorted[i - 1]) / \
                            (rhop_sorted[i + 1] - rhop_sorted[i - 1]) * \
                            (rhop_sorted[i] - rhop_sorted[i - 1])
    return rhop_sorted, jpar_sorted, c_err.value  # * 10.0,



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

def make_B_min(shot, time, rhop_vec, exp="AUGD", diag="EQH", ed=0):
    error = ct.c_int(0)
    fpf = 0.0
    c_exp = ct.c_char_p(exp)
    c_diag = ct.c_char_p(diag)
    shot = shot
    c_ed = ct.c_int(ed)
    c_time = ct.c_float(time)

    if(np.isscalar(rhop_vec)):
        npts = 1
    else:
        npts = len(rhop_vec)
    Rn = (ct.c_float * npts)()
    zn = (ct.c_float * npts)()
    R_mat = []
    z_mat = []
    c_rhop_vec = (ct.c_float * npts)()
    angle_cnt = 360
    if(npts == 1):
        c_rhop_vec[0] = ct.c_float(rhop_vec)
    else:
        for i in range(len(rhop_vec)):
            c_rhop_vec[i] = ct.c_float(rhop_vec[i])
    for angle in np.linspace(0.0, 360.0, angle_cnt):
        c_l_rhop = ct.c_int(npts)
        libkk.kkrhorz(ct.byref(error), c_exp, c_diag, ct.c_int(shot), ct.byref(c_ed), ct.byref(c_time), \
                          c_rhop_vec, c_l_rhop, 0, ct.c_float(angle), \
                          ct.byref(Rn), ct.byref(zn))
        if(npts != c_l_rhop.value):
            print("Problem with points")
            print(npts)
        else:
            R_mat.append(np.array([Rn[i] for i in range(npts)]))
            z_mat.append(np.array([zn[i] for i in range(npts)]))
    R_mat = np.array(R_mat).T
    z_mat = np.array(z_mat).T
    c_l_rhop = ct.c_int(npts)
    b_min_vec = np.zeros(npts)
    for i in range(npts):
        Rn = (ct.c_float * angle_cnt)()
        zn = (ct.c_float * angle_cnt)()
        bt = (ct.c_float * angle_cnt)()
        br = (ct.c_float * angle_cnt)()
        bz = (ct.c_float * angle_cnt)()
        fpf = (ct.c_float * angle_cnt)()
        jpol = (ct.c_float * angle_cnt)()
        for j in range(360):
            Rn[j] = ct.c_float(R_mat[i][j])
            zn[j] = ct.c_float(z_mat[i][j])
        libkk.kkrzbrzt(ct.byref(error), c_exp, c_diag, ct.c_int(shot), ct.byref(c_ed), ct.byref(c_time), \
               Rn, zn, c_l_rhop, \
               ct.byref(br), ct.byref(bz), ct.byref(bt), ct.byref(fpf), ct.byref(jpol))
        b_min = np.inf
        for j in range(angle_cnt):
            b_tot = np.sqrt(bt[j] ** 2 + br[j] ** 2 + bz[j] ** 2)
            if(b_tot < b_min and b_tot > 0):
                b_min = b_tot
        if(b_min == np.inf):
            print("Something wrong with b_min search")
        b_min_vec[i] = b_min
    return b_min_vec

def show_Volume():
    kk = KKeqi(33698, "AUGD", "EQH")
    vol = kk.get_volume(3.67, np.zeros(200))
    special_points = kk.get_special_points(3.55)
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111)
    ax.plot(np.sqrt((vol["pfl"] - special_points['pfxx'][0]) / (special_points['pfxx'][1] - special_points['pfxx'][0])), 2 * (1.0 + np.sqrt(vol["vol"]) / (1.65 * 2.e0 * np.pi)))
    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax.set_ylabel(r"$D\,[\si{\metre\squared\per\second}]$")
    plt.show()

class EQData(EQDataExt):
    def __init__(self, shot, external_folder='', EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005, Ext_data=False):
        EQDataExt.__init__(self, shot, external_folder, EQ_exp, EQ_diag, EQ_ed, bt_vac_correction, Ext_data)

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
        self.adjust_external_Bt_vac(B_t, R, opt.x[0], self.bt_vac_correction)
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
            self.GQH = dd.shotfile("GQH", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
        elif(self.EQ_diag == "IDE"):
            self.GQH = dd.shotfile("IDG", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
            self.IDF = dd.shotfile("IDF", int(self.shot), experiment=self.EQ_exp, edition=self.EQ_ed)
        else:
            print("EQ diagnostic {0:s} not supported - only EQH and IDE are currently supported!".format(self.EQ_diag))
        self.MBI_shot = dd.shotfile('MBI', int(self.shot))
        self.shotfile_ready = True

    def read_EQ_from_shotfile(self, time):
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
        return EQDataSlice(time, R, z, Psi, B_r, B_t, B_z, special_points=special_points, rhop=rhop)

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

    def rhop_to_rot(self, time, rhop):
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

    def get_R_aus(self, time, rhop_in):
        EQSlice = self.read_EQ_from_shotfile(time)
        R = EQSlice.R
        z = EQSlice.z
        rhop = EQSlice.rhop
        unwrap = False
        if(np.isscalar(rhop_in)):
            unwrap = True
            R_LFS = np.zeros(1)
            z_LFS = np.zeros(1)
            rhop_in = np.array([rhop_in])
        else:
            R_LFS = np.zeros(len(rhop_in))
            z_LFS = np.zeros(len(rhop_in))
        constraints = {}
        constraints["type"] = "eq"
        constraints["fun"] = eval_rhop
        rhop_spl = RectBivariateSpline(R, z, rhop)
        constraints["args"] = [rhop_spl, rhop_in[0]]
        options = {}
        options['maxiter'] = 100
        options['disp'] = False
        R_ax, z_ax = self.get_axis(time)
        x0 = np.array([R_ax, z_ax])
        for i in range(len(rhop_in)):
            constraints["args"][1] = rhop_in[i]
            res = scopt.minimize(eval_R, x0, method='SLSQP', bounds=[[1.2, 2.3], [-1.0, 1.0]], \
                                 constraints=constraints, options=options)
            if(not res.success):
                print("Error could not find R_aus for ", rhop_in[i])
                print("Cause: ", res.message)
                print("Falling back to axis position")
                R_LFS[i] = R_ax
                z_LFS[i] = z_ax
                x0 = np.array([R_ax, z_ax])
            else:
                R_LFS[i] = res.x[0]
                z_LFS[i] = res.x[1]
                x0 = res.x
#        plt.plot(R_LFS, z_LFS, "+r")
#        cont = plt.contour(R, z, rhop.T, levels=np.linspace(0.0, 1.2, 13))
#        plt.show()
        if(unwrap):
            return R_LFS[0], z_LFS[0]
        return R_LFS, z_LFS

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
