'''
Created on Jan 29, 2017

@author: sdenk
'''
from GlobalSettings import AUG, TCV, itm
import sys
import os
if(not itm):
    sys.path.append("/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib")
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import copy
from Geometry_utils import get_contour, get_Surface_area_of_torus, get_arclength, get_av_radius
from scipy import __version__ as scivers
import scipy.optimize as scopt

def eval_spline(x_vec, spl):
    return np.array([spl[0](x_vec[0], x_vec[1])])

def eval_spline_grad(x_vec, spl):
    return np.array([[spl[0].ev(x_vec[0], x_vec[1], dx=1, dy=0), spl[0].ev(x_vec[0], x_vec[1], dx=0, dy=1)]])


def eval_R(x):
    return -x[0] ** 3

def eval_rhop(x, spl, rhop_target):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1]) - rhop_target) ** 2
    else:
        return (spl(x[0], x[1], grid=False) - rhop_target) ** 2

def eval_Btot(x, args):
    spl = args[0]
    if(scivers == '0.12.0'):
        return spl.ev(x[0], x[1])
    else:
        return spl(x[0], x[1], grid=False)

class special_points:
    def __init__(self, R_ax, z_ax, psi_ax, R_sep, z_sep, psi_sep):
        self.Raxis = R_ax
        self.zaxis = z_ax
        self.Rspx = R_sep
        self.zspx = z_sep
        self.psiaxis = psi_ax
        self.psispx = psi_sep

class EQDataSlice:
    def __init__(self, time, R, z, Psi, Br, Bt, Bz, special=None, R_ax=None, Psi_sep=None, rhop=None, ripple=None):
        self.time = time
        self.R = R
        self.z = z
        self.Psi = Psi
        if(len(Psi.T[0]) != len(R) or len(Psi[0]) != len(z)):
            print("Shapes ", Psi.shape, R.shape, z.shape)
        self.rhop = rhop
        self.Br = Br
        self.Bt = Bt
        self.Bz = Bz
        if(special is not None):
            self.R_ax = special.Raxis
            self.z_ax = special.zaxis
            self.R_sep = special.Rspx
            self.z_sep = special.zspx
            self.Psi_ax = special.psiaxis
            self.Psi_sep = special.psispx
        elif(R_ax is not None and Psi_sep is not None):
            self.R_ax = R_ax
            self.Psi_sep = Psi_sep
        else:
            raise ValueError("Either special points or R_ax and Psi_sep must not be None")
        self.special = np.array([self.R_ax, self.Psi_sep])
        self.ripple = ripple

#        if(self.Psi_sep < self.Psi[self.Psi.shape[0] / 2][self.Psi.shape[1] / 2]):
#            self.Psi_sep *= -1.e0
#            self.Psi *= -1.e0

    def transpose_matrices(self):
        self.Br = self.Br.T
        self.Bt = self.Bt.T
        self.Bz = self.Bz.T
        self.Psi = self.Psi.T
        if(self.rhop is not None):
            self.rhop = self.rhop.T

    def inject_ripple(self, ripple):
        self.ripple = ripple

class EQDataExt:
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

    def adjust_external_Bt_vac(self, B_t, R, R_axis, bt_vac_correction):
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

    def insert_slices_from_ext(self, times, slices, transpose=False):
        self.slices = copy.deepcopy(slices)
        self.times = copy.deepcopy(times)
        self.Ext_data = True
        self.loaded = True
        if(transpose):
            for eq_slice in self.slices:
                eq_slice.transpose_matrices()

    def load_slices_from_mat(self, time, mdict):
        self.times = copy.deepcopy(time)
        self.slices = []
        self.Ext_data = True
        for it in range(len(time)):
            self.slices.append(EQDataSlice(self.times[it], mdict["R"], mdict["z"], mdict["Psi"][it], mdict["Br"][it], \
                                           mdict["Bt"][it], mdict["Bz"][it], R_ax=0.0, \
                                           Psi_sep=mdict["Psi_sep"][it]))
            R_ax, z_ax, Psi_ax = self.get_axis(self.times[it], get_Psi=True, force_no_load=True)
            self.slices[-1].R_ax = R_ax
            self.slices[-1].z_ax = z_ax
            self.slices[-1].R_sep = -1.e0
            self.slices[-1].z_sep = -1.e0
            self.slices[-1].Psi_ax = Psi_ax
            self.slices[-1].rhop = np.sqrt((self.slices[-1].Psi - self.slices[-1].Psi_ax) / \
                                           (self.slices[-1].Psi_sep - self.slices[-1].Psi_ax))
        self.loaded = True

    def read_EQ_from_Ext(self):
        t = np.loadtxt(os.path.join(self.external_folder, "t"), ndmin=1)
        self.times = []
        self.slices = []
        for it in range(len(t)):
            self.slices.append(self.read_EQ_from_Ext_single_slice(t[it], it))
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
        special = np.loadtxt(os.path.join(self.external_folder, "special{0:d}".format(index)))
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
        special_pnts = special(opt.x[0], opt.x[1], psi_ax, 2.17, 0.0, special[1])
        return EQDataSlice(time, R, z, Psi, B_r, B_t, B_z, special_pnts, rhop=rhop)

    def GetSlice(self, time):
        if(not self.loaded):
            print("EQObj is empty")
            raise ValueError
        else:
            itime = np.argmin(np.abs(self.times - time))
            return self.slices[itime]

    def get_axis(self, time, get_Psi=False):
        cur_slice = self.GetSlice(time)
        R = cur_slice.R
        z = cur_slice.z
        Psi = np.copy(cur_slice.Psi)
        if(Psi.shape[0] != len(R)):
            Psi = Psi
        special = cur_slice.special
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
        if(get_Psi):
            return opt.x[0], opt.x[1], psi_spl(opt.x[0], opt.x[1])
        else:
            return opt.x[0], opt.x[1]

    def get_B_on_axis(self, time):
        R_ax, z_ax = self.get_axis(time)
        if(self.external_folder != '' or self.Ext_data):
            cur_slice = self.GetSlice(time)
            R = cur_slice.R
            z = cur_slice.z
            B_tot = np.sqrt(cur_slice.Br ** 2 + cur_slice.Bt ** 2 + cur_slice.Bz ** 2)
            B_tot_spl = RectBivariateSpline(R, z, B_tot)
            return B_tot_spl(R_ax, z_ax)
        else:
            print("Parent class EQData does not support shotfile handling - use AUG child instead")

    def get_surface_area(self, time, rho):
        if(rho > 1.0):
            print("Plasma surface area not defined for open flux surfaces!")
            raise ValueError
        R, z = self.get_Rz_contour(time, rho)
#        plt.plot(cont[closed][0][0], cont[closed][0][1])
#        plt.show()
        A = get_Surface_area_of_torus(R, z)  # makes sure only closed contours enters
        return A

    def get_Rz_contour(self, time, rhop_in):
        cur_slice = self.GetSlice(time)
        R = cur_slice.R
        z = cur_slice.z
        rhop = cur_slice.rhop
        info, cont = get_contour(R, z, rhop, rhop_in)
        return cont.flatten()

    def get_mean_r(self, time, rhop):
        rhop_ar = np.atleast_1d(rhop)
        R_av = []
        R_ax, z_ax = self.get_axis(time)
        cont = self.get_Rz_contour(time, rhop, only_single_closed=True)
#        for cont in sorted_conts:
#            plt.plot(cont.T[0], cont.T[1], "-")
#        plt.show()
        for R_cont, z_cont, rho in zip(cont.T[0], cont.T[1], rhop_ar):
#            plt.plot(R_cont, z_cont)
            S = get_arclength(R_cont, z_cont)
            R_av.append(get_av_radius(R_cont, z_cont, S, R_ax, z_ax))
            print(rho, S, R_av[-1], 2.0 * np.pi * R_av[-1])
        if(np.isscalar(rhop)):
            R_av = R_av[0]
        else:
            R_av = np.array(R_av)
        return R_av

    def get_B_min(self, time, rhop_in, append_B_ax=False):
        cur_slice = self.GetSlice(time)
        R = cur_slice.R
        z = cur_slice.z
        Br = cur_slice.Br
        Bt = cur_slice.Bt
        Bz = cur_slice.Bz
        rhop = cur_slice.rhop
        Btot_spl = RectBivariateSpline(R, z, np.sqrt(Br ** 2 + Bt ** 2 + Bz ** 2))
        unwrap = False
        if(np.isscalar(rhop_in)):
            unwrap = True
            B_min = np.zeros(1)
        else:
            B_min = np.zeros(len(rhop_in))
        constraints = {}
        constraints["type"] = "eq"
        constraints["fun"] = eval_rhop
        rhop_spl = RectBivariateSpline(R, z, rhop)
        constraints["args"] = [rhop_spl, rhop_in[0]]
        options = {}
        options['maxiter'] = 100
        options['disp'] = False
        R_ax, z_ax = self.get_axis(time)
        B_ax = Btot_spl(R_ax, z_ax, grid=False)
        x0 = np.array([R_ax, z_ax])
        for i in range(len(rhop_in)):
            if(rhop_in[i] == 0):
                B_min[i] = 0
                continue
            constraints["args"][1] = rhop_in[i]
            res = scopt.minimize(eval_Btot, x0, method='SLSQP', bounds=[[1.2, 2.3], [-1.0, 1.0]], \
                                 constraints=constraints, options=options, args=[Btot_spl])
            if(not res.success):
                print("Error could not find Bmin for ", rhop_in[i])
                print("Cause: ", res.message)
                if(rhop_in[i] < 0.1 and not unwrap):
                    print("Current rhop very small -> will interpolate using magn. axis value")
                    B_min[i]  = 0.0
                else:
                    raise ValueError
            else:
                B_min[i] = Btot_spl(res.x[0], res.x[1])
        if(unwrap):
            return B_min[0]
        B_min[rhop_in == 0] = B_ax
        if(np.any(B_min == 0)):
            rhop_short  =  np.copy(rhop_in[B_min != 0])
            B_min_short =  np.copy(B_min[B_min != 0])
            if(0.0 not in rhop_short):
                rhop_short  =  np.concatenate([[0], rhop_short])
                B_min_short =  np.concatenate([[B_ax], B_min_short])
            B_min_spl = InterpolatedUnivariateSpline(rhop_short, B_min_short)
            B_min = B_min_spl(rhop_in)
        if(append_B_ax):
            if(0.0 in rhop_in):
                return np.copy(rhop_in), B_min
            return np.concatenate([[0], rhop_in]), np.concatenate([[B_ax], B_min])
        else:
            return B_min

    def get_R_aus(self, time, rhop_in):
        cur_slice = self.GetSlice(time)
        R = cur_slice.R
        z = cur_slice.z
        rhop = cur_slice.rhop
        unwrap = False
        if(np.isscalar(rhop_in)):
            unwrap = True
            R_LFS = np.zeros(1)
            z_LFS = np.zeros(1)
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
    #    plt.plot(R_LFS, z_LFS, "+r")
    #    cont = plt.contour(R, z, Psi.T, levels=Psi_grid)
    #    plt.show()
        if(unwrap):
            return R_LFS[0], z_LFS[0]
        return R_LFS, z_LFS

    def map_Rz_to_rhop(self, time, R, z):
        cur_slice = self.GetSlice(time)
        if(hasattr(cur_slice, "rhop")):
            rhop_spl = RectBivariateSpline(cur_slice.R, cur_slice.z, cur_slice.rhop)
            return rhop_spl(R,z, grid=False)
        if(hasattr(cur_slice, "Psi_ax")):
            rhop = np.sqrt((cur_slice.Psi - cur_slice.Psi_ax) / (cur_slice.Psi_sep - cur_slice.Psi_ax))
            rhop_spl = RectBivariateSpline(cur_slice.R, cur_slice.z, rhop)
            return rhop_spl(R,z, grid=False)
        Psi = cur_slice.Psi
        special = cur_slice.special
        if(Psi[len(cur_slice.R) / 2][len(cur_slice.z) / 2] > special[1]):
        # We want a minimum in the flux at the magn. axis
            Psi *= -1.0
            special[1] *= -1.0
        psi_spl = RectBivariateSpline(cur_slice.R, cur_slice.z, Psi)
        indicies = np.unravel_index(np.argmin(Psi), Psi.shape)
        R_init = np.array([cur_slice.R[indicies[0]], cur_slice.z[indicies[1]]])
        opt = minimize(eval_spline, R_init, args=[psi_spl], \
                 bounds=[[np.min(cur_slice.R), np.max(cur_slice.R)], [np.min(cur_slice.z), np.max(cur_slice.z)]])
        psi_ax = psi_spl(opt.x[0], opt.x[1])
        rhop = np.sqrt((Psi - psi_ax) / (special[1] - psi_ax))
        rhop_spl = RectBivariateSpline(cur_slice.R, cur_slice.z, rhop)
        return rhop_spl(R, z, grid=False)


