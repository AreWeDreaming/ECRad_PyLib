'''
Created on Jun 19, 2019

@author: sdenk
'''

# Distribution (GENE, LUKE, RELAX) class, beam class and distribution interpolator class
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import scipy.constants as cnst
import h5py
import os
from distribution_helper_functions import get_dist_moments_non_rel, get_0th_and_2nd_moment,remap_f_Maj
from plotting_configuration import *
from distribution_functions import Juettner1D

class beam:
    # For ECRH beams. Rays structure contains information on the individual rays forming the beam.
    def __init__(self, rhot, rhop, PW, j, PW_tot, j_tot, PW_beam=None, j_beam=None, rays=None):
        self.rhot = rhot
        self.rhop = rhop
        self.PW = PW / 1.e6  # W - MW
        self.j = j / 1.e6  # A - MA
        self.PW_tot = PW_tot / 1.e6  # W - MW
        self.j_tot = j_tot / 1.e6  # A - MA
        self.PW_beam = PW_beam
        self.j_beam = j_beam
        self.rays = rays

class Gene:
    # Distribution class for distribution functions given by GENE in the output format specified by T. Goerler
    def __init__(self, path, time=None, EqSlice=None, it=0, EQObj=None):
        h5_fileID = h5py.File(os.path.join(path, "xvsp_electrons_1e.h5"), 'r')['xvsp_electrons']
        gene_pos = h5py.File("/ptmp1/work/sdenk/ECRad7/xvspelectrons_1c.h5")
        if(EqSlice is None):
            if(time is None):
                print("The Gene class has to be initialized with either time or an EqSlice object present")
            if(EQObj is None):
                print("Either EqSlice or EQData must be present when GENE class is initialized")
            EqSlice = EQObj.GetSlice(time)
        beta_max = 1.0 # Cuts off distribution at high beta to avoid negative distribution, 1.0 -> no cut off
        self.R = np.array(gene_pos["axes"]["Rpos_m"]).flatten()
        self.z = np.array(gene_pos["axes"]["Zpos_m"]).flatten()
        dR_down = self.R[1] - self.R[0]
        dR_up = self.R[-1] - self.R[-2]
        dz_down = self.z[1] - self.z[0]
        dz_up = self.z[-1] - self.z[-2]
        self.R = np.concatenate([[self.R[0] - dR_down], self.R, [self.R[-1] + dR_up]])
        self.z = np.concatenate([[self.z[0] - dz_down], self.z, [self.z[-1] + dz_up]])
        self.v_par = np.array(h5_fileID["axes"]['vpar_m_s']).flatten() 
        self.beta_par = self.v_par / cnst.c
        self.mu_norm = np.array(h5_fileID["axes"]['mu_Am2']).flatten() / (cnst.m_e * cnst.c ** 2)
        self.total_time_cnt = len(h5_fileID['delta_f'].keys())
        self.g = np.array(h5_fileID['delta_f'][u'{0:010d}'.format(it)]).T * cnst.c ** 3
        self.f0 = np.array(h5_fileID['misc']['F0']).T * cnst.c ** 3
        self.Te = h5_fileID['general information'].attrs["Tref,eV"]
        self.ne = h5_fileID['general information'].attrs["nref,m^-3"]
        rhop_spl = RectBivariateSpline(EqSlice.R, EqSlice.z, EqSlice.rhop)
        self.rhop = rhop_spl(self.R, self.z, grid=False)
        self.B0 = h5_fileID["misc"].attrs['Blocal,T']
        self.beta_perp = np.sqrt(self.mu_norm * 2.0 * self.B0)
        self.f0 = self.f0[:, self.beta_perp < beta_max]
        self.f0 = self.f0[np.abs(self.beta_par) < beta_max, :]
        self.f0_log = np.copy(self.f0)
        self.f0_log[self.f0_log < 1.e-20] = 1.e-20
        self.f0_log = np.log(self.f0_log)
        self.g = self.g[:, :, self.beta_perp < beta_max]
        self.g = self.g[:, np.abs(self.beta_par) < beta_max, :]
        self.mu_norm = self.mu_norm[self.beta_perp < beta_max]
        self.beta_par = self.beta_par[np.abs(self.beta_par) < beta_max]
        self.f = np.concatenate([[self.f0], self.g + self.f0, [self.f0]])
        self.f_log = np.copy(self.f)
        self.f_log[self.f_log < 1.e-20] = 1.e-20
        self.f_log = np.log(self.f_log)
        rhopindex = np.argsort(self.rhop)
        self.rhop = self.rhop[rhopindex]

class Gene_BiMax(Gene):
    # Creates artifical GENE distribution based on the BiMaxwellian
    def __init__(self, path, shot, time=None, EqSlice=None, it=0, EQObj = None):
        Gene.__init__(self, path, shot, time, EqSlice, it, EQObj)

    def make_bi_max(self):
        self.Te_perp, self.Te_par = get_dist_moments_non_rel(self.rhop, self.beta_par, self.mu_norm, \
                                                             self.f, self.Te, self.ne, self.B0, \
                                                             slices=1, ne_out=False)
        
# Provides radial interpolation distribution functions

class f_interpolator:
    # Radial interpolation of distribution function
    # This method is completely analog to the ECRad implementation and even uses the same spline routines
    def __init__(self, working_dir=None, dist_obj=None, rhop_Bmin=None, Bmin=None, dist="Re", order=3, EqSlice=None):
        self.static_dist = False
        self.spl = None
        self.B0 = None
        self.B_min_spline = None
        self.order = order
        if(dist == "thermal"):
            self.thermal = True
            self.x = np.linspace(0.0, 3.0, 200)
            self.y = np.linspace(0.0, np.pi, 200)
            return
        else:
            self.thermal = False
        # ipsi, psi, x, y, Fe = read_Fe(os.path.join(working_dir, "ECRad_data") + os.path.sep)
        if(dist_obj is not None):
            self.rhop = dist_obj.rhop
            self.x = dist_obj.u
            self.y = dist_obj.pitch
            self.Fe = dist_obj.f_log
            self.B_min_spline = InterpolatedUnivariateSpline(rhop_Bmin, Bmin)
        else:
            if(dist == "Lu" or dist == "Re"):
                # Not recommended used distribution object instead to load from .mat files
                f_folder = os.path.join(working_dir, "ECRad_data", "f" + dist)
                x = np.loadtxt(os.path.join(f_folder, "u.dat"), skiprows=1)
                y = np.loadtxt(os.path.join(f_folder, "pitch.dat"), skiprows=1)
                self.rhop = np.loadtxt(os.path.join(f_folder, "frhop.dat"), skiprows=1)
                Fe = []
                for irhop in range(len(self.rhop)):
                    Fe.append(np.loadtxt(os.path.join(f_folder, "fu{0:03d}.dat".format(irhop))))
                Fe = np.array(Fe)
                rhop_B_min, B_min = np.loadtxt(os.path.join(working_dir, "ECRad_data", "B_min.dat"), unpack=1)
                self.B_min_spline = InterpolatedUnivariateSpline(rhop_B_min, B_min)
            elif(dist == "Ge" or dist == "GeO"):
                if(EqSlice is None):
                    print("An EqSlice must be provided to load GENE distribution data.")
                gene_obj = Gene(working_dir, time=None, EqSlice=EqSlice, it=0)
                x = gene_obj.beta_par
                y = gene_obj.beta_perp
                self.rhop = gene_obj.rhop
                self.B0 = gene_obj.B0
                if(dist == "Ge" ):
                    Fe = gene_obj.f_log
                else:
                    Fe = gene_obj.f0_log
                    self.static_dist = True
            else:
                print("Invalid distribution flag", dist)
                raise ValueError
            self.x = x
            self.y = y
            self.Fe = Fe
        if(self.static_dist):
            self.spline_mat = None
        else:
            # Spline to interpolate rho
            self.spline_mat = []
            for i in range(len(self.x)):
                self.spline_mat.append([])
                for j in range(len(self.y)):
                    self.spline_mat[-1].append(InterpolatedUnivariateSpline(self.rhop, self.Fe.T[j][i], k=1))

    def get_spline(self, rhop, Te):
        # Returns 2D spline at rhop for momentum space interpolation
        Fe_new = np.zeros((len(self.x), len(self.y)))
        if(self.thermal or rhop < np.min(self.rhop) or rhop > np.max(self.rhop)):
            for i in range(len(self.x)):
                f = Juettner1D(self.x[i], Te)
                if(f > 1.e-20):
                    Fe_new[i, :] = np.log(f)
                else:
                    Fe_new[i, :] = np.log(1.e-20)
            return self.x, self.y, RectBivariateSpline(self.x , self.y, Fe_new, kx=self.order, ky=self.order)
        elif(self.static_dist):
            return self.x, self.y, RectBivariateSpline(self.x , self.y, self.Fe, kx=self.order, ky=self.order)
        else:
            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    Fe_new[i][j] = self.spline_mat[i][j](rhop)
            return self.x, self.y, RectBivariateSpline(self.x , self.y, Fe_new, kx=self.order, ky=self.order)

class distribution:
    # Distribution class for bounce averaged distributions, for example from RELAX or LUKE
    def __init__(self, rhot, rhop, u, pitch, f, rhot_1D_profs, rhop_1D_profs, Te_init, ne_init, B_min=None):
        self.rhot = rhot
        self.rhop = rhop
        self.rhot_1D_profs = rhot_1D_profs
        self.rhop_1D_profs = rhop_1D_profs
        self.u = u
        self.pitch = pitch
        self.f = f
        if(B_min is not None):
            self.B_min = B_min
        else:
            self.B_min = np.zeros(len(self.rhop))
            self.B_min[:] = 100.0
        zero = 1.e-30
        self.f_log = f
        self.f_log[self.f_log < zero] = zero
        self.f_log10 = np.log10(self.f_log)
        self.f_log = np.log(self.f_log)
        self.ull = np.linspace(-np.max(u), np.max(u), 151)
        self.uxx = np.linspace(0, np.max(u), 75)
        self.f_cycl = np.zeros((len(self.rhop), len(self.ull), len(self.uxx)))
        self.f_cycl_log = np.zeros(self.f_cycl.shape)
        print("Remapping distribution hold on ...")
        self.Te_init = Te_init
        self.ne_init = ne_init
        ne_spl = InterpolatedUnivariateSpline(self.rhop_1D_profs[self.rhop_1D_profs < 1.0], self.ne_init[self.rhop_1D_profs < 1.0])
        self.ne = np.zeros(len(rhop))
        self.Te = np.zeros(len(rhop))
        for i in range(len(self.rhop)):
            # Remap for LFS
            remap_f_Maj(self.u, self.pitch, self.f_log, i, self.ull, self.uxx, self.f_cycl_log[i], 1, 1, LUKE=True)
            self.ne[i], self.Te[i] = get_0th_and_2nd_moment(self.ull, self.uxx, np.exp(self.f_cycl_log[i]))
            print("Finished distribution profile slice {0:d}/{1:d}".format(i + 1, len(self.rhop)))
        self.ne = self.ne * ne_spl(self.rhop)
        self.f_cycl = np.exp(self.f_cycl_log)
        self.f_cycl_log10 = np.log10(self.f_cycl)
        print("distribution shape:", self.f.shape)
        print("Finished remapping.")
