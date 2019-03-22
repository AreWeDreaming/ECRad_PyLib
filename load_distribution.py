'''
Created on Oct 4, 2016

@author: sdenk
'''
import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import scipy.constants as cnst
from scipy.integrate import simps
from scipy import __version__ as scivers
import scipy.optimize as scopt

def load_f_from_ASCII(path, rhop_in=None):
    x = np.loadtxt(os.path.join(path, "u.dat"), skiprows=1)
    y = np.loadtxt(os.path.join(path, "pitch.dat"), skiprows=1)
    ne_data = np.loadtxt(os.path.join(path, "..", "ne_file.dat"), skiprows=1)
    Te_data = np.loadtxt(os.path.join(path, "..", "Te_file.dat"), skiprows=1)
    rhop_ne = ne_data.T[0]
    ne = ne_data.T[1]
    Te = Te_data.T[1]
    rhop = np.loadtxt(os.path.join(path, "frhop.dat"), skiprows=1)
    if(rhop_in is not None):
        irhop = np.argmin(np.abs(rhop - rhop_in))
        rhop = np.array([rhop[irhop]])
        Fe = np.array([np.loadtxt(os.path.join(path, "fu{0:03n}.dat".format(irhop)))])
    else:
        Fe = np.zeros((len(rhop), len(x), len(y)))
        for irhop in range(len(rhop)):
            Fe[irhop, :, :] = np.loadtxt(os.path.join(path, "fu{0:03n}.dat".format(irhop)))
    return distribution(None, rhop, x, y, np.exp(Fe), None, rhop_ne, Te, ne)



class distribution:
    def __init__(self, rhot, rhop, u, pitch, f, rhot_1D_profs, rhop_1D_profs, Te_init, ne_init):
        self.rhot = rhot
        self.rhop = rhop
        self.rhot_1D_profs = rhot_1D_profs
        self.rhop_1D_profs = rhop_1D_profs
        self.u = u
        self.pitch = pitch
        self.f = f
        zero = 1.e-30
        self.f_log = f
        self.f_log[self.f_log < zero] = zero
        self.f_log = np.log10(self.f_log)
        self.ull = np.linspace(-np.max(u), np.max(u), 100)
        self.uxx = np.linspace(0, np.max(u), 100)
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
            self.ne[i], self.Te[i] = get_0th_and_2nd_moment(self.ull, self.uxx, 10.0 ** self.f_cycl_log[i])
            # print(self.Te[i], self.ne[i])
            print("Finished distribution profile slice {0:d}/{1:d}".format(i + 1, len(self.rhop)))
        self.ne = self.ne * ne_spl(self.rhop)
        self.f_cycl = 10.0 ** self.f_cycl
        print("distribution shape:", self.f.shape)
        print("Finished remapping.")

def eval_R(x):
    return -x[0] ** 3

def eval_Psi(x, spl, psi_target):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1]) - psi_target) ** 2
    else:
        return (spl(x[0], x[1], grid=False) - psi_target) ** 2

def get_R_aus(R, z, Psi, R_ax, z_ax, Psi_target):
    unwrap = False
    if(np.isscalar(Psi_target)):
        unwrap = True
    R_LFS = np.zeros(len(Psi_target))
    z_LFS = np.zeros(len(Psi_target))
    constraints = {}
    constraints["type"] = "eq"
    constraints["fun"] = eval_Psi
    psi_spl = RectBivariateSpline(R, z, Psi)
    constraints["args"] = [psi_spl, Psi_target[0]]
    options = {}
    options['maxiter'] = 100
    options['disp'] = False
    x0 = np.array([R_ax, z_ax])
    for i in range(len(Psi_target)):
        constraints["args"][1] = Psi_target[i]
        res = scopt.minimize(eval_R, x0, method='SLSQP', bounds=[[1.2, 2.3], [-1.0, 1.0]], \
                             constraints=constraints, options=options)
        if(not res.success):
            print("Error could not find R_aus for ", Psi_target[i])
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

def remap_f_Maj(x, y, Fe, ipsi, ull, uxx, Fe_rem, freq_2X, B0, LUKE=False):

    """ Comment this!
    """

    # First generate an interpolation object on the rectangular x-y grid
#    if(LUKE):
#        Fe_int = RectBivariateSpline(x, y, Fe[:, :, ipsi])
#    else:
    Fe_int = RectBivariateSpline(x, y, Fe[ipsi], kx=3, ky=3)
    # Initialize the arrays for pll, pxx on the point of crossing
    # pmax = max(x)
    # npts = 100
    # pll = np.linspace(-pmax, +pmax, 2 * npts)
    # pxx = np.linspace(0., pmax, npts)
    # pll, pxx = np.meshgrid(pll, pxx)
    xi = freq_2X * np.pi * cnst.m_e / (B0 * cnst.e)
    if(xi < 1.0):
        xi = 1.0

    # Fe_rem= Fe_remapped(x,y,Fe[ipsi], xi, pll, pxx)
    # Corresponding equatorial pitch-angle cosine
    # Remapped distribution function
    for i in range(len(ull)):
        for j in range(len(uxx)):
            x_eq = np.sqrt(ull[i] ** 2 + uxx[j] ** 2)
            if(LUKE):
                y_eq = ull[i] / x_eq
            else:
                mu = np.sqrt((ull[i] ** 2 + uxx[j] ** 2 * (xi - 1.) / xi) / (ull[i] ** 2 + uxx[j] ** 2))
                mu = np.copysign(mu, ull[i])
                while(mu > 1.0):
                    mu += -1.0
                while(mu < -1.0):
                    mu += 1.0
                y_eq = np.arccos(mu)
            Fe_rem[i][j] = Fe_int(x_eq, y_eq)
            # print("point:", uxx[j], ull[i], x_eq, y_eq, Fe_rem[i][j])
            # if(x_cur > 0.5):
            #   print(Fe_rem[i][j])
    # Fe_rem = Fe_rem.reshape(np.shape(pll))
    # Exit
    return Fe_rem


def get_0th_and_2nd_moment(ull, uxx, f):
    f_0th_moment = np.zeros(len(ull))
    f_1th_moment = np.zeros(len(ull))
    f_2th_moment = np.zeros(len(ull))
    f_mean_gamma = np.zeros(len(ull))
    for i in range(len(ull)):
        f_0th_moment[i] = simps(uxx * f[i, :] * np.pi * 2.e0, uxx)
        f_1th_moment[i] = simps(uxx * f[i, :] * np.pi * 2.e0 * ull[i], uxx)
    zeros_mom = simps(f_0th_moment, ull)
    first_mom = simps(f_1th_moment, ull)
    for i in range(len(ull)):
        u_sq = uxx ** 2 + (ull[i] - first_mom / zeros_mom) ** 2
        gamma_sq = (1 + u_sq)
        f_2th_moment[i] = simps(uxx * f[i, :] * np.pi * 2.e0 * u_sq / np.sqrt(gamma_sq), uxx)
        # f_mean_gamma[i] = simps(uxx * f[i, :] * np.pi * 2.e0 * gamma, uxx)
    # print(zeros_mom, second_mom)
    second_mom = simps(f_2th_moment, ull)
    # gamma_mean = simps(f_mean_gamma, ull)
    Te = cnst.c ** 2 * cnst.m_e / cnst.e * second_mom / zeros_mom / 3.0
    return zeros_mom, Te
load_f_from_ASCII("/afs/ipp-garching.mpg.de/home/s/sdenk/public/fRelax/")
