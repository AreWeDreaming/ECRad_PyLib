'''
Created on Jun 19, 2019

@author: Severin Denk
'''
# Collection of operations on distributions, like computing 1st moment ect.
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import scipy.constants as cnst
from scipy.integrate import simps 
from scipy.integrate import nquad
from Distribution_Functions import Juettner2D, Juettner1D, rel_thermal_beta

def zeros_mom_non_rel(betall, betaxx, f_spl):
    return 2.0 * np.pi * betaxx * np.exp(f_spl(betall, betaxx))

def first_mom_non_rel(betall, betaxx, f_spl):
    return 2.0 * np.pi * betall * betaxx * np.exp(f_spl(betall, betaxx))

def scnd_mom_non_rel(betall, betaxx, f_spl):
    beta_sq = (betall ** 2 + betaxx ** 2)
    return 2.0 * np.pi * beta_sq * betaxx * np.exp(f_spl(betall, betaxx))

def scnd_mom_par_non_rel(betall, betaxx, f_spl, beta_par_mean):
    return 2.0 * np.pi * (betall - beta_par_mean) ** 2 * betaxx * np.exp(f_spl(betall, betaxx))

def scnd_mom_perp_non_rel(betall, betaxx, f_spl):
    return 2.0 * np.pi * betaxx ** 3 * np.exp(f_spl(betall, betaxx))

def zeros_mom(betall, betaxx, f_spl):
    gamma = 1.0 / np.sqrt(1.0 - betall ** 2 - betaxx ** 2)
    return 2.0 * np.pi * betaxx * gamma ** 5 * np.exp(f_spl(betall, betaxx))

def first_mom(betall, betaxx, f_spl):
    gamma = 1.0 / np.sqrt(1.0 - betall ** 2 - betaxx ** 2)
    return 2.0 * np.pi * betall * betaxx * gamma ** 5 * np.exp(f_spl(betall, betaxx))

def scnd_mom(betall, betaxx, f_spl):
    beta_sq = (betall ** 2 + betaxx ** 2)
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    return 2.0 * np.pi * beta_sq * betaxx * gamma ** 6 * np.exp(f_spl(betall, betaxx))

def scnd_mom_par(betall, betaxx, f_spl, beta_par_mean):
    gamma = 1.0 / np.sqrt(1.0 - betall ** 2 - betaxx ** 2)
    return 2.0 * np.pi * (betall - beta_par_mean) ** 2 * betaxx * gamma ** 6 * np.exp(f_spl(betall, betaxx))

def scnd_mom_perp(betall, betaxx, f_spl):
    gamma = 1.0 / np.sqrt(1.0 - betall ** 2 - betaxx ** 2)
    return 2.0 * np.pi * betaxx ** 3 * gamma ** 6 * np.exp(f_spl(betall, betaxx))

def zeros_mom_u(ull, uxx, f_spl):
    return 2.0 * np.pi * uxx * np.exp(f_spl(ull, uxx))

def scnd_mom_par_u(ull, uxx, f_spl, ull_mean):
    gamma = np.sqrt(1.e0 + ull ** 2 + uxx ** 2)
    return 2.0 * np.pi * (ull - ull_mean) ** 2 / gamma ** 2 * uxx * np.exp(f_spl(ull, uxx))

def scnd_mom_perp_u(ull, uxx, f_spl):
    gamma = np.sqrt(1.e0 + ull ** 2 + uxx ** 2)
    return 2.0 * np.pi * uxx ** 3 / gamma ** 2 * np.exp(f_spl(ull, uxx))


def get_E_perp_and_E_par(ull_min, ull_max, uxx_min, uxx_max, f_spl):
    normalization = nquad(zeros_mom_u, [[ull_min, ull_max], [uxx_min, uxx_max]], args=[f_spl], opts={"epsabs":1.e-5, \
                                                                                                  "epsrel":1.e-4})[0]
    Ell_th = nquad(scnd_mom_par_u, [[ull_min, ull_max], [uxx_min, uxx_max]], args=[f_spl, 0.0], \
                   opts={"epsabs":1.e-4, "epsrel":1.e-4})[0]
    Exx_th = nquad(scnd_mom_perp_u, [[ull_min, ull_max], [uxx_min, uxx_max]], args=[f_spl], opts={"epsabs":1.e-5, \
                                                                                                  "epsrel":1.e-5})[0]
    Exx_th *= cnst.c ** 2 * cnst.m_e / cnst.e / normalization / 2.0
    Ell_th *= cnst.c ** 2 * cnst.m_e / cnst.e / normalization
    return Exx_th, Ell_th

def get_thermal_av_cyc_freq(Te, f_c):
    uxx = np.linspace(0.0, 2.0, 200)
    ull = np.linspace(-2.0, 2.0, 200)
    f = np.zeros((len(ull), len(uxx)))
    f_0th_moment = np.zeros(len(ull))
    f_1th_moment = np.zeros(len(ull))
    f_2th_moment = np.zeros(len(ull))
    for i in range(len(ull)):
        f[i] = Juettner2D(ull[i], uxx, Te)
        f_0th_moment[i] = simps(uxx * f[i, :] * np.pi * 2.e0, uxx)
        f_1th_moment[i] = simps(uxx * f[i, :] * np.pi * 2.e0 * ull[i], uxx)
    zeros_mom = simps(f_0th_moment, ull)
    first_mom = simps(f_1th_moment, ull)
    for i in range(len(ull)):
        u_sq = uxx ** 2 + (ull[i] - first_mom / zeros_mom) ** 2
        gamma_sq = (1 + u_sq)
        f_2th_moment[i] = simps(uxx * f[i, :] * np.pi * 2.e0 * 1.0 / np.sqrt(gamma_sq), uxx)
        # f_mean_gamma[i] = simps(uxx * f[i, :] * np.pi * 2.e0 * gamma, uxx)
    # print(zeros_mom, second_mom)
    av_cyc_freq = f_c * simps(f_2th_moment, ull)
    return av_cyc_freq

def get_bimaxwellian_moments(betall_min, betall_max, betaxx_min, betaxx_max, f_spl, ne_out=False):
    normalization = nquad(zeros_mom, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl], opts={"epsabs":1.e-5, \
                                                                                                  "epsrel":1.e-4})[0]
    mean_u_par = nquad(first_mom, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl], opts={"epsabs":1.e-4, \
                                                                                                  "epsrel":1.e-4})[0]
    Ell_th = nquad(scnd_mom_par, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl, mean_u_par / normalization], \
                   opts={"epsabs":1.e-4, "epsrel":1.e-4})[0]
    Exx_th = nquad(scnd_mom_perp, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl], opts={"epsabs":1.e-5, \
                                                                                                  "epsrel":1.e-5})[0]
    Te_perp = cnst.c ** 2 * cnst.m_e / cnst.e * Exx_th / normalization / 2.0
    Te_par = cnst.c ** 2 * cnst.m_e / cnst.e * Ell_th / normalization
    if(not ne_out):
        return Te_perp, Te_par
    else:
        return Te_perp, Te_par, normalization

def get_bimaxwellian_moments_non_rel(betall_min, betall_max, betaxx_min, betaxx_max, f_spl, ne_out=False):
    normalization = nquad(zeros_mom_non_rel, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl], opts={"epsabs":1.e-5, \
                                                                                                  "epsrel":1.e-4})[0]
    mean_u_par = nquad(first_mom_non_rel, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl], opts={"epsabs":1.e-4, \
                                                                                                  "epsrel":1.e-4})[0]
    Ell_th = nquad(scnd_mom_par_non_rel, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl, mean_u_par / normalization], \
                   opts={"epsabs":1.e-4, "epsrel":1.e-4})[0]
    Exx_th = nquad(scnd_mom_perp_non_rel, [[betall_min, betall_max], [betaxx_min, betaxx_max]], args=[f_spl], opts={"epsabs":1.e-5, \
                                                                                                  "epsrel":1.e-5})[0]
    Te_perp = cnst.c ** 2 * cnst.m_e / cnst.e * Exx_th / normalization / 2.0
    Te_par = cnst.c ** 2 * cnst.m_e / cnst.e * Ell_th / normalization
    if(not ne_out):
        return Te_perp, Te_par
    else:
        return Te_perp, Te_par, normalization

def get_E_perp_and_E_par_profile(dist_obj):
    E_par = np.zeros(len(dist_obj.rhop))
    E_perp = np.zeros(len(dist_obj.rhop))
    for i in range(len(dist_obj.rhop)):
        print("Now working on flux surface {0:d}/{1:d}".format(i + 1, len(dist_obj.rhop)))
        f_spl = RectBivariateSpline(dist_obj.ull, dist_obj.uxx, np.log(dist_obj.f_cycl[i]))
        E_perp[i], E_par[i] = get_E_perp_and_E_par(np.min(dist_obj.ull), np.max(dist_obj.ull), \
                                                         np.min(dist_obj.uxx), np.max(dist_obj.uxx), f_spl)
    return E_perp, E_par

def get_dist_moments(rhop, beta_par, mu_norm, f, Te, ne, B0, slices=1, ne_out=False):
    rhop_Gene = np.copy(rhop)
    f_Gene = np.copy(f)
    if(slices > 1):
        rhop_Gene = rhop_Gene[::slices]
        f_Gene = f_Gene[::slices]
    Te_par = np.zeros(len(rhop_Gene))
    Te_perp = np.zeros(len(rhop_Gene))
    if(ne_out):
        ne_prof = np.zeros(len(rhop))
    f_Gene /= ne
    f_Gene[f_Gene < 1.e-20] = 1.e-20
    for i in range(len(rhop_Gene)):
        print("Now working on flux surface {0:d}/{1:d}".format(i + 1, len(rhop_Gene)))
        beta_perp = np.sqrt(mu_norm * 2.0 * B0)
        f_spl = RectBivariateSpline(beta_par, beta_perp, np.log(f_Gene[i]))
        if(ne_out):
            Te_perp[i], Te_par[i], ne_prof[i] = get_bimaxwellian_moments(np.min(beta_par), np.max(beta_perp), \
                                                         np.min(beta_perp), np.max(beta_perp), f_spl, ne_out=True)
        else:
            Te_perp[i], Te_par[i] = get_bimaxwellian_moments(np.min(beta_par), np.max(beta_perp), \
                                                         np.min(beta_perp), np.max(beta_perp), f_spl)
        print("Te, Te_perp, Te_par, 1 - Te_perp/Te, 1 - Te_par/Te", Te, \
              Te_perp[i], Te_par[i], (1.0 - Te_perp[i] / Te), (1.0 - Te_par[i] / Te))
    if(ne_out):
        return Te_perp, Te_par, ne_prof
    else:
        return Te_perp, Te_par

def get_dist_moments_non_rel(rhop, beta_par, mu_norm, f, Te, ne, B0, slices=1, ne_out=False):
    rhop_Gene = np.copy(rhop)
    f_Gene = np.copy(f)
    if(slices > 1):
        rhop_Gene = rhop_Gene[::slices]
        f_Gene = f_Gene[::slices]
    Te_par = np.zeros(len(rhop_Gene))
    Te_perp = np.zeros(len(rhop_Gene))
    if(ne_out):
        ne_prof = np.zeros(len(rhop))
    f_Gene /= ne
    f_Gene[f_Gene < 1.e-20] = 1.e-20
    for i in range(len(rhop_Gene)):
#         print("Now working on flux surface {0:d}/{1:d}".format(i + 1, len(rhop_Gene)))
        beta_perp = np.sqrt(mu_norm * 2.0 * B0)
        f_spl = RectBivariateSpline(beta_par, beta_perp, np.log(f_Gene[i]))
        if(ne_out):
            Te_perp[i], Te_par[i], ne_prof[i] = get_bimaxwellian_moments_non_rel(np.min(beta_par), np.max(beta_perp), \
                                                         np.min(beta_perp), np.max(beta_perp), f_spl, ne_out=True)
        else:
            Te_perp[i], Te_par[i] = get_bimaxwellian_moments_non_rel(np.min(beta_par), np.max(beta_perp), \
                                                         np.min(beta_perp), np.max(beta_perp), f_spl)
#         print("Te, Te_perp, Te_par, 1 - Te_perp/Te, 1 - Te_par/Te", Te, \
#               Te_perp[i], Te_par[i], (1.0 - Te_perp[i] / Te), (1.0 - Te_par[i] / Te))
    if(ne_out):
        return Te_perp, Te_par, ne_prof
    else:
        return Te_perp, Te_par
    
    
def get_0th_and_2nd_moment(ull, uxx, f):
    f_0th_moment = np.zeros(len(ull))
    f_1th_moment = np.zeros(len(ull))
    f_2th_moment = np.zeros(len(ull))
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


def Fe_remapped(x, y, Fe, xi, ull, uxx, LUKE=False):

    """ Comment this!
    """

    # First generate an interpolation object on the rectangular x-y grid
    Fe_int = RectBivariateSpline(x, y, Fe, kx=1, ky=1)

    # Initialize the arrays for pll, pxx on the point of crossing


    # Corresponding equatorial pitch-angle cosine
    # mu = np.sqrt((ull**2 + uxx**2 * (xi - 1.) / xi) / (ull**2 + uxx**2))
    # mu = np.copysign(mu, ull)

    # Remapped coordinates on the equatorial plane
    # x_eq = np.sqrt(ull**2 + uxx**2)
    # while(any(mu > 1.0):
    #    mu += -1.0
    # while(mu < -1.0):
    #    mu += 1.0
    # y_eq = np.arccos(mu)
    print("shape", np.shape(ull), np.shape(uxx))
    # Remapped distribution function
    Fe_rem = np.zeros([len(ull), len(uxx)])
    for i in range(len(ull)):
        for j in range(len(uxx)):
            x_eq = np.sqrt(ull[i] ** 2 + uxx[j] ** 2)
            if(LUKE):
                y_eq = ull[i] / x_eq
            else:
                mu = np.sqrt((ull[i] ** 2 + uxx[j] ** 2 * (xi - 1.) / xi) / (ull[i] ** 2 + uxx[j] ** 2))
                mu = np.copysign(mu, ull[i])
            y_eq = np.arccos(mu)
            Fe_rem[i][j] = Fe_int(x_eq, y_eq)  # .flatten()
    # Fe_rem = Fe_rem.reshape(np.shape(ull))

    # Exit
    return Fe_rem

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
            if(x_eq == 0.0):
                y_eq = 0.0
            elif(LUKE):
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

def remap_f_Maj_single(x, y, Fe, ull, uxx, Fe_rem, freq_2X, B0, LUKE=False):

    """ Comment this!
    """

    # First generate an interpolation object on the rectangular x-y grid
#    if(LUKE):
#        Fe_int = RectBivariateSpline(x, y, Fe[:, :, ipsi])
#    else:
    Fe_int = RectBivariateSpline(x, y, Fe, kx=3, ky=3)
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

def remap_f_Maj_res_single(x, y, Fe, ull, uxx, Fe_rem, freq_2X, B0, LUKE=False):

    """ Comment this!
    """

    # First generate an interpolation object on the rectangular x-y grid
    try:
        Fe_int = RectBivariateSpline(x, y, Fe, kx=3, ky=3)
    except TypeError as e:
        print(e)
        print(x.shape, y.shape, Fe.shape)
        raise TypeError

    # Initialize the arrays for pll, pxx on the point of crossing
    # pmax = max(x)
    # npts = 100
    # pll = np.linspace(-pmax, +pmax, 2 * npts)
    # pxx = np.linspace(0., pmax, npts)
    # pll, pxx = np.meshgrid(pll, pxx)
    xi = freq_2X * np.pi * cnst.m_e / (B0 * cnst.e)
    if(xi < 1.0):
        xi = 1.0
    # Corresponding equatorial pitch-angle cosine

    # Remapped coordinates on the equatorial plane
    for i in range(len(ull)):
        x_eq = np.sqrt(ull[i] ** 2 + uxx[i] ** 2)
        if(x_eq > np.max(x)):
            Fe_rem[i] = 0.e0
        else:
#            if(LUKE):
#                y_eq = ull[i] / x_eq
#            else:
            mu = np.sqrt((ull[i] ** 2 + uxx[i] ** 2 * (xi - 1.) / xi) / (ull[i] ** 2 + uxx[i] ** 2))
            mu = np.copysign(mu, -ull[i])
            # while(mu > 1.0):
            #    mu += -1.0
            # while(mu < -1.0):
            #    mu += 1.0
            y_eq = np.arccos(mu)
            Fe_rem[i] = Fe_int.ev(x_eq, y_eq)
    # Remapped distribution function
    # Fe_rem = Fe_int.ev(x_eq.flatten(), y_eq.flatten())
    # Fe_rem = Fe_rem.reshape(np.shape(pll))

    # Exit
    return Fe_rem



def remap_f_Maj_res(x, y, Fe, ipsi, ull, uxx, Fe_rem, freq_2X, B0, LUKE=False):

    """ Comment this!
    """

    # First generate an interpolation object on the rectangular x-y grid
    # try:
    Fe_int = RectBivariateSpline(x, y, Fe, kx=3, ky=3)
    # except TypeError:
    #    print(x)

    # Initialize the arrays for pll, pxx on the point of crossing
    # pmax = max(x)
    # npts = 100
    # pll = np.linspace(-pmax, +pmax, 2 * npts)
    # pxx = np.linspace(0., pmax, npts)
    # pll, pxx = np.meshgrid(pll, pxx)
    xi = freq_2X * np.pi * cnst.m_e / (B0 * cnst.e)
    if(xi < 1.0):
        xi = 1.0
    # Corresponding equatorial pitch-angle cosine

    # Remapped coordinates on the equatorial plane
    for i in range(len(ull)):
        x_eq = np.sqrt(ull[i] ** 2 + uxx[i] ** 2)
#        if(LUKE):
#            y_eq = ull[i] / x_eq
#        else:
        mu = np.sqrt((ull[i] ** 2 + uxx[i] ** 2 * (xi - 1.) / xi) / (ull[i] ** 2 + uxx[i] ** 2))
        mu = np.copysign(mu, ull[i])
        y_eq = np.arccos(mu)
        Fe_rem[i] = Fe_int.ev(x_eq, y_eq)
    # Remapped distribution function
    # Fe_rem = Fe_int.ev(x_eq.flatten(), y_eq.flatten())
    # Fe_rem = Fe_rem.reshape(np.shape(pll))

    # Exit
    return Fe_rem

def remap_f1D(x, y, Fe_rhop, uxx, ull, Fe_remapped):
    spline = RectBivariateSpline(x, y, Fe_rhop, kx=5, ky=5)
    for i in range(len(ull)):
        cur_x = np.sqrt(uxx ** 2 + ull[i] ** 2)
        cur_y = np.arctan2(uxx, ull[i])
        Fe_remapped[i] = spline.ev(cur_x, cur_y)
        # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped

def remap_f1D_uxx(x, y, Fe_rhop, uxx, ull, Fe_remapped):
    spline = RectBivariateSpline(x, y, Fe_rhop, kx=5, ky=5)
    for i in range(len(uxx)):
        cur_x = np.sqrt(uxx[i] ** 2 + ull ** 2)
        cur_y = np.arctan2(uxx[i], ull)
        Fe_remapped[i] = spline.ev(cur_x, cur_y)
        # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped

def remap_f_res(x, y, Fe_rhop, uxx, ull, Fe_remapped):
    spline = RectBivariateSpline(x, y, Fe_rhop, kx=5, ky=5)
    for i in range(len(ull)):
        cur_x = np.sqrt(uxx[i] ** 2 + ull[i] ** 2)
        cur_y = np.arctan2(uxx[i], ull[i])
        Fe_remapped[i] = spline.ev(cur_x, cur_y)
            # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped

def fill_zeros_with_thermal(Fe, rhop_LUKE, rhop_Te, Te, u):
    # Fills all zero values with thermal distributions
    Te_spl = InterpolatedUnivariateSpline(rhop_Te, Te, k=3)
    zero = 1.e-30
    indices = np.where(Fe <= zero)
    # print("indices", indices)
    for i in range(len(indices[0])):
        irhop = indices[0][i]
        iu = indices[1][i]
        ipitch = indices[2][i]
        val = Juettner1D(u[iu], np.abs(Te_spl(rhop_LUKE[irhop])))
        Fe[irhop, iu, ipitch] = val
    print("Replaced a total of " + str(len(indices[0])) + " zero values.")
    Fe[Fe < zero] = zero
    return Fe



def check_distribution(rhop, u, pitch, Fe):
    good = True
    tolenrance = 1.e-6
    for i in range(len(rhop)):
        spl = RectBivariateSpline(u, pitch, np.log(Fe[i]))
        u_temp = np.zeros(len(pitch))
        Fe_grid = np.zeros([len(u), len(pitch)])
        for j in range(len(u)):
            u_temp[:] = u[j]
            Fe_grid[j] = spl(u_temp, pitch, dx=1, grid=False)
        if(np.any(Fe_grid * Fe[i] > tolenrance)):
            good = False
            print("Found faulty distribution at rhop = ", rhop[i])
            indices = np.where(Fe_grid * Fe[i] > tolenrance)
            print("Troublesome u", u[indices[0]])
            print("Troublesome pitch", pitch[indices[1]])
            print("df/du", (Fe_grid * Fe[i])[indices[0], indices[1]])
    return good

class RDiffRelax:
    def __init__(self, Te_0, a,b,c,d):
        # Copy of the RELAX radial diffusion model
        # Does not include the inward pinch and only does one flux surface at a time
        # and does not support rdiff
        self.a= a
        self.b=b
        self.c=c
        self.d=d
        self.u_th_0 = rel_thermal_beta(cnst.electron_mass * cnst.c**2 / (Te_0* cnst.e))
        self.u_th_0 /= np.sqrt(1.0 - self.u_th_0**2)
    
    def __call__(self,x):   
        if(self.c != 0.e0 and self.d != 0.e0):
            tot_vel_diff = self.c * np.exp(-(x / self.u_th_0 / self.d)**2)
        else:
            tot_vel_diff = 0.e0
        return tot_vel_diff + (self.a + self.b * (x / self.u_th_0)**2)
        
        
class DistWeightInt:
    def __init__(self, rdiff_relax):
        self.rdiff_relax = rdiff_relax
        self.eval_mode = "thermal"
        self.N_count = 0
        
    def set_distribution(self, Te=None, Spl=None):
        # Spline has to be the logarithm of distribution
        self.N_count = 0
        if(Spl is not None):
            self.Spl = Spl
            self.eval_mode = "Spl"
        else:
            self.Te = Te
            self.eval_mode = "thermal"
    
    def eval_int(self, ull, uxx):
        self.N_count += 1
        if(self.eval_mode == "Spl"):
            return( 2.0 * np.pi * np.exp(self.Spl(ull,uxx)) * self.rdiff_relax(np.sqrt(ull**2 + uxx**2))* uxx)
        else:
            return( 2.0 * np.pi * Juettner2D(ull, uxx, self.Te) * self.rdiff_relax(np.sqrt(ull**2 + uxx**2)) * uxx)
            
    def make_integral(self, ull_range, uxx_range, points):
        return nquad(self.eval_int, [[ull_range[0], ull_range[1]], [uxx_range[0], uxx_range[1]]], \
                   opts=[{"epsabs":1.e-4, "epsrel":1.e-4, "points":points[0]},\
                         {"epsabs":1.e-4, "epsrel":1.e-4, "points":points[1]}])[0]
            
            

