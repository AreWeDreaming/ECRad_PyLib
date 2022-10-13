'''
Created on Jun 19, 2019

@author: sdenk
'''
# Collection of various distribution functions in a variety of spaces

import numpy as np

import scipy.constants as cnst
from scipy.special import kve, erf
from scipy.integrate import nquad
from scipy.interpolate import InterpolatedUnivariateSpline
 
def Coloumb_log(Te, ne):
    omega_p = np.sqrt(ne * cnst.e ** 2 / (cnst.m_e * cnst.epsilon_0))
    theta_min = cnst.hbar * omega_p / (Te * cnst.e)
    Clog = -np.log(np.tan(theta_min / 2.0))
    return Clog

def relax_time(E_kin, Te, ne):
    # Supposedly computes the relaxation time of an electron with E_kin for Te, ne
    # No idea if it works
    v = np.sqrt((1 + E_kin / (cnst.m_e * cnst.c ** 2 / cnst.e)) ** 2 - 1) * cnst.c
    print(v)
    mu = cnst.m_e * cnst.c ** 2 / (2.0 * Te * cnst.e)
    lb = 1.e0 / (rel_thermal_beta(mu) * cnst.c)
    print(lb)
    cLn = 23.e0 - np.log(ne ** 0.5 / Te ** (3.0 / 2.0))
    print(cLn)
    AD = 8.e0 * np.pi * cnst.e ** 8 * ne * cLn / cnst.m_e ** 2
    print(AD)
    x = lb * v
    print(x)
    G = 0.463 * x ** (-1.957)
    print(G)
    tau = v / (2.e0 * AD * lb ** 2 * G)
    print(tau)

def rel_thermal_beta(mu):
    # computes thermal velocity/c for mu
    return np.sqrt(1.e0 - (kve(1, mu) / kve(2, mu) + 3.e0 / mu) ** (-2))

def Gauss_norm(x, beta):
    return np.sqrt(1.0 / (np.pi * beta[2] ** 2)) * beta[0] * np.exp(-((x - beta[1]) / beta[2]) ** 2)

def Gauss_not_norm(x, beta):
    return beta[0] * np.exp(-((x - beta[1]) / beta[2]) ** 2)

def Juettner1D(u, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / (abs(Te) * cnst.e)
    a = np.sqrt(2.0 * mu / np.pi) * (1.0 / (1 + 105 / (128 * mu ** 2) + 15 / (8 * mu)))
    gamma = np.sqrt(1 + u ** 2)
    return gamma * u * a * mu * \
            np.exp(mu * (1 - gamma))

def Juettner1D_beta(beta, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / (abs(Te) * cnst.e)
    a = np.sqrt(2.0 * mu / np.pi) * (1.0 / (1 + 105 / (128 * mu ** 2) + 15 / (8 * mu)))
    gamma = 1.0 / np.sqrt(1 - beta ** 2)
    return gamma ** 2 * beta * a * mu * \
            np.exp(mu * (1 - gamma))

def make_f(Te, uxx, ull):
    mu = cnst.physical_constants["electron mass"][0] * \
         cnst.c ** 2 / (abs(Te) * cnst.e)
    a = 1.0 / (1. + 105. / (128. * mu ** 2) + 15. / (8. * mu))
    # print(a)
    gamma = np.sqrt(1 + uxx ** 2 + ull ** 2)
    return a * np.sqrt(mu / (2 * np.pi)) ** 3 * \
            np.exp(mu * (1 - gamma))

def make_f_1D(Te, u):
    # f = []
    mu = cnst.physical_constants["electron mass"][0] * \
         cnst.c ** 2 / (abs(Te) * cnst.e)
    a = 1.0 / (1. + 105. / (128. * mu ** 2) + 15. / (8. * mu))
    # print(a)
    gamma = np.sqrt(1 + u ** 2)
    return a * np.sqrt(mu / (2 * np.pi)) ** 3 * \
            np.exp(mu * (1 - gamma))

def MJ_approx(Te, u):
    mu = cnst.physical_constants["electron mass"][0] * \
         cnst.c ** 2 / (abs(Te) * cnst.e)
    gamma = np.sqrt(1 + u ** 2)
    return np.sqrt(mu ** 3 / (2 * np.pi) ** 3) * np.exp(-mu * u ** 2 / (1.e0 + gamma))

def Maxwell1D(Te, u):
    mu = cnst.physical_constants["electron mass"][0] * \
         cnst.c ** 2 / (abs(Te) * cnst.e)
    gamma = np.sqrt(1 + u ** 2)
    beta = u / gamma
    return beta * np.sqrt(2.e0 / np.pi) * \
           np.sqrt(mu ** 3) * np.exp(-mu / 2.0 * (beta ** 2))

def Maxwell1D_beta(beta, Te):
    mu = cnst.physical_constants["electron mass"][0] * \
         cnst.c ** 2 / (abs(Te) * cnst.e)
    return beta * np.sqrt(2.e0 / np.pi) * \
           np.sqrt(mu ** 3) * np.exp(-mu / 2.0 * (beta ** 2))

def make_f_beta(Te, betaxx, betall):
    mu = cnst.physical_constants["electron mass"][0] * \
         cnst.c ** 2 / (abs(Te) * cnst.e)
    a = 1.0 / (1. + 105. / (128. * mu ** 2) + 15. / (8. * mu))
    # print(a)
    gamma = 1.0 / np.sqrt(1.0 - betaxx ** 2 - betall ** 2)
    return a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
            np.exp(mu * (1.0 - gamma)) * gamma ** 5

def Juettner2D_cycl(u, Te):
    # Spherical coordintes -> Different norm
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / (abs(Te) * cnst.e)
    a = 1.0 / (1 + 105 / (128 * mu ** 2) + 15 / (8 * mu))
    gamma = np.sqrt(1 + u ** 2)
    return a * gamma * u * mu * np.exp(mu * (1 - gamma)) / 2.0

def RunAway2D(u_par, u_perp, Te, ne, nr, Zeff, E_E_c):
    lnLambda = 14.9 - 0.5 * np.log(ne / 1e20) + np.log(Te)
    # tau = 1.e0 / ( 4.e0 * np.pi, cnst.e)
    alpha = (E_E_c - 1.e0) / (Zeff + 1)
    cZ = np.sqrt(3 * (Zeff + 5.e0) / np.pi) * lnLambda
    f = alpha / (2.e0 * np.pi * cZ * u_par) * \
       np.exp(-u_par / (cZ * lnLambda) - 0.5 * alpha * u_perp ** 2 / u_par)
    if(f < 0):
        f = 0.e0
    return f

def Juettner2D(u_par, u_perp, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
    gamma = np.sqrt(1.0 + u_par ** 2 + u_perp ** 2)
    return a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
            np.exp(mu * (1.0 - gamma))

def Juettner2D_beta(beta_par, beta_perp, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
    gamma = 1.e0 / np.sqrt(1.0 - beta_par ** 2 - beta_perp ** 2)
    return a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
            np.exp(mu * (1.0 - gamma)) * gamma ** 5

def multi_slope_not_norm(u_perp, u_par, mu, mu_slope, gamma_switch, norm):
    gamma = np.sqrt(1 + u_par ** 2 + u_perp ** 2)
    if(np.isscalar(gamma)):
        if(gamma > gamma_switch):
            exp = np.exp(mu_slope * \
                    (1.0 - gamma)) * norm
        else:
            exp = np.exp(mu * (1.0 - gamma))
    else:
        exp = np.exp(mu * (1.0 - gamma))
        exp[gamma > gamma_switch] = np.exp(mu_slope * \
                    (1.0 - gamma[gamma > gamma_switch])) * norm
    return exp

def multi_slope_not_norm_w_jac(u_perp, u_par, mu, mu_slope, gamma_switch, norm):
    return multi_slope_not_norm(u_perp, u_par, mu, mu_slope, gamma_switch, norm) * 2.e0 * np.pi * u_perp

def multi_slope(u_par, u_perp, Te, gamma_switch, Te_slope):
    print("MultiSlope Te, Te_slope, gamma_switch", Te, Te_slope, gamma_switch)
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    mu_slope = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te_slope) * cnst.e)
    if(abs(Te_slope) * abs(Te) == 0):
        print(abs(Te_slope), abs(Te))
        raise IOError
    norm = np.exp(mu * (1.0 - gamma_switch)) / np.exp(mu_slope * (1.0 - gamma_switch))
    gamma_max = (10.0 / mu_slope) + 1.e0
    u_max = np.sqrt(gamma_max ** 2 - 1.e0)
    args = [mu, mu_slope, gamma_switch, norm]
    normalization = 1.0 / nquad(multi_slope_not_norm_w_jac, [[0, u_max], [-u_max, u_max]], \
                                args=args, opts={"epsabs":1.e-5, "epsrel":1.e-4})[0]
    return normalization * multi_slope_not_norm(u_perp, u_par, mu, mu_slope, gamma_switch, norm)

def multi_slope_simpl(u_par, u_perp, Te, gamma_switch, Te_slope):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    mu_slope = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te_slope) * cnst.e)
    if(abs(Te_slope) * abs(Te) == 0):
        print(abs(Te_slope), abs(Te))
    a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
    gamma = np.sqrt(1 + u_par ** 2 + u_perp ** 2)
    norm = np.exp(mu * (1.0 - gamma_switch)) / np.exp(mu_slope * (1.0 - gamma_switch))
    if(gamma > gamma_switch and Te_slope > Te):
        exp = np.exp(mu_slope * (1.0 - gamma)) * norm
    else:
        exp = np.exp(mu * (1.0 - gamma))
    return a * np.sqrt(mu / (2 * np.pi)) ** 3 * \
            exp

def multi_slope_cyl_beta_not_norm(beta, mu, mu_slope, gamma_switch, norm):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
    if(np.isscalar(gamma)):
        if(gamma > gamma_switch):
            exp = np.exp(mu_slope * \
                    (1.0 - gamma)) * norm
        else:
            exp = np.exp(mu * (1.0 - gamma))
    else:
        exp = np.exp(mu * (1.0 - gamma))
        exp[gamma > gamma_switch] = np.exp(mu_slope * \
                    (1.0 - gamma[gamma > gamma_switch])) * norm
    return a * np.sqrt(mu / (2 * np.pi)) ** 3 * exp * gamma ** 2 * beta  # Not exactly normalized but this brings the result closer to 1

def multi_slope_cyl_beta_not_norm_w_jac(beta, mu, mu_slope, gamma_switch, norm):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    return multi_slope_cyl_beta_not_norm(beta, mu, mu_slope, gamma_switch, norm) * 2.e0 * np.pi * beta ** 2 * 1.0 / gamma ** 5


def multi_slope_cyl_beta(beta, Te, gamma_switch, Te_slope):
    print("MultiSlope Te, Te_slope, gamma_switch", Te, Te_slope, gamma_switch)
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    mu_slope = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te_slope) * cnst.e)
    if(abs(Te_slope) * abs(Te) == 0):
        print(abs(Te_slope), abs(Te))
        raise IOError
    norm = np.exp(mu * (1.0 - gamma_switch)) / np.exp(mu_slope * (1.0 - gamma_switch))
    gamma_max = (10.0 / min(mu, mu_slope)) + 1.e0
    beta_max = np.sqrt(1.0 - 1.0 / gamma_max ** 2)
    args = [mu, mu_slope, gamma_switch, norm]
    normalization = 1.0 / nquad(multi_slope_cyl_beta_not_norm_w_jac, [[0, beta_max]], \
                                args=args, opts={"epsabs":1.e-5, "epsrel":1.e-4})[0]
    return normalization * multi_slope_cyl_beta_not_norm(beta, mu, mu_slope, gamma_switch, norm)

def Maxwell2D(u_par, u_perp, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    gamma = np.sqrt(1.0 + u_par ** 2 + u_perp ** 2)
    beta_par = u_par / gamma
    beta_perp = u_perp / gamma
    return np.sqrt(mu / (2 * np.pi)) ** 3 * \
            np.exp(-mu / 2.0 * (beta_par ** 2 + beta_perp ** 2))

def BiMaxwell2DV(beta_par, beta_perp, Te_par, Te_perp):
    mu_par = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te_par) * cnst.e)
    mu_perp = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te_perp) * cnst.e)
    return np.sqrt(mu_par * mu_perp ** 2 / (2 * np.pi) ** 3) * \
            np.exp((-mu_par / 2.0 * beta_par ** 2 - mu_perp / 2.0 * beta_perp ** 2))

def BiMaxwellJuettner2DV(beta_par, beta_perp, Te_par, Te_perp):
    T0 = Te_par ** (1.0e0 / 3.0e0) * Te_perp ** (2.0e0 / 3.0e0)
    mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
    r = T0 / Te_par
    s = T0 / Te_perp
    gamma = 1.0 / np.sqrt(1.0 - beta_par ** 2 - beta_perp ** 2)
    u_par = beta_par * gamma
    u_perp = beta_perp * gamma
    gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
    a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
    return a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * np.exp(mu * (1 - gamma_drift_m))

def Maxwell2D_beta(beta_par , beta_perp, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    return np.sqrt(mu / (2 * np.pi)) ** 3 * \
            np.exp(-mu / 2.0 * (beta_par ** 2 + beta_perp ** 2))

def Juettner2D_drift(u_par, u_perp, Te, u_par_drift, u_perp_drift):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    a = 1.0 / (np.exp(-u_par_drift ** 2 * mu) + np.sqrt(np.pi * u_par_drift * \
        (1 + erf(u_par_drift * np.sqrt(mu)))))
    return a * np.sqrt(mu / (2 * np.pi)) ** 3 * \
            np.exp(-mu * ((u_perp - u_perp_drift) ** 2 + (u_par - u_par_drift) ** 2))

def Juettner2D_bidrift(u_par, u_perp, Te_par, Te_perp, u_par_drift, u_perp_drift):
    T0 = Te_par ** (1.0e0 / 3.0e0) * Te_perp ** (2.0e0 / 3.0e0)
    mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
    r = T0 / Te_par
    s = T0 / Te_perp
    gamma_drift_m = np.sqrt(1.0e0 + r * (u_par - u_par_drift) ** 2 + s * u_perp ** 2)
    a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
    return a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * np.exp(mu * (1 - gamma_drift_m))

def BiJuettner(ull, uxx, Te, beta):
    return (1.e0 - beta[0]) * (Juettner2D(ull, uxx, Te) + \
                                    beta[0] * Juettner2D_bidrift(ull, \
                                    uxx, beta[3], beta[4], beta[5], beta[6]))

def g2_precise(alpha, beta, u):
    # Use 0.2 as lower boundary to avoid divergence
    u_int = np.linspace(0.01, np.max(u), 120)
    gamma_int = np.sqrt(1.0 + u_int ** 2)
    g2_int = u_int ** 4 / gamma_int ** 2 * (u_int / (gamma_int + 1.0)) ** beta
    g2_spl = InterpolatedUnivariateSpline(u_int, g2_int)
    gamma = np.sqrt(1.0 + u ** 2)
#    plt.plot(u_int, g2_int)
#    plt.show()
    if(np.isscalar(u)):
        g2 = alpha * ((gamma + 1.e0) / u) ** beta * g2_spl.integral(0.01, u)
    else:
        g2 = np.zeros(len(u))
        for i in range(len(u)):
            g2[i] = alpha * ((gamma[i] + 1.e0) / u[i]) ** beta * g2_spl.integral(0.01, u[i])
    return g2

def SynchrotonDistribution(u, zeta, Te, ne, B, Z_eff=1.0):
    lambda_C = Coloumb_log(Te, ne)
    mu = cnst.m_e * cnst.c ** 2 / (Te * cnst.e)
    epsilon = 1.0 / mu
    tau = 4.0 * np.pi * cnst.epsilon_0 ** 2 * cnst.m_e ** 2 * cnst.c ** 3 / (ne * cnst.e ** 4 * lambda_C)
    tau_r = 6.0 * np.pi * cnst.epsilon_0 * (cnst.m_e * cnst.c) ** 3 / (cnst.e ** 4 * B ** 2)
    alpha = 2.0 * tau / (3.0 * tau_r * epsilon)
    print("alpha", alpha)
    beta = 3.0 * (Z_eff + 1.0)
    g2 = g2_precise(alpha, beta, u)
    g0 = g0_approx(alpha, u)
    if(np.isscalar(g2) and not np.isscalar(zeta)):
        f = np.zeros(zeta.shape)
    elif(not np.isscalar(g2) and np.isscalar(zeta)):
        f = np.zeros(g2.shape)
    else:
        print("Matrix evaluation not yet supported - supply either scalar u or scalar zeta")
#    print("WAAAAARNING g2 not included!!!!!!!")
    f += g2
    f *= 3.0 * (zeta ** 2 - 1.0) / 2.0
    f += g0
    return g0, g2, f


def g2_approx(alpha, u):
    gamma = np.sqrt(1.0 + u ** 2)
    return alpha * ((gamma + 1.0) / u) ** 6 * (32.0 / u * (gamma - 1.0) + 17.0 * u + u ** 3 / 3.0 - 3.0 * u * gamma - 29.0 * np.arcsinh(u) - np.arctan(u))

def g0_approx(alpha, u):
    return -alpha * (np.arctan(u) - u + u ** 3 / 3.0)

def SynchrotonDistribution_approx(u, zeta, Te, ne, B):
    lambda_C = Coloumb_log(Te, ne)
    mu = cnst.m_e * cnst.c ** 2 / (Te * cnst.e)
    epsilon = 1.0 / mu
    tau = 4.0 * np.pi * cnst.epsilon_0 ** 2 * cnst.m_e ** 2 * cnst.c ** 3 / (ne * cnst.e ** 4 * lambda_C)
    tau_r = 6.0 * np.pi * cnst.epsilon_0 * (cnst.m_e * cnst.c) ** 3 / (cnst.e ** 4 * B ** 2)
    alpha = 2.0 * tau / (3.0 * tau_r * epsilon)
    print(alpha)
    g0 = g0_approx(alpha, u)
    g2 = g2_approx(alpha, u)
    if(np.isscalar(g2) and not np.isscalar(zeta)):
        f = np.zeros(zeta.shape)
    elif(not np.isscalar(g2) and np.isscalar(zeta)):
        f = np.zeros(g2.shape)
    else:
        print("Matrox evaluation not yet supported - supply either scalar u or scalar zeta")
    f += g2
    f *= 3.0 * (zeta ** 2 - 1.0) / 2.0
    f += g0
    return g0, g2, f


if(__name__== "__main__"):
    from Plotting_Configuration import plt
    u_par = np.linspace(0,0.5,200)
    u_perp = 0.0
    Te = 6.e3
    plt.plot(u_par, Juettner2D(u_par, u_perp, Te)/Maxwell2D(u_par, u_perp, Te), "-")
    plt.show()
    