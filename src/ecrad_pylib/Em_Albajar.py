'''
Created on Mar 22, 2016

@author: Severin Denk
'''
stand_alone = False
import numpy as np
import scipy.constants as cnst
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, newton
from scipy.special import kve
try:
    from scipy.special import loggamma as log_gm_fun, wofz
except ImportError:
    print("Failed to import loggamma, log_gm_fun and woft some absorption routines might not work")
    print("If you are an ECRad_GUI user this error should not cause any issues!")
if(not stand_alone):
    from Plotting_Configuration import plt
mass_e = cnst.m_e
e0 = cnst.e
c0 = cnst.c
# Only use false
warm_plasma = False
eps0 = cnst.epsilon_0
dstf_comp = "Th"

# Implementation of the Albajar absorption coefficient analog to the Fortran implementation in ECRad
# Verified with Solve_rad_transp up to 5th running digit


def N_with_pol_vec(X, Y, sin_theta, cos_theta, mode):
# Calculates the cold refractive index and the polarization vector
# For the coordinate system the Stix-reference frame is used. I.e. the wave propagates in the e_x e_z plane and the magnetic field is aligned with e_z.
# This routine assumes a phase for which the e_y-component is purely imaginary and positive for X-mode and purely imaginary and negative for O-mode.
        e = np.zeros(3, dtype=np.complex)
        if(X >= 1.e0):
            N = 0.e0
            return N, e
        rho = Y ** 2 * sin_theta ** 4 + 4.e0 * (1.e0 - X) ** 2 * cos_theta ** 2
        if(rho < 0.e0):
            N = 0.e0
            return N, e
        rho = np.sqrt(rho)
        f = (2.e0 * (1.e0 - X)) / (2.e0 * (1.e0 - X) - Y ** 2 * sin_theta ** 2 - float(mode) * Y * rho)
        N = 1.e0 - X * f
        if(N < 0.e0):
            N = 0.e0
            return N, e
        N = np.sqrt(N)
        if(cos_theta ** 2 < 1.e-5 or 1.0 - sin_theta ** 2 < 1.e-5):  # For quasiperendicular propagation the calculation of e_y
                                                                    # is numerically unstable, hence we use the quasi-erpendicular
                                                                    # approximation for this case
#            print("Quasi-perp")
            if(mode > 0):  # X-mode
                e[1] = np.complex(0.0, np.sqrt(1.0 / N))
                e[0] = np.complex(0.0, 1.0 / Y * (1.0 - (1.0 - Y ** 2) * f)) * e[1]
                # e_z zero for quasi-perpendicular X-mode
            else:
                e[2] = 1.0 / np.sqrt(N)  #
                # e_x, e_y zero for quasi-perpendicular O-mode
        else:
            a_sq = sin_theta ** 2 * (1.e0 + (((1.e0 - X) * N ** 2 * cos_theta ** 2) / \
              (1.e0 - X - N ** 2 * sin_theta ** 2) ** 2) * \
              1.e0 / Y ** 2 * (1.e0 - (1.e0 - Y ** 2) * f) ** 2) ** 2
            b_sq = cos_theta ** 2 * (1.e0 + ((1.e0 - X) / \
              (1.e0 - X - N ** 2 * sin_theta ** 2)) * \
              1.e0 / Y ** 2 * (1.e0 - (1.e0 - Y ** 2) * f) ** 2) ** 2
            e[1] = np.complex(0.0, np.sqrt(1.0 / (N * np.sqrt(a_sq + b_sq))))
            if(mode < 0):
                e[1] *= -1.0
            e[0] = np.complex(0.0, 1.0 / Y * (1.0 - (1.0 - Y ** 2) * f)) * e[1]
            e[2] = np.complex(-(N ** 2 * sin_theta * cos_theta) / (1.0 - X - N ** 2 * sin_theta ** 2), 0.0) * e[0]
        return N, e

def rotate_vec_around_axis(vec, axis, theta):
    """
    Rotate a vector counterclockwise rotation about
    the given axis by theta radians. Source: https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    This routine has been double checked.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    if(np.abs(theta) < 1.e-5):
        # There are numerical issues for very small rotation
        return vec
    else:
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        return np.dot(rotation_matrix, vec)


class Bornatici_abs:
#     DO NOT USE THIS ROUTINE WITHOUT PROPER DEBUGGING AND BENCHMARKING!!!
#     This class calculates the absorption coefficient and optical depth for X-mode and harmonics n >= 2
#     There are two modes: "quasi-perpendicular" and "oblique" which refers to the propagation direction of the wave
#     Which method is used is determined by N_c cos(theta) > v_t/c
#     In its current state this routine is broken and does not deliver ANY trustworthy results
#     DO NOT USE THIS ROUTINE WITHOUT PROPER DEBUGGING AND BENCHMARKING!!!

    # DO NOT USE THESE ROUTINES BEFORE FIXING THEM
    # The following routines calculate the absorption coefficient according to 3.1.35 and the optical depth according to table IV of Bornatici's review paper
    # Equation numbers below indicate which equation was used
    # This routine compares very poorly to the other absorption routine in this module
    # There is at least one bug somewhere!!!!
    # Most likely candidate are the Dnestrovskii functions
    # DO NOT USE THESE ROUTINES BEFORE FIXING THEM

    def __init__(self):
        pass

    def Dnestrovskii_recursion(self, z, q):
        # THIS ROUTINE DOES NOT WORK
        # However, this routine is probably the one that can be most easily fixed!
        # It uses the relation between the Dnestrovskii functions and the plasma dispersion function.
        # The plasma dispersion function in turn can be related to the Faddeeva function, for which a highly accurate implementation exists (wofz)
        # This implmentation uses the recurrence relation given in
        # F_(q + 3/2)(z) = 1 / (q + 1/2) (1 - z F_(q+1/2)(z))
        # and the relation between the plasma dispersion function and F_3/2
        # which yields for negative real arguments z:
        # F_(3/2)(z) = 2 (1 - i Sqrt(pi (-z)Exp(z)Erfc(-(sqrt(z))
        # THIS ROUTINE DOES NOT WORK
        if(q < 1.5 or np.abs(np.round(np.imag(z), 2)) > 0 or
           np.real(z) >= 0 or np.abs(np.round(q * 2.0) - q * 2.0) > 1.e-5):
            print("This implementation of the Dnestrovskii function only holds for real negative arguments and z and q = n/2, n >= 3")
            print("q", q, "z", z)
            raise ValueError
        else:
            nu_max = np.round(q , 1)
            x = -np.real(z)
            F = np.complex(0.0, 1.0) * np.sqrt(np.pi / x) * wofz(np.sqrt(x))
            for nu in range(3, int(np.round(nu_max * 2)) + 1, 2):
                F = (1.0 - z * F) / (float(nu) / 2.0 - 1.0)
            return np.conjugate(F)  # If we use the conjugate here the result matches the figures 2 (a) and (b)  in Bornatici -- WHY?

    def test_Dnestrovskii_recursion(self, ax1, ax2):
        # Plots imaginary and real part of the sharosvsky function
        # Note that positive arguments are not allowed
        z_ar = np.linspace(-40, -1.e-8, 500)
        F_ar = []
        for z in z_ar:
            F = self.Dnestrovskii_recursion(z, 3.5)
            F_ar.append(F)
        ax1.plot(z_ar, np.real(F_ar))
        ax2.plot(z_ar, -np.imag(F_ar))
#        plt.show()

    def Dnestrovskii_serios(self, z, q):
        # This implmentation uses the series 2.3.70 and 2.3.71 - the series does not seem to be converging
        # THIS ROUTINE DOES NOT WORK
        nu = np.linspace(0, 100, 101, dtype=np.int)
        if(np.real(np.sqrt(np.complex(z))) == 0.0):
            sigma = 1.0
        elif(np.real(np.sqrt(np.complex(z))) <= 0.0):
            sigma = 2.0
        else:
            sigma = 0.0
#        if(np.abs(z) > 40):
#            F = np.complex(0.0, 0.0)
#        elif(np.abs(z) > 15.0):
        F_vals_a = np.exp(-np.log(-z) * (nu + 1) + log_gm_fun(q + nu) - log_gm_fun(q))
#        plt.plot(-np.log(-z) * (nu + 1) + log_gm_fun(q + nu) - log_gm_fun(q))
#        plt.show()
        Fa = -np.sum(F_vals_a) - np.complex(0.0, 1.0) * sigma * np.pi * np.exp(-log_gm_fun(q) + np.log(-z) * (q - 1.0) + z)
#        else:
        F_vals_b = np.real(np.exp(np.log(-z) * nu - log_gm_fun(q) + log_gm_fun(q - 1 - nu)))
#            plt.plot(F_vals)
#            plt.show()
            # (-z) ** nu[q - 1.0 - nu < 0] * factorial(np.round(-(q - 1.0 - nu[q - 1.0 - nu <= 0]) - 0.5)) * \
            #                            (-4) ** np.round(q - 1.0 - nu[q - 1.0 - nu <= 0] + 0.5) * \
            #                            np.sqrt(np.pi) / factorial(2 * np.round(-(q - 1.0 - nu[q - 1.0 - nu <= 0]) - 0.5))
            # F_vals = (-z) ** nu * gm_fun(q - 1.0 - nu) / gm_fun(q)
        Fb = np.sum(F_vals_b) - np.complex(0.0, 1.0) * np.pi * np.exp(-log_gm_fun(q) + np.log(-z) * (q - 1.0) + z)
        return Fa, Fb


    def test_Dnestrovskii_series(self, ax1, ax2):
        # Plots imaginary and real part of the sharosvsky function
        # Note that positive arguments are not allowed
        z_ar = np.linspace(-40, -4.e-1, 500)
        Fa_ar = []
        Fb_ar = []
        for z in z_ar:
            Fa, Fb = self.Dnestrovskii_series(z, 3.5)
            Fa_ar.append(Fa)
            Fb_ar.append(Fb)
        ax1.plot(z_ar, np.real(Fa_ar))
        ax1.plot(z_ar, np.real(Fb_ar), "--")
        ax2.plot(z_ar, np.imag(Fa_ar))
        ax2.plot(z_ar, np.imag(Fb_ar), "--")

    def Dnestrovskii(self, z, q):
        # This implmentation uses 2.3.69
        # THIS ROUTINE DOES NOT WORK
        nu_max = np.round(q - 1.5)
        nu = np.linspace(0, nu_max, nu_max + 1)
        x = -np.real(z)
        F_vals = np.real(np.exp(np.log(x) * nu - log_gm_fun(q) + log_gm_fun(q - 1 - nu)))
        F = np.sum(F_vals) + np.complex(0.0, np.pi * np.real(np.exp(-log_gm_fun(q) + np.log(x) * (nu_max)))) * np.sqrt(x) * wofz(np.sqrt(x))
        return np.conjugate(F)  # If we use the conjugate here the result matches the figure 2 (a) and (b) in Bornatici -- WHY

    def test_Dnestrovskii(self, ax1, ax2):
        # Plots imaginary and real part of the sharosvsky function
        # Note that positive arguments are not allowed
        z_ar = np.linspace(-40, -1.e-8, 500)
        F_ar = []
        for z in z_ar:
            F = self.Dnestrovskii(z, 3.5)
            F_ar.append(F)
        ax1.plot(z_ar, np.real(F_ar))
        ax2.plot(z_ar, -np.imag(F_ar))


    def An_2(self, z, Y, X, N_abs):
        # Quasi-perpendicular propagation
        # See abs_bornatici
        F = self.Dnestrovskii(z, 3.5)
        omega_ratio = X / Y ** 2
        a_approx = -0.5 * omega_ratio * (1.0 - Y ** 2) / (1.0 - Y ** 2 - X) * F  # (3.1.19a)
        b_approx = -2.0 * (1.0 - X / (1.0 - Y)) * a_approx  # (3.1.18b)
        # 3.1.1.17 we disregard the branch with "-", whether this is a valid approach needs to be verified
        N_perp_sq = (-(1.0 + b_approx) + np.sqrt(1.0 + b_approx ** 2 + 4 * a_approx * 4.0 * N_abs)) / (2.0 * a_approx)
        # Now we calculate the corrected a and b
        a_2 = 0.5 * omega_ratio * (1.0 + 3.0 * N_perp_sq * F) / (3.0 - omega_ratio * (1.0 + 1.5 * N_perp_sq * F))  # 3.1.20
        b = np.abs(1 + 0.5 * omega_ratio * (1.0 + a_2) ** 2 * np.real(F))  # 3.1.38
        b = 1.0 / b
        N_perp_prime = N_abs ** 2 * (1.0 - np.real(b_approx + a_2 * N_abs ** 2))  # (3.1.19a)
        A = N_perp_prime * np.abs(1.0 + a_2) ** 2 * b  # 3.1.37
        return A, F

    def integrand_mean_An_2(self, z, args):  # p. 1191 argument of integral at the bottom right
        # See abs_bornatici
        X = args[0]
        Y = args[1]
        N_abs = args[2]
        A, F = self.An_2(z, X, Y, N_abs)
        A *= -np.imag(F)
        # print(A)
        return A


    def An_bornatici(self, omega, svec, n, mean=False):
        # See abs_bornatici
        omega_p = cnst.e * np.sqrt(svec.ne / (eps0 * cnst.m_e))
        omega_c = svec.freq_2X * np.pi
        X = omega_p ** 2 / omega ** 2
        Y = omega_c / omega
        N_abs = N_with_pol_vec(X, Y, 1.0, 0.0, 1)[0]
        mu = cnst.m_e * cnst.c ** 2 / (svec.Te * cnst.e)
        beta_t = np.sqrt(1.e0 - (kve(1, mu) / kve(2, mu) + 3.e0 / mu) ** (-2))
        if(n == 2 and mean):
            An = 1.0 / np.pi * quad(self.integrand_mean_An_2, -100.0, -1.e-8, args=[X, Y, N_abs])[0]
        elif(n == 2):
            z = 1.0 / beta_t ** 2 * (1.0 - n * Y)
            if(z > 0):
                An = 0
            else:
                An = self.An_2(z, X, Y, N_abs)[0]
        else:
            a_n = X * Y / (1.0 - Y ** 2 - X ** 2)  # 3.1.14a
            An = N_abs ** (2 * n - 3) * np.abs(1.0 + a_n ** 2)  # 3.1.37 and 3.1.12
        return An, beta_t

    def mu_n(self, X, Y, cos_theta, sin_theta, n):
        mode = 1  # Only X mode
        if(X >= 1.e0):
            return 0.0
        rho = Y ** 2 * sin_theta ** 4 + 4.e0 * (1.e0 - X) ** 2 * cos_theta ** 2
        if(rho < 0.e0):
            return 0.0
        rho = np.sqrt(rho)
        f = (2.e0 * (1.e0 - X)) / (2.e0 * (1.e0 - X) - Y ** 2 * sin_theta ** 2 - float(mode) * Y * rho)
        N = 1.e0 - X * f
        if(N < 0.e0):
            N = 0.e0
            f = 0.e0
            return 0.0
        N = np.sqrt(N)
        a_sq = sin_theta ** 2 * (1.e0 + (((1.e0 - X) * N ** 2 * cos_theta ** 2) / \
          (1.e0 - X - N ** 2 * sin_theta ** 2) ** 2) * \
          1.e0 / Y ** 2 * (1.e0 - (1.e0 - Y ** 2) * f) ** 2) ** 2
        b_sq = cos_theta ** 2 * (1.e0 + ((1.e0 - X) / \
          (1.e0 - X - N ** 2 * sin_theta ** 2)) * \
           1.e0 / Y ** 2 * (1.e0 - (1.e0 - Y ** 2) * f) ** 2) ** 2
        mu_n = N ** (2 * n - 3) * (n - 1) ** 2 * (1.0 - (n + 1) / n * f) ** 2
        mu_n /= (1.0 + cos_theta ** 2) * np.sqrt(a_sq ** 2 + b_sq ** 2)
        return mu_n

    def abs_bornatici(self, omega, svec, n_min, n_max):
        # DOES NOT WORK - DO NOT USE
        # This routine and the three routines above only hold for X-mode and propagation perpendicular to the magnetic field
        # This routine evaluates 3.1.35
        if(n_min == 1):
            # First harmonic needs to be treated separetly --> not implemented
            return
        abs_coeff = 0
        omega_p = cnst.e * np.sqrt(svec.ne / (eps0 * cnst.m_e))
        omega_c = svec.freq_2X * np.pi
        omega_ratio = omega_p ** 2 / (omega_c) ** 2
        X = omega_p ** 2 / omega ** 2
        Y = omega_c / omega
        n_fac = 1
        N_abs = N_with_pol_vec(X, Y, np.sin(svec.theta), np.cos(svec.theta), 1)[0]
        mu = cnst.m_e * cnst.c ** 2 / (svec.Te * cnst.e)
        beta_t = np.sqrt(1.e0 - (kve(1, mu) / kve(2, mu) + 3.e0 / mu) ** (-2))
        for n in range(n_min, 3):
            n_fac *= (n - 1)
            if(N_abs * np.cos(svec.theta) < beta_t):  # Perpendicular propagation
                An, beta_t = self.An_bornatici(omega, svec, n)
                z = 1.0 / beta_t ** 2 * (omega - n * omega_c) / omega
                alpha_bornat_n = An * n ** (2 * (n - 1)) / (2 ** (n - 1) * n_fac) * \
                                 omega_ratio * beta_t ** (2 * (n - 2)) * omega_c / cnst.c * \
                                 (-np.imag(self.Dnestrovskii(z, n + 1.5)))
                abs_coeff += alpha_bornat_n
            else:  # Oblique
                mu_n = self.mu_n(self, X, Y, np.sin(svec.theta), np.cos(svec.theta), n)
                zeta_n = 1.0 / np.sqrt(2.0) * (1.0 - n * Y) / (beta_t * N_abs * np.cos(svec.theta))
                Phi_n = np.exp(-zeta_n ** 2)
                Phi_n /= np.sqrt(2.0 * np.pi) * n * omega_c * N_abs * np.abs(np.cos(svec.theta)) * beta_t
                alpha_bornat_n = np.pi * n ** (2 * (n - 1)) / (2 ** (n - 1) * n_fac) * omega_p ** 2 / cnst.c * \
                                  beta_t ** (2 * n - 1) * np.sin(svec.theta) ** (2 * (n - 1)) * (1.0 + np.cos(svec.theta) ** 2)
                alpha_bornat_n *= Phi_n * mu_n
                abs_coeff += alpha_bornat_n
        return abs_coeff

    def tau_bornatici(self, omega, svec, L_B, n_min, n_max):
        # DOES NOT WORK - DO NOT USE
        # Analytical solution optical depth tau for propagation perpendicular to the magnetic field
        tau_bornat = 0
        omega_p = cnst.e * np.sqrt(svec.ne / (eps0 * cnst.m_e))
        omega_c = svec.freq_2X * np.pi
        omega_ratio = omega_p ** 2 / (omega_c) ** 2
        lamda_O = cnst.c * 2.0 * np.pi / omega_c
        X = omega_p ** 2 / omega ** 2
        Y = omega_c / omega
        n_fac = 1
        N_abs = N_with_pol_vec(X, Y, np.sin(svec.theta), np.cos(svec.theta), 1)[0]
        mu = cnst.m_e * cnst.c ** 2 / (svec.Te * cnst.e)
        beta_t = np.sqrt(1.e0 - (kve(1, mu) / kve(2, mu) + 3.e0 / mu) ** (-2))
        for n in range(n_min, n_max + 1):
            n_fac *= (n - 1)
            if(N_abs * np.cos(svec.theta) < beta_t):
                An_mean, beta_t = self.An_bornatici(omega, svec, n, mean=True)
                tau_bornat_n = np.pi ** 2 * n ** (2 * (n - 1)) / (2 ** (n - 1) * n_fac) * \
                               An_mean * omega_ratio * beta_t ** (2 * (n - 1)) * L_B / lamda_O
                tau_bornat += tau_bornat_n
            else:
                mu_n = self.mu_n(X, Y, np.sin(svec.theta), np.cos(svec.theta), n)
                tau_bornat_n = np.pi ** 2 * n ** (2 * (n - 1)) / (2 ** (n - 1) * n_fac) * omega_ratio * beta_t ** (2 * n - 1) * \
                                 np.sin(svec.theta) ** (2 * (n - 1)) * (1.0 + np.cos(svec.theta) ** 2)
                tau_bornat_n *= mu_n * L_B / lamda_O
                tau_bornat += tau_bornat_n
        return tau_bornat


def get_abs(s, args):
    # Wrapper function for use in numerical integration
    abs_obj = args[0]
    svec = args[1]
    freq = args[2]
    R0 = args[3]
    nmax = args[4]
    if(np.isscalar(s)):
        svec.freq_2X = freq / (R0 + s) * R0
        c_abs = abs_obj.abs_Albajar(svec, freq * 2.0 * np.pi, 1, nmax)[0]
        return c_abs
    else:
        c_abs_arr = np.zeros(len(s))
        for i in range(len(s)):
            svec.freq_2X = freq / (R0 + s[i]) * R0
            c_abs  = abs_obj.abs_Albajar(svec, freq * 2.0 * np.pi, 1, nmax)[0]
            c_abs_arr[i] = c_abs
        return c_abs

def abs_Te():
    # Compares absorption coefficients
    abs_obj = EmAbsAlb()
    Te_arr = np.logspace(2, 5, 100)
    tau_slab_arr = []
    tau_plasma_arr = []
    tau_slab_TCV_arr = []
    tau_bornat = []
    tau_suttrop = []
    freq = 140.0e9
    freq_TCV = 86.e9  #
    ds = 0.02
    R0 = 1.65
    R0_TCV = 0.89  #
    s = np.linspace(-0.5 * ds, +0.5 * ds, 10)
    nmin = 2
    nmax = 3
    theta = 85.0 / 180.0 * np.pi
    svec = SVec(0.2, 1.e3, 1.5e19, freq, theta)
    bornat = Bornatici_abs()
    for Te in Te_arr:
        svec = SVec(0.2, Te, 1.5e19, freq, theta)
        if((Te > 20 and nmax < 4) or (Te > 50 and nmax < 5) or (Te > 80 and nmax < 6)):
            nmax += 1
        tau_slab_arr.append(quad(get_abs, s[0], s[-1], args=[abs_obj, svec, freq, R0, nmax], \
                            points=[0])[0])
        tau_bornat.append(bornat.tau_bornatici(freq * 2.0 * np.pi, svec, R0, nmin, nmax))
        tau_suttrop.append(3.9e-22 * svec.ne * svec.Te)
        print(Te, tau_bornat[-1])
    for Te in Te_arr:
        svec = SVec(0.2, Te, 1.5e19, freq_TCV, theta)
        if((Te > 20 and nmax < 4) or (Te > 50 and nmax < 5) or (Te > 80 and nmax < 6)):
            nmax += 1
        tau_slab_TCV_arr.append(quad(get_abs, s[0], s[-1], args=[abs_obj, svec, freq_TCV, R0_TCV, nmax], \
                            points=[0])[0])
        #        tau_bornat.append(tau_bornatici(freq * 2.0 * np.pi, svec, R0, nmin, nmax))
        print(Te, tau_slab_TCV_arr[-1])
    ds = 1.0
    R0 = 1.65
    s = np.linspace(-0.5 * ds, +0.5 * ds, 10)
    for Te in Te_arr:
        svec = SVec(0.2, Te, 5.e19, freq, theta)
        if((Te > 20 and nmax < 4) or (Te > 50 and nmax < 5) or (Te > 80 and nmax < 6)):
            nmax += 1
        tau_plasma_arr.append(quad(get_abs, s[0], s[-1], args=[abs_obj, svec, freq, R0, nmax], \
                            points=[0])[0])
        print(Te, tau_slab_arr[-1])
        tau_bornat.append(bornat.tau_bornatici(freq * 2.0 * np.pi, svec, R0, nmin, nmax))
#    plt.loglog(Te_arr / 1.e3, tau_slab_arr, label=r"$\tau_\omega[\SI{2}{\centi\metre}\mathrm{\,slab}]$ Albajar")
#    plt.loglog(Te_arr / 1.e3, tau_plasma_arr, label=r"$\tau_\omega[\SI{1}{\metre}\mathrm{\,plasma}]$ ASDEX Upgrade")
#    plt.loglog(Te_arr / 1.e3, tau_slab_TCV_arr, label=r"$\tau_\omega[\SI{2}{\centi\metre}\mathrm{\,slab}]$ TCV")
    plt.semilogx(Te_arr / 1.e3, tau_bornat, label=r"$\tau_\omega[\SI{2}{\centi\metre}\mathrm{\,slab}]$ Bornatici")
    plt.loglog(Te_arr / 1.e3, tau_suttrop, label=r"$\tau_\omega[\mathrm{entire\,plasma\,approximation}]$")
    plt.gca().set_xlabel(r"$T_\mathrm{e}$\, [\si{\kilo\electronvolt}]")
    plt.gca().set_ylabel(r"$\tau_{\omega, \mathrm{slab}}$")
    plt.legend()
    plt.show()


def cyl_to_pol(u_par, u_perp, zeta):
        # Chain rule for derivatives
        u = np.sqrt(u_par ** 2 + u_perp ** 2)
        if(zeta == 1.0):
            mu = u_par / u
        else:
            mu = np.sqrt((u_par ** 2 + u_perp ** 2 * (zeta - 1.0) / zeta)) / u * np.sign(u_par)
        if(np.isscalar(mu)):
            if(u == 0):
                mu = 0
        else:
            mu[u == 0] = 0.0
        pitch = np.arccos(mu)
        if(np.any(pitch != pitch)):
            print("Nan in coordinate transform!")
            print("u_par, u_perp, zeta,", u_par, u_perp, zeta)
            print("mu", mu)
            raise(ValueError)
#        plt.plot(u_par, u)
#        plt.plot(u_par, pitch)
#        plt.show()
        return u, pitch

class DistributionInterpolator:
# Used to interpolate electron momentum distribution given in momentum and pitch
    def __init__(self, u , pitch, spl=None, Te_perp=None, Te_par=None):
        self.u_min = np.min(u)
        self.u_max = np.max(u)
        self.pitch_min = np.min(pitch)
        self.pitch_max = np.max(pitch)
        self.spl = spl
        self.Te_perp = Te_perp
        self.Te_par = Te_par

    def eval_dist_cyl(self, u, pitch, Te=None, mode="spline"):
        # returns f
        if(mode == "spline"):
            return self.spl(u, pitch, grid=False)
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_cyl_du(self, vpar, magmom, mode="spline"):
        # returns df/du
        if(mode == "spline"):
            return self.spl(vpar, magmom, dx=1, grid=False)
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_cyl_dpitch(self, vpar, magmom, mode="spline"):
        # returns df/dmu
        if(mode == "spline"):
            return self.spl(vpar, magmom, dy=1, grid=False)
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist(self, u_perp, u_par, Te=None, mode="thermal", zeta=None):
        # returns f for given u_perp and u_par
        if(mode == "thermal"):
            mu = mass_e * c0 ** 2 / (e0 * Te)
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
            return a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(mode == "BM"):
            T0 = self.Te_par ** (1.0e0 / 3.0e0) * self.Te_perp ** (2.0e0 / 3.0e0)
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
            r = T0 / self.Te_par
            s = T0 / self.Te_perp
            gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
            a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
            return a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * np.exp(mu * (1 - gamma_drift_m))
        elif(mode == "spline"):
            if(zeta is None):
                raise ValueError("Zeta must be supplied in case of numerical distributions")
            u, pitch = cyl_to_pol(u_par, u_perp, zeta)
            return np.exp(self.spl(u, pitch, grid=False))
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_du_perp(self, u_perp, u_par, Te=None, mode="thermal"):
        # df/du_perp includes chain rule for spherical f
        if(mode == "thermal"):
            mu = mass_e * c0 ** 2 / (e0 * Te)
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
            return -a * mu * u_perp / gamma * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(mode == "BM"):
            T0 = self.Te_par ** (1.0e0 / 3.0e0) * self.Te_perp ** (2.0e0 / 3.0e0)
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
            r = T0 / self.Te_par
            s = T0 / self.Te_perp
            gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
            a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
            return -(mu * s * u_perp) / gamma_drift_m * a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * \
                   np.exp(mu * (1 - gamma_drift_m))
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_du_par(self, u_perp, u_par, Te=None, mode="thermal"):
        # df/du_par includes chain rule for spherical f
        if(mode == "thermal"):
            mu = mass_e * c0 ** 2 / (e0 * Te)
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
            return -a * mu * u_par / gamma * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(mode == "BM"):
            T0 = self.Te_par ** (1.0e0 / 3.0e0) * self.Te_perp ** (2.0e0 / 3.0e0)
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
            r = T0 / self.Te_par
            s = T0 / self.Te_perp
            gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
            a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
            return -(mu * r * u_par) / gamma_drift_m * a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * \
                   np.exp(mu * (1 - gamma_drift_m))
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

class GeneDistributionInterpolator:
    # Same as distribution_interpolator with slight changes to support the distribution coordinate system used in GENE
    def __init__(self, vpar , mu, spl=None, Te_perp=None, Te_par=None):
        self.vpar_min = np.min(vpar)
        self.vpar_max = np.max(vpar)
        self.mu_min = np.min(mu)
        self.mu_max = np.max(mu)
        self.spl = spl
        self.Te_perp = Te_perp
        self.Te_par = Te_par

    def eval_dist_gene(self, vpar, magmom, mode="spline"):
        if(mode == "spline"):
            return self.spl(vpar, magmom, grid=False)
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_gene_dvpar(self, vpar, magmom, mode="spline"):
        if(mode == "spline"):
            return self.spl(vpar, magmom, dx=1, grid=False)
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_gene_dmu(self, vpar, magmom, mode="spline"):
        if(mode == "spline"):
            return self.spl(vpar, magmom, dy=1, grid=False)
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist(self, u_perp, u_par, Te=None, mode="thermal"):
        if(mode == "thermal"):
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * Te)
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
            return a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(mode == "BM"):
            T0 = self.Te_par ** (1.0e0 / 3.0e0) * self.Te_perp ** (2.0e0 / 3.0e0)
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
            r = T0 / self.Te_par
            s = T0 / self.Te_perp
            gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
            a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
            return a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * np.exp(mu * (1 - gamma_drift_m))
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_du_perp(self, u_perp, u_par, Te=None, mode="thermal"):
        if(mode == "thermal"):
            mu = mass_e * c0 ** 2 / (e0 * Te)
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
            return -a * mu * u_perp / gamma * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(mode == "BM"):
            T0 = self.Te_par ** (1.0e0 / 3.0e0) * self.Te_perp ** (2.0e0 / 3.0e0)
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
            r = T0 / self.Te_par
            s = T0 / self.Te_perp
            gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
            a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
            return -(mu * s * u_perp) / gamma_drift_m * a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * \
                   np.exp(mu * (1 - gamma_drift_m))
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

    def eval_dist_du_par(self, u_perp, u_par, Te=None, mode="thermal"):
        if(mode == "thermal"):
            mu = mass_e * c0 ** 2 / (e0 * Te)
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_perp ** 2 + u_par ** 2)
            return -a * mu * u_par / gamma * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(mode == "BM"):
            T0 = self.Te_par ** (1.0e0 / 3.0e0) * self.Te_perp ** (2.0e0 / 3.0e0)
            mu = (cnst.physical_constants["electron mass"][0] * cnst.c ** 2.e0) / (cnst.e * T0)
            r = T0 / self.Te_par
            s = T0 / self.Te_perp
            gamma_drift_m = np.sqrt(1.0e0 + r * u_par ** 2 + s * u_perp ** 2)
            a = 1.0e0 / (1.0e0 + 105.0e0 / (128.0e0 * mu ** 2) + 15.0e0 / (8.0e0 * mu))
            return -(mu * r * u_par) / gamma_drift_m * a * (np.sqrt(mu / (2.e0 * np.pi)) ** 3) * \
                   np.exp(mu * (1 - gamma_drift_m))
        else:
            print("Selected mode {0:s} not supported".format(mode))
            raise ValueError

class SVec:
    # Structure that stores all information at LOS point s that is required to calculate the absorption coefficient / emissivity
    def __init__(self, rhop, Te, ne, freq_2X, theta):
        self.rhop = rhop
        self.Te = Te
        self.ne = ne
        self.freq_2X = freq_2X
        self.theta = theta
        self.cos_theta = np.cos(theta)
        self.sin_theta = np.sin(theta)

class EmAbsAlb:
# Provides absorption coefficeint and emissivity according to [2]
# [2] F. Albajar, N. Bertelli, M. Bornatici, and F. Engelmann, Plasma Physics and Controlled Fusion 49, 15 (2007), ISSN 0741-3335.

    def __init__(self, dist_mode="thermal"):
        # Sets distribution mode, allowed are:
        #    thermal
        #    FP -> Fokker-Planck calculated distribution -> distribution_interpolator
        #    BM -> Analytical BiMaxwellian on grid -> distribution_interpolator
        #    gene -> GENE calculated distribution -> gene_distribution_interpolator
        self.dist_mode = dist_mode
        self.B_min = None
        self.N_res = 200
        self.t = np.linspace(-1, 1, self.N_res)

    def refr_index(self, svec, omega, mode):
        # Calculate refractive index
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        if(warm_plasma):
        # Apply weakly relativistic correction, which is accurate the near cut-off
        # Hard to rigorously justify ->
        # USE WITH CAUTION
            w_mass_e = mass_e * np.sqrt(1.e0 + 5.e0 / mu)
        else:
            w_mass_e = mass_e
        Y = svec.freq_2X * np.pi * mass_e / w_mass_e / omega
        omega_p = e0 * np.sqrt(svec.ne / (eps0 * w_mass_e))
        X = omega_p ** 2 / omega ** 2
        N_abs = N_with_pol_vec(X, Y, svec.sin_theta, svec.cos_theta, mode)[0]
        return N_abs

    def is_resonant(self, rhop, Te, ne, freq_2X, theta, freq, m):
        svec = SVec(rhop, Te, ne, freq_2X, theta)
        mode = 1
        omega = freq * 2.e0 * np.pi
        self.abs_Albajar_resonance_line(svec, omega, mode, m=m)
        return self.resonant

    def j_abs_Alb(self, rhop, Te, ne, freq_2X, theta, freq, ext_dist=None, B_min=None, dist_source='FP', calc_abs=True, m=2):
        # Calculates the emissivity for a given point and distribution
        # Can also calculate absorption if calc_abs given
        # rhop normalized minor radius, not used
        # Te [eV]
        # ne [m^-3]
        # freq_2X second harmonic cyclotron frequency = omega_c / pi [Hz]
        # theta = acos((B.k)/(|B||k|)) [rad]
        # freq measured frequency [Hz]
        # Ext. dist. is expected to be a distribution_interpolator obj
        # B_min smallest magnetic field on current flux surface [T]
        # m harmonic number
        svec = SVec(rhop, Te, ne, freq_2X, theta)
        mode = 1
        omega = freq * 2.e0 * np.pi
        if(ext_dist is None):
            self.dist_mode = "thermal"
        elif(dist_source == "FP"):
            self.dist_mode = "ext"
            self.ext_dist = ext_dist
            if(B_min is None):
                print("Need B_min for external distribution")
                raise(ValueError)
            self.B_min = B_min
        elif(dist_source == "BM"):
            self.dist_mode = "BM"
            self.ext_dist = ext_dist
        else:
            self.dist_mode = "gene"
            self.ext_dist = ext_dist
        self.resonant = True
        if(calc_abs):
            u_par, u_perp, c_abs, j = self.abs_Albajar_along_res(svec, omega, mode, m=m)
        else:
            u_par, u_perp, j = self.j_Albajar_along_res(svec, omega, mode, m=m)
            c_abs = None
#        u_par_approx, u_perp_approx, c_abs_approx, j_approx = self.em_Hutch_along_res(svec, omega, 2)
#        u_par_max = self.get_u_par_max(svec, omega, 2)
        if(self.resonant):
            self.u_par = u_par
            self.u_perp = u_perp
            self.c_abs = c_abs
            self.j = j
#            self.u_par_approx = u_par_approx
#            self.u_perp_approx = u_perp_approx
#            self.c_abs_approx = c_abs_approx
#            self.j_approx = j_approx
#            self.u_par_max = u_par_max

    def dist(self, u_par, u_perp, mu, svec, grid=False):
        # Evaluates distribution function
        if(self.dist_mode == "thermal"):
            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
            gamma = np.sqrt(1.0 + u_par ** 2 + u_perp ** 2)
            return a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
                    np.exp(mu * (1.0 - gamma))
        elif(self.dist_mode == "ext"):
            if(self.B_min == 0.0):
                zeta = 1.0
            else:
                zeta = np.pi * svec.freq_2X * mass_e / (e0 * self.B_min)
                if(zeta < 1.0):
                    zeta = 1.0
            u, pitch = cyl_to_pol(u_par, u_perp, zeta)
            if(np.isscalar(u)):
                if(u < self.ext_dist.u_min or u > self.ext_dist.u_max):
                    return self.ext_dist.eval_dist(u_perp, u_par, Te=svec.Te, mode="thermal")
                else:
                    f = np.exp(self.ext_dist.eval_dist_cyl(u, pitch, mode="spline"))
            else:
                u_temp = np.copy(u)
                # Will be replaced later
                u_temp[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)] = self.ext_dist.u_min + 1.e-5
                f = np.exp(self.ext_dist.eval_dist_cyl(u_temp, pitch, mode="spline"))
                f[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)] = \
                    self.ext_dist.eval_dist(u_perp[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)], \
                                            u_par[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)], Te=svec.Te, mode="thermal")
            return f
        elif(self.dist_mode == "gene"):
            vpar, magmom = self.v_par_mu_to_cyl(u_par, u_perp, svec)
            if(np.isscalar(vpar)):
                if(self.ext_dist.vpar_min < vpar and  self.ext_dist.vpar_max > vpar and \
                    self.ext_dist.mu_min < magmom and self.ext_dist.mu_max > magmom):
                    return np.exp(self.ext_dist.eval_dist_gene(vpar, magmom, mode="spline"))
                else:
                    return 0.e0
            else:
                temp_vpar = np.copy(vpar)
                temp_magmom = np.copy(magmom)
                temp_vpar[temp_vpar <= self.ext_dist.vpar_min] = self.ext_dist.vpar_min + 1.e-11
                temp_vpar[temp_vpar >= self.ext_dist.vpar_max] = self.ext_dist.vpar_max - 1.e-11
                temp_magmom[temp_magmom <= self.ext_dist.mu_min] = self.ext_dist.mu_min + 1.e-11
                temp_magmom[temp_magmom >= self.ext_dist.mu_max] = self.ext_dist.mu_max - 1.e-11
                f = np.exp(self.ext_dist.eval_dist_gene(vpar, magmom, mode="spline"))
                f[np.logical_or(np.logical_or(vpar <= self.ext_dist.vpar_min, \
                                                vpar >= self.ext_dist.vpar_max), \
                                                np.logical_or(magmom <= self.ext_dist.mu_min, \
                                                magmom >= self.ext_dist.mu_max))] = 0.e0
#                print("upar    uperp    f")
#                for i in range(len(vpar)):
#                    print("{0:1.2e} {1:1.2e} {2:1.2e}".format(vpar[i], magmom[i], f[i]))
            return f
        else:
            print("Bad dist mode")
            raise(ValueError)

    def Rdist(self, u_par, u_perp, m_omega_bar, N_par, mu, svec):
        # Evaluates operator R applied onto distribution function - see eq. 2d of ref [2]
        if(self.dist_mode == "thermal"):
            return -mu * self.dist(u_par, u_perp, mu, svec)
        elif(self.dist_mode == "ext"):
            if(np.any(u_perp <= 1.e-7)):
                return (m_omega_bar * self.f_deriv_u_perp(u_par, u_perp, svec) / (u_perp + 1.e-7) + \
                                 N_par * self.f_deriv_u_par(u_par, u_perp, svec))
            else:
                return (m_omega_bar * self.f_deriv_u_perp(u_par, u_perp, svec) / (u_perp) + \
                                 N_par * self.f_deriv_u_par(u_par, u_perp, svec))
        elif(self.dist_mode == "gene"):
#            a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
#            gamma = np.sqrt(1.0 + u_par ** 2 + u_perp ** 2)
#            Rf = -mu * a * np.sqrt(mu / (2.0 * np.pi)) ** 3 * \
#                    np.exp(mu * (1.0 - gamma))
            if(np.any(u_perp <= 1.e-7)):
                return (m_omega_bar * self.f_deriv_u_perp(u_par, u_perp, svec) / (u_perp + 1.e-7) + \
                                 N_par * self.f_deriv_u_par(u_par, u_perp, svec))  # + Rf

            else:
                return (m_omega_bar * self.f_deriv_u_perp(u_par, u_perp, svec) / (u_perp) + \
                                 N_par * self.f_deriv_u_par(u_par, u_perp, svec))  # + Rf
        else:
            print("Bad dist mode", self.dist_mode)
            raise(ValueError)

    def v_par_mu_to_cyl(self, u_par, u_perp, svec):
        # Chain rule for derivatives
        gam = np.sqrt(1.e0 + u_par ** 2 + u_perp ** 2)
        vpar = u_par / gam
        mu = (u_perp / gam) ** 2 / (2.e0 * self.B_min)
        return vpar, mu

    def f_deriv_u_par(self, u_par, u_perp, svec):
        # df/du_par
        if(self.dist_mode == "ext"):
            if(self.B_min == 0.0):
                zeta = 1.0
            else:
                zeta = np.pi * svec.freq_2X * mass_e / (e0 * self.B_min)
                if(zeta < 1.0):
                    zeta = 1.0
            u, pitch = cyl_to_pol(u_par, u_perp, zeta)
            h = (zeta - 1.e0) / zeta
            if(self.B_min != 0.0):
                dpitch_du_par = ((u_par * np.sqrt((u ** 2 - u_par ** 2 - h * u_perp ** 2) * (u_par ** 2 + h * u_perp ** 2))) / \
                                 (u ** 2 * (u_par ** 2 + h * u_perp ** 2)))
                if(np.isscalar(dpitch_du_par)):
                    if(u_par > 0.e0):
                        dpitch_du_par *= -1
                else:
                    dpitch_du_par[u_par > 0.e0] *= -1
            else:
                dpitch_du_par = -u_perp / (u_par ** 2 + u_perp ** 2)
            if(np.isscalar(dpitch_du_par)):
                    if(np.isnan(dpitch_du_par)):
                        dpitch_du_par = -u_perp / (u_par ** 2 + u_perp ** 2)
            else:
                dpitch_du_par[np.isnan(dpitch_du_par)] = -u_perp[np.isnan(dpitch_du_par)] / (u_par[np.isnan(dpitch_du_par)] ** 2 + u_perp[np.isnan(dpitch_du_par)] ** 2)
            du_du_par = u_par / u
    #        u_par_diff = np.zeros(4)
    #        u_par_diff[0] = u_par + 2.e-4
    #        u_par_diff[1] = u_par + 1.e-4
    #        u_par_diff[2] = u_par - 1.e-4
    #        u_par_diff[3] = u_par - 2.e-4
    #        u_diff = np.zeros(4)
    #        pitch_diff = np.zeros(4)
#            if(self.B_min == 0.0):
#                        zeta = 1.0
#                    else:
#                        zeta = np.pi * svec.freq_2X * mass_e / (e0 * self.B_min)
#                        if(zeta < 1.0):
#                            zeta = 1.0
    #        for i in range(4):
    #            u_diff[i], pitch_diff[i] = cyl_to_pol(u_par_diff[i], u_perp, zeta)
    #        print("du_du_par ana", du_du_par)
    #        print("du_du_par num", (-u_diff[0] + 8.e0 * u_diff[1] - \
    #                 8.e0 * u_diff[2] + u_diff[3]) / (12.e-4))
    #        print("dpitch_du_par ana", dpitch_du_par)
    #        print("dpitch_du_par num", (-pitch_diff[0] + 8.e0 * pitch_diff[1] - \
    #                 8.e0 * pitch_diff[2] + pitch_diff[3]) / (12.e-4))
    #        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
    #        print("ana", (self.ext_dist.derx(u, pitch) * du_du_par + dpitch_du_par * self.ext_dist.dery(u, pitch)) * np.exp(self.ext_dist.eval(u, pitch)))
    #        print("num", (-self.dist(u_par_diff[0], u_perp, mu, svec) + 8.e0 * self.dist(u_par_diff[1], u_perp, mu, svec) - \
    #                      8.e0 * self.dist(u_par_diff[2], u_perp, mu, svec) + self.dist(u_par_diff[3], u_perp, mu, svec)) / (12.e-4))
            if(np.isscalar(u)):
                if(u < self.ext_dist.u_min or u > self.ext_dist.u_max):
                    return 0.e0
                else:
                    df_dupar = (self.ext_dist.eval_dist_cyl_du(u, pitch, mode="spline") * du_du_par + \
                                dpitch_du_par * self.ext_dist.eval_dist_cyl_dpitch(u, pitch, mode="spline")) * \
                                np.exp(self.ext_dist.eval_dist_cyl(u, pitch, mode="spline"))
            else:
                u_temp = np.copy(u)
                # Will be replaced later
                u_temp[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)] = self.ext_dist.u_min + 1.e-5
                df_dupar = (self.ext_dist.eval_dist_cyl_du(u_temp, pitch, mode="spline") * du_du_par + \
                            dpitch_du_par * self.ext_dist.eval_dist_cyl_dpitch(u_temp, pitch, mode="spline")) * \
                            np.exp(self.ext_dist.eval_dist_cyl(u_temp, pitch, mode="spline"))
                df_dupar[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)] = 0.e0
            return df_dupar
        else:
            gamma = np.sqrt(1.e0 + u_perp ** 2 + u_par ** 2)
            vpar, mu = self.v_par_mu_to_cyl(u_par, u_perp, svec)
            dvpar_du_par = (1.e0 + u_perp ** 2) / gamma ** 3
            dmu_du_par = -(u_par * u_perp ** 2) / (self.B_min * gamma ** 4)
            if(np.isscalar(vpar)):
                if(vpar <= self.ext_dist.vpar_min or \
                   vpar >= self.ext_dist.vpar_max or \
                   mu <= self.ext_dist.mu_min or  \
                   mu >= self.ext_dist.mu_max):
                    fderiv_u_par = 0.e0
                else:
                    fderiv_u_par = self.ext_dist.eval_dist_gene_dvpar(vpar, mu, mode="spline") * dvpar_du_par + \
                                    dmu_du_par * self.ext_dist.eval_dist_gene_dmu(vpar, mu, mode="spline") * \
                                    np.exp(self.ext_dist.eval_dist_gene(vpar, mu, mode="spline"))
            else:
                temp_vpar = np.copy(vpar)
                temp_mu = np.copy(mu)
                temp_vpar[temp_vpar <= self.ext_dist.vpar_min] = self.ext_dist.vpar_min + 1.e-11
                temp_vpar[temp_vpar >= self.ext_dist.vpar_max] = self.ext_dist.vpar_max - 1.e-11
                temp_mu[temp_mu <= self.ext_dist.mu_min] = self.ext_dist.mu_min + 1.e-11
                temp_mu[temp_mu >= self.ext_dist.mu_max] = self.ext_dist.mu_max - 1.e-11
                fderiv_u_par = self.ext_dist.eval_dist_gene_dvpar(temp_vpar, temp_mu, mode="spline") * dvpar_du_par + \
                                dmu_du_par * self.ext_dist.eval_dist_gene_dmu(temp_vpar, temp_mu, mode="spline") * \
                                np.exp(self.ext_dist.eval_dist_gene(temp_vpar, temp_mu, mode="spline"))
                fderiv_u_par[np.logical_or(np.logical_or(vpar <= self.ext_dist.vpar_min, \
                                            vpar >= self.ext_dist.vpar_max), \
                                            np.logical_or(mu <= self.ext_dist.mu_min, \
                                            mu >= self.ext_dist.mu_max))] = 0.e0
            return fderiv_u_par

    def f_deriv_u_perp(self, u_par, u_perp, svec):
        # df/du_perp
        if(self.dist_mode == "ext"):
            if(self.B_min == 0.0):
                zeta = 1.0
            else:
                zeta = np.pi * svec.freq_2X * mass_e / (e0 * self.B_min)
                if(zeta < 1.0):
                    zeta = 1.0
            u, pitch = cyl_to_pol(u_par, u_perp, zeta)
            h = (zeta - 1.e0) / zeta
            if(np.any(u ** 2 - u_par ** 2 - h * u_perp ** 2 < 0)):
                print("u**2 <  u_par**2 + u_perp**2??", u ** 2 - u_par ** 2 - h * u_perp ** 2)
            if(self.B_min != 0.0):
                dpitch_du_perp = -((u_perp * (u_par ** 2 + h * (-u ** 2 + u_perp ** 2))) / \
                                    (u ** 2 * np.sqrt((u ** 2 - u_par ** 2 - h * u_perp ** 2) * (u_par ** 2 + h * u_perp ** 2))))
                if(np.isscalar(dpitch_du_perp)):
                    if(u_par > 0.e0):
                        dpitch_du_perp *= -1
                    if(u_perp == 0.e0):
                        dpitch_du_perp = 0.e0
                else:
                    dpitch_du_perp[u_par > 0.e0] *= -1
                    dpitch_du_perp[u_perp == 0.e0] = 0.e0
            else:
                dpitch_du_perp = u_par / (u_par ** 2 + u_perp ** 2)
                if(np.isscalar(dpitch_du_perp)):
                    if(np.isnan(dpitch_du_perp)):
                        dpitch_du_perp = u_par / (u_par ** 2 + u_perp ** 2)
                else:
                    dpitch_du_perp[np.isnan(dpitch_du_perp)] = u_par[np.isnan(dpitch_du_perp)] / (u_par[np.isnan(dpitch_du_perp)] ** 2 + u_perp[np.isnan(dpitch_du_perp)] ** 2)
            du_du_perp = u_perp / u
            try:
                if(np.isscalar(u)):
                    if(u < self.ext_dist.u_min or u > self.ext_dist.u_max):
                        return 0.e0
                    else:
                        df_duperp = (self.ext_dist.eval_dist_cyl_du(u, pitch, mode="spline") * du_du_perp + \
                                     dpitch_du_perp * self.ext_dist.eval_dist_cyl_dpitch(u, pitch, mode="spline")) * \
                                     np.exp(self.ext_dist.eval_dist_cyl(u, pitch, mode="spline"))
                else:
                    u_temp = np.copy(u)
                    # Will be replaced later
                    u_temp[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)] = self.ext_dist.u_min + 1.e-5
                    df_duperp = (self.ext_dist.eval_dist_cyl_du(u_temp, pitch, mode="spline") * du_du_perp + \
                                     dpitch_du_perp * self.ext_dist.eval_dist_cyl_dpitch(u_temp, pitch, mode="spline")) * \
                                     np.exp(self.ext_dist.eval_dist_cyl(u_temp, pitch, mode="spline"))
                    df_duperp[np.logical_or(u > self.ext_dist.u_max, u < self.ext_dist.u_min)] = 0.e0
                return df_duperp
            except ValueError as e:
                print(e)
                print("Bispline failed")
                print(np.shape(u), np.shape(pitch))
                raise ValueError
            #        u_perp_diff = np.zeros(4)
#        u_perp_diff[0] = u_perp + 2.e-4
#        u_perp_diff[1] = u_perp + 1.e-4
#        u_perp_diff[2] = u_perp - 1.e-4
#        u_perp_diff[3] = u_perp - 2.e-4
#        u_diff = np.zeros(4)
#        pitch_diff = np.zeros(4)
#        for i in range(4):
#            u_diff[i], pitch_diff[i] = cyl_to_pol(u_par, u_perp_diff[i], svec, self.B_min)
#        print("du_du_perp ana", du_du_perp)
#        print("du_du_perp num", (-u_diff[0] + 8.e0 * u_diff[1] - \
#                 8.e0 * u_diff[2] + u_diff[3]) / (12.e-4))
#        print("dpitch_du_perp ana", dpitch_du_perp)
#        print("dpitch_du_perp num", (-pitch_diff[0] + 8.e0 * pitch_diff[1] - \
#                 8.e0 * pitch_diff[2] + pitch_diff[3]) / (12.e-4))
#        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
#        print("ana", (self.ext_dist.derx(u, pitch) * du_du_perp + dpitch_du_perp * self.ext_dist.dery(u, pitch)) * np.exp(self.ext_dist.eval(u, pitch)))
#        print("num", (-self.dist(u_par, u_perp_diff[0], mu, svec) + 8.e0 * self.dist(u_par, u_perp_diff[1], mu, svec) - \
#                      8.e0 * self.dist(u_par, u_perp_diff[2], mu, svec) + self.dist(u_par, u_perp_diff[3], mu, svec)) / (12.e-4))

        else:
            gamma = np.sqrt(1.e0 + u_perp ** 2 + u_par ** 2)
            vpar, mu = self.v_par_mu_to_cyl(u_par, u_perp, svec)
            dvpar_du_perp = -(u_par * u_perp) / gamma ** 3
            dmu_du_perp = (1.e0 + u_par ** 2) * u_perp / (self.B_min * gamma ** 4)
            if(np.isscalar(vpar)):
                if(vpar <= self.ext_dist.vpar_min or \
                   vpar >= self.ext_dist.vpar_max or \
                   mu <= self.ext_dist.mu_min or  \
                   mu >= self.ext_dist.mu_max):
                    fderiv_u_perp = 0.e0
                else:
                    fderiv_u_perp = self.ext_dist.eval_dist_gene_dvpar(vpar, mu, mode="spline") * dvpar_du_perp + \
                                dmu_du_perp * self.ext_dist.eval_dist_gene_dmu(vpar, mu, mode="spline") * \
                                np.exp(self.ext_dist.spl(vpar, mu, grid=False))

            else:
                temp_vpar = np.copy(vpar)
                temp_mu = np.copy(mu)
                temp_vpar[temp_vpar <= self.ext_dist.vpar_min] = self.ext_dist.vpar_min + 1.e-11
                temp_vpar[temp_vpar >= self.ext_dist.vpar_max] = self.ext_dist.vpar_max - 1.e-11
                temp_mu[temp_mu <= self.ext_dist.mu_min] = self.ext_dist.mu_min + 1.e-11
                temp_mu[temp_mu >= self.ext_dist.mu_max] = self.ext_dist.mu_max - 1.e-11
                fderiv_u_perp = self.ext_dist.eval_dist_gene_dvpar(temp_vpar, temp_mu, mode="spline") * dvpar_du_perp + \
                                dmu_du_perp * self.ext_dist.eval_dist_gene_dmu(temp_vpar, temp_mu, mode="spline") * \
                                np.exp(self.ext_dist.eval_dist_gene(temp_vpar, temp_mu, mode="spline"))
                fderiv_u_perp[np.logical_or(np.logical_or(vpar <= self.ext_dist.vpar_min, \
                                            vpar >= self.ext_dist.vpar_max), \
                                            np.logical_or(mu <= self.ext_dist.mu_min, \
                                            mu >= self.ext_dist.mu_max))] = 0.e0
            return fderiv_u_perp


    def pol_vec(self, omega, X, Y, sin_theta, cos_theta, mode):
        # Helper routine to retrieve polarization vector
        N_abs = np.zeros(len(sin_theta))
        e = np.zeros((len(sin_theta), 3), dtype=np.complex)
        for i in range(len(sin_theta)):
            N_abs[i], e[i] = N_with_pol_vec(X[i], Y[i], sin_theta[i], cos_theta[i], mode)
        return e

    def abs_Albajar(self, svec, omega, mode, n_max=3, n_min=2):
        # Get integrated absorption coefficient as sum of multiple harminics
#        print("Te, ne", svec.Te, svec.ne)
        if(svec.Te < 3.0 or svec.ne < 1.e15):
            return 0, 0
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        if(warm_plasma):
            w_mass_e = mass_e * np.sqrt(1.e0 + 5.e0 / mu)
        else:
            w_mass_e = mass_e
        omega_c = svec.freq_2X * np.pi
        Y = svec.freq_2X * np.pi * mass_e / w_mass_e / omega
        omega_p = e0 * np.sqrt(svec.ne / (eps0 * w_mass_e))
        omega_p_cold = e0 * np.sqrt(svec.ne / (eps0 * mass_e))
        X = omega_p ** 2 / omega ** 2
        omega_bar = omega / omega_c
        c_abs = 0.e0
        c_abs_non_rel = 0.e0
        j = 0.e0
        N_abs, e = N_with_pol_vec(X, Y, svec.sin_theta, svec.cos_theta, mode)
        N_par = svec.cos_theta * N_abs
        N_perp = svec.sin_theta * N_abs # Note that we need an absolute value here, which is what is done in the Fortran version
        # However, a negative sign of N_perp does not change the results at all.
        # To avoid implementing the correctly signed N_perp consistenly we leave it here with the possibly negative sign for convenience
        if(N_par ** 2 >= 1.0 or N_abs <= 0.0 or N_abs > 1.0):
            return 0, 0
        m_0 = np.sqrt(1.e0 - N_par ** 2) * omega_bar
        for m_sum in range(n_min, n_max + 1):
            if(float(m_sum) > m_0):
                c_abs_m, j_m = self.abs_Al_integral_nume(svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m_sum)
                c_abs = c_abs + np.sqrt((float(m_sum) / m_0) ** 2 - 1.e0) * c_abs_m
                j = j + np.sqrt((float(m_sum) / m_0) ** 2 - 1.e0) * j_m
        if(c_abs == 0.e0):
            return 0, 0
        c_abs = -(c_abs * 2.e0 * np.pi ** 2 / m_0)
        c_abs = c_abs * omega_p_cold ** 2 / (omega_c * c0)
        c_abs_non_rel = -(c_abs_non_rel * 2.e0 * np.pi ** 2 / m_0)
        c_abs_non_rel = c_abs_non_rel * omega_p_cold ** 2 / (omega_c * c0)
        j = j * 2.e0 * np.pi ** 2 / m_0
        j = j * omega_p_cold ** 2 / (omega_c * c0) * omega ** 2 * mass_e / (4.e0 * np.pi ** 2)
        return c_abs, j

    def abs_Albajar_resonance_line(self, svec, omega, mode, m=2):
        # Gives resonance line
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        if(warm_plasma):
            w_mass_e = mass_e * np.sqrt(1.e0 + 5.e0 / mu)
        else:
            w_mass_e = mass_e
        omega_c = svec.freq_2X * np.pi
        Y = svec.freq_2X * np.pi * mass_e / w_mass_e / omega
        omega_p = e0 * np.sqrt(svec.ne / (eps0 * w_mass_e))
        X = omega_p ** 2 / omega ** 2
        omega_bar = omega / omega_c
        N_abs = N_with_pol_vec(X, Y, svec.sin_theta, svec.cos_theta, mode)[0]
        N_par = svec.cos_theta * N_abs
        if(N_par ** 2 >= 1.0 or N_abs <= 0.0 or N_abs > 1.0):
            self.resonant = False
            return 0.0, 0.0
        m_0 = np.sqrt(1.e0 - N_par ** 2) * omega_bar
        u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * self.t)
        u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - self.t ** 2)
        u_perp_sq[u_perp_sq <= 0] += 1.e-7
        if(np.any(u_perp_sq < 0)):
            self.resonant = False
            return 0.0, 0.0
        u_perp = np.sqrt(u_perp_sq)
        self.resonant = True
        return u_par, u_perp

    def abs_Albajar_along_res(self, svec, omega, mode, m=2):
        # Evaluates absorption coefficient along a resonance line
        self.resonant = True
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        if(warm_plasma):
            w_mass_e = mass_e * np.sqrt(1.e0 + 5.e0 / mu)
        else:
            w_mass_e = mass_e
        omega_c = svec.freq_2X * np.pi
        Y = svec.freq_2X * np.pi * mass_e / w_mass_e / omega
        omega_p = e0 * np.sqrt(svec.ne / (eps0 * w_mass_e))
        omega_p_cold = e0 * np.sqrt(svec.ne / (eps0 * mass_e))
        X = omega_p ** 2 / omega ** 2
        omega_bar = omega / omega_c
        c_abs = 0.e0
        j = 0.e0
        N_abs, e = N_with_pol_vec(X, Y, svec.sin_theta, svec.cos_theta, mode)
        N_par = svec.cos_theta * N_abs
        N_perp = svec.sin_theta * N_abs
        if(N_par ** 2 >= 1.0 or N_abs <= 0.0 or N_abs > 1.0):
            self.resonant = False
            return 0.0, 0.0, 0.0, 0.0
        m_0 = np.sqrt(1.e0 - N_par ** 2) * omega_bar
        if(np.any(((float(m) / m_0) ** 2 - 1.e0) < 0.0)):
            self.resonant = False
            return 0.0, 0.0, 0.0, 0.0
        u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * self.t)
        u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - self.t ** 2)
        u_perp_sq[u_perp_sq <= 0] += 1.e-7
        if(np.any(u_perp_sq < 0)):
            self.resonant = False
            return 0.0, 0.0, 0.0, 0.0
        u_perp = np.sqrt(u_perp_sq)
        if(not float(m) < m_0):
            c_abs = self.abs_Al_integrand_c_abs(self.t, [svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m])
            c_abs = np.sqrt((float(m) / m_0) ** 2 - 1.e0) * c_abs
            j = self.abs_Al_integrand_j(self.t, [svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m])
            j = np.sqrt((float(m) / m_0) ** 2 - 1.e0) * j
        else:
            self.resonant = False
            return 0, 0, 0, 0
        c_abs = -(c_abs * 2.e0 * np.pi ** 2 / m_0)
        c_abs = c_abs * omega_p_cold ** 2 / (omega_c * c0)
        j = j * 2.e0 * np.pi ** 2 / m_0
        j = j * omega_p_cold ** 2 / (omega_c * c0) * omega ** 2 * mass_e / (4.e0 * np.pi ** 2)
        return u_par, u_perp, c_abs, j

    def j_Albajar_along_res(self, svec, omega, mode, m=2):
        # Evaluates emissivity along a resonance line
        self.resonant = True
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        if(warm_plasma):
            w_mass_e = mass_e * np.sqrt(1.e0 + 5.e0 / mu)
        else:
            w_mass_e = mass_e
        omega_c = svec.freq_2X * np.pi
        Y = svec.freq_2X * np.pi * mass_e / w_mass_e / omega
        omega_p = e0 * np.sqrt(svec.ne / (eps0 * w_mass_e))
        omega_p_cold = e0 * np.sqrt(svec.ne / (eps0 * mass_e))
        X = omega_p ** 2 / omega ** 2
        omega_bar = omega / omega_c
        j = 0.e0
        N_abs, e = N_with_pol_vec(X, Y, svec.sin_theta, svec.cos_theta, mode)
        N_par = svec.cos_theta * N_abs
        N_perp = svec.sin_theta * N_abs
        if(N_par ** 2 >= 1.0 or N_abs <= 0.0 or N_abs > 1.0):
            self.resonant = False
            return 0.0, 0.0, 0.0, 0.0
        m_0 = np.sqrt(1.e0 - N_par ** 2) * omega_bar
        # Weights not needed atm
        # t_weights = np.concatenate([[0.5 * (t[1] - t[0])], t[2:-1] - t[1:-2], 0.5 * (t[-1] - t[-2])])
        u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * self.t)
        u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - self.t ** 2)
        u_perp_sq[u_perp_sq <= 0] += 1.e-7
        if(np.any(u_perp_sq < 0)):
            self.resonant = False
            return 0.0, 0.0, 0.0, 0.0
        u_perp = np.sqrt(u_perp_sq)
        if(not float(m) < m_0):
            j = self.abs_Al_integrand_j(self.t, [svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m])
            j = np.sqrt((float(m) / m_0) ** 2 - 1.e0) * j
        else:
            self.resonant = False
            return 0, 0, 0, 0
        j = j * 2.e0 * np.pi ** 2 / m_0
        j = j * omega_p_cold ** 2 / (omega_c * c0) * omega ** 2 * mass_e / (4.e0 * np.pi ** 2)
        return u_par, u_perp, j

    def abs_Al_integral_nume(self, svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m):
        # Integrates the absorption coefficient along the resonance line using gaussian quadrature
        c_abs_res = quad(self.abs_Al_integrand_c_abs, -1.0, 1.0, args=[svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m])
        j_res = quad(self.abs_Al_integrand_j, -1.0, 1.0, args=[svec, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m])
        return c_abs_res[0], j_res[0]

    def abs_Al_integrand_c_abs(self, t, args):
        # Returns integrand of resonacne line integral for absorption coefficient
        svec = args[0]
        X = args[1]
        Y = args[2]
        omega_bar = args[3]
        m_0 = args[4]
        N_abs = args[5]
        N_par = args[6]
        N_perp = args[7]
        e = args[8]
        m = args[9]
        m_omega_bar = float(m) / omega_bar
        mu = c0 ** 2 * mass_e / (svec.Te * e0)
        u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * t)
        u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - t ** 2)
        if(np.isscalar(u_perp_sq)):
            if(u_perp_sq < 0):
                u_perp_sq += 1.e-7
        else:
            u_perp_sq[u_perp_sq <= 0] += 1.e-7
        if(np.any(u_perp_sq < 0)):
            print(u_perp_sq)
            raise ValueError
        u_perp = np.sqrt(u_perp_sq)
        pol_fact = self.abs_Al_pol_fact(t, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m)
        c_abs_int = pol_fact * self.Rdist(u_par, u_perp, m_omega_bar, N_par, mu, svec)
        return c_abs_int

    def abs_Al_integrand_j(self, t, args):
        # Returns integrand of resonacne line integral for emissiviity
        svec = args[0]
        X = args[1]
        Y = args[2]
        omega_bar = args[3]
        m_0 = args[4]
        N_abs = args[5]
        N_par = args[6]
        N_perp = args[7]
        e = args[8]
        m = args[9]
        mu = c0 ** 2 * mass_e / (svec.Te * e0)
        u_par = 1.e0 / np.sqrt(1.e0 - N_par ** 2) * (float(m) / m_0 * N_par + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * t)
        u_perp_sq = ((float(m) / m_0) ** 2 - 1.e0) * (1.e0 - t ** 2)
        if(np.isscalar(u_perp_sq)):
            if(u_perp_sq < 0):
                u_perp_sq += 1.e-7
        else:
            u_perp_sq[u_perp_sq <= 0] += 1.e-7
        if(np.any(u_perp_sq < 0)):
            print(u_perp_sq)
            raise ValueError
        u_perp = np.sqrt(u_perp_sq)
        pol_fact = self.abs_Al_pol_fact(t, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m)
        j_int = pol_fact * self.dist(u_par, u_perp, mu, svec)
        return j_int

# Following rotuines are a failed attempt to calculate energy loss from the Albajar emissivity
# DO NOT USE!!!

    def eval_omega(self, u_par, gamma, omega_c, N_par, n):

        return n * omega_c / (gamma - u_par * N_par)

    def omega_root(self, omega, u_par, gamma, theta, omega_c, omega_p, mode, n):
        X = omega_p ** 2 / omega ** 2
        Y = omega_c / omega
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        N_abs = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)[0]
        return omega - self.eval_omega(u_par, gamma, omega_c, N_abs * cos_theta, n)

    def calc_omega(self, u_par, gamma, theta, omega_c, omega_p, mode, n):
        args = (u_par, gamma, theta, omega_c, omega_p, mode, n)
        omega_0 = self.eval_omega(u_par, gamma, omega_c, np.cos(theta), n)
        omega = newton(self.omega_root, omega_0, args=args)
        return omega

    def single_electron_emissivity(self, theta, args):
        u_par = args[0]
        u_perp = args[1]
        omega_c = args[2]
        omega_p = args[3]
        gamma = np.sqrt(1.0 + u_par ** 2 + u_perp ** 2)
        warm_plasma = False
        emissivity = 0
        for mode in [-1]:  # -1,
            for n in [2, 10]:
                if(np.isscalar(theta)):
                    omega = self.calc_omega(u_par, gamma, theta, omega_c, omega_p, mode, n)
                else:
                    omega = np.zeros(len(theta))
                    for i in range(len(theta)):
                        omega[i] = self.calc_omega(u_par, gamma, theta[i], omega_c, omega_p, mode, n)
                omega_bar = omega / omega_c
                X = omega_p ** 2 / omega ** 2
                Y = omega_c / omega
                if(np.isscalar(theta)):
                    omega = self.calc_omega(u_par, gamma, theta, omega_c, omega_p, mode, n)
                else:
                    N_abs = np.zeros(len(theta))
                    e = np.zeros((len(theta), 3), dtype=np.complex)
                    for i in range(len(theta)):
                        N_abs[i], e[i], = N_with_pol_vec(X[i], Y[i], np.sin(theta[i]), np.cos(theta[i]), mode)
                        N_perp = np.sin(theta) * N_abs
                        b = omega_bar * N_perp * u_perp
                        emissivity_add = (e[0] + (omega_bar * N_perp * u_par * e[2] / float(n))) * self.BesselJ(n, b)
                        emissivity_add += 0.0 + (e[1] * b / float(n) * 1.0 / 2.0 * (self.BesselJ(n - 1, b) - self.BesselJ(n + 1, b))) * 1.j
                        emissivity_add = np.abs(emissivity_add * np.conjugate(emissivity_add))
                        emissivity += (float(n) / (omega_bar * N_perp)) ** 2 * omega ** 3 / (omega_c * n) * emissivity_add
        emissivity /= gamma * u_perp
        emissivity *= (cnst.e ** 2) / (4.0 * cnst.c * np.pi * cnst.epsilon_0) / cnst.e  # units of eV

        # emissivity *= cnst.e ** 2 * omega ** 4 / (omega_c ** 2 * c0 * 2.e0 * np.pi * cnst.epsilon_0) / cnst.e  # We want eV that makes the scale of the problem closer to unity
        return emissivity

    def integrated_emissivity(self, u_par, u_perp, omega_c, omega_p):
        return 2.0 * np.pi * quad(self.single_electron_emissivity, -np.pi / 2.0, np.pi / 2.0, args=[u_par, u_perp, omega_c, omega_p])


    def abs_Al_pol_fact(self, t, X, Y, omega_bar, m_0, N_abs, N_par, N_perp, e, m):
        # "Polarization factor" see ref. 2
        x_m = N_perp * omega_bar * np.sqrt((float(m) / m_0) ** 2 - 1.e0)
        N_eff = (N_perp * N_par) / (1.e0 - N_par ** 2)
        Axz = e[0] + N_eff * e[2]
        Axz_sq = np.abs(Axz) ** 2
        Re_Axz_ey = np.real(np.complex(0.0, 1.0) * Axz * np.conjugate(e[1]))
        Re_Axz_ez = np.real(np.complex(0.0, Axz * np.conjugate(e[2])))
        Re_ey_ez = np.real(np.complex(0.0, np.conjugate(e[1]) * e[2]))
        ey_sq = np.abs(e[1]) ** 2
        ez_sq = np.abs(e[2]) ** 2
        pol_fact = (Axz_sq + ey_sq) * self.BesselJ(m , x_m * np.sqrt(1.e0 - t ** 2)) ** 2
        pol_fact = pol_fact + Re_Axz_ey * x_m / float(m) * self.abs_Al_bessel_sqr_deriv(m, x_m, np.sqrt(1.e0 - t ** 2))
        pol_fact = pol_fact - (x_m * np.sqrt(1.e0 - t ** 2) / float(m)) ** 2 * \
            ey_sq * self.BesselJ(m - 1, x_m * np.sqrt(1.e0 - t ** 2)) * self.BesselJ(m + 1, x_m * np.sqrt(1.e0 - t ** 2))
        pol_fact = pol_fact + (x_m / (float(m) * np.sqrt(1.0 - N_par ** 2))) ** 2 * \
            ez_sq * (t * self.BesselJ(m , x_m * np.sqrt(1.e0 - t ** 2))) ** 2
        pol_fact = pol_fact + x_m / (float(m) * np.sqrt(1.0 - N_par ** 2)) * \
            2.e0 * Re_Axz_ez * t * self.BesselJ(m , x_m * np.sqrt(1.e0 - t ** 2)) ** 2
        pol_fact = pol_fact + x_m / (float(m) * np.sqrt(1.0 - N_par ** 2)) * \
            Re_ey_ez * t * x_m / float(m) * self.abs_Al_bessel_sqr_deriv(m, x_m, np.sqrt(1.e0 - t ** 2))
        pol_fact = pol_fact * (float(m) / (N_perp * omega_bar)) ** 2
        return pol_fact

    def abs_Al_bessel_sqr_deriv(self, m , x, a):
        # Calculates dJ/dxn
        return a * self.BesselJ(m, a * x) * (self.BesselJ(m - 1, a * x) - \
          self.BesselJ(m + 1, a * x))

    def BesselJ(self, n, x):
        # Fast method for evaluating Besselfunctions
        # Probably not useful for python adapted from the Fortran90 code
        n_fac = 1.0
        for i in range(1, n + 1):
            n_fac = float(i) * n_fac
        BesselJ = 1.0 / n_fac
        n_fac = n_fac * float(n + 1)
        BesselJ = BesselJ - x ** 2 / (n_fac * 2.e0 ** (2))
        n_fac = n_fac * float(n + 2)
        BesselJ = BesselJ + x ** 4 / (n_fac * 2.e0 ** (5))
        n_fac = n_fac * float(n + 3)
        BesselJ = BesselJ - x ** 6 / (n_fac * 2.e0 ** (7) * 3.e0)
        BesselJ = BesselJ * x ** n / (2.0) ** n
        return BesselJ


    def get_pol_vec_carth(self, omega, X, Y, mode, N_vec, B_vec):
        debug = False
        cos_theta = np.sum(N_vec * B_vec) / np.sqrt(np.sum(N_vec ** 2)) / np.sqrt(np.sum(B_vec ** 2))
        sin_theta = np.sin(np.arccos(cos_theta))
        N_abs, pol_vec = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)
        if(N_abs == 0.e0):
            return np.zeros(3)
        pol_vec[:] /= np.sqrt(np.sum(np.abs(pol_vec) ** 2))  # Normalized to 1 for calculation of X/O-mode fraction
        N_perp = N_abs * sin_theta
        pol_vec_real = np.zeros(3, dtype=np.complex)
        pol_vec_real[0] = np.real(pol_vec[0])
        pol_vec_real[1] = np.imag(pol_vec[1])
        pol_vec_real[2] = np.real(pol_vec[2])
        if(np.any(pol_vec_real != pol_vec_real)):
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("N_perp", N_perp)
            print("omega", omega)
            return None
        if(debug):
            print("pol_vec", pol_vec_real)
        # First we rotate the polarization vector into the carthesian coordinate system of the machine
        # Now calculate rotation from Cartesian to reference frame of the wave and the polarization vector.
        # The polarization coefficients are given in a way so that
        # vec(N) = (vec(e_x) * sin(theta) + vec(e_z) * cos(theta)) * N_abs
        # vec(B) = B e_z
        # Hence, the unit vectors are given by:
        # vec(e_x) = vec(N) / N_abs - cos(theta) * vec(B) / B_abs
        # vec(e_y) = vec(e_x) x vec(e_z)
        # vec(e_z) = vec(B) / B_abs
        # The rotation is then given according to https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotations_and_motions
        N_abs_ray = np.sqrt(sum(N_vec ** 2))
        N_vec_norm = N_vec / N_abs_ray
        B_abs = np.sqrt(np.sum(B_vec ** 2))
        if(debug):
            print("omega", omega)
            print("X", X)
            print("Y", Y)
            print("N_vec", N_vec)
            print("B_vec", B_vec)
            print("theta", np.rad2deg(np.arccos(cos_theta)))
        if(mode > 0 and debug):
            print("X-mode")
        if(mode < 0 and debug):
            print("O-mode")
        # N_vec already points towards the antenna when its is copied into svec
        e_x = N_vec_norm - cos_theta * B_vec / B_abs
        e_x = e_x / np.sqrt(sum(e_x ** 2))
        e_z = B_vec / B_abs
        # e_y = e_z x e_x
        e_y = np.zeros(3)
        e_y[0] = e_z[1] * e_x[2] - e_z[2] * e_x[1]
        e_y[1] = e_z[2] * e_x[0] - e_z[0] * e_x[2]
        e_y[2] = e_z[0] * e_x[1] - e_z[1] * e_x[0]
        e_y = e_y / np.sqrt(np.sum(e_y ** 2))  # not necessary because e_x and e_z perpendicular
        if(debug):
            print("e_x", e_x)
            print("e_y", e_y)
            print("e_z", e_z)
            print("e_x . e_y", np.sum(e_x * e_y))
            print("e_x . e_z", np.sum(e_x * e_y))
            print("e_y . e_z", np.sum(e_y * e_z))
            print("e_x.N_vec", np.sum(e_x * N_vec_norm))
            print("e_y.N_vec", np.sum(e_y * N_vec_norm))
            print("e_z.N_vec", np.sum(e_z * N_vec_norm))
        pol_vector_lab = pol_vec_real[0] * e_x + \
                            pol_vec_real[1] * e_y + \
                            pol_vec_real[2] * e_z
        if(debug):
            print("Polvec in laboratory frame before removal of longitudinal component", pol_vector_lab)
            print("Dot product N_vec pol_vec in lab frame", np.sum(N_vec * pol_vector_lab))
        # Now remove portion that points along N_vec
        pol_vector_lab = pol_vector_lab - N_vec_norm * np.sum(N_vec_norm * pol_vector_lab)
        pol_vector_lab = pol_vector_lab / np.sqrt(np.sum(np.abs(pol_vector_lab) ** 2))
        return pol_vector_lab


    def get_filter_transmittance(self, omega, X, Y, mode, x_vec, N_vec, B_vec, x_launch):
        # Do not USE - WRONG
        # The wave vector N in the plasma is rotated by the lens system so that the rotated wave vector N' is perpendicular to the polarization filter
        # Four steps are required:
        #    1. Calculate polarization vector e, assuming e_y is purely imaginary.
        #    2. Determine N' and e'.
        #    3. Express e' in the coordinate system spanned by N', the filter vector F and N'xF
        #    4. Apply the Jones matrix
        # in the same plane as the polarizer.
        debug = False
        cos_theta = np.sum(N_vec * B_vec) / np.sqrt(np.sum(N_vec ** 2)) / np.sqrt(np.sum(B_vec ** 2))
        sin_theta = np.sin(np.arccos(cos_theta))
        N_abs, pol_vec = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)
        pol_vec[:] /= np.sqrt(np.sum(np.abs(pol_vec) ** 2))  # Normalized to 1 for calculation of X/O-mode fraction
        if(N_abs == 0.e0):
            return 0.e0
        N_perp = N_abs * sin_theta
        pol_vec_real = np.zeros(3, dtype=np.complex)
        pol_vec_real[0] = np.real(pol_vec[0])
        pol_vec_real[1] = np.imag(pol_vec[1])
        pol_vec_real[2] = np.real(pol_vec[2])
        if(np.any(pol_vec_real != pol_vec_real)):
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("N_perp", N_perp)
            print("omega", omega)
            return None
        if(debug):
            print("pol_vec", pol_vec_real)
        # First we rotate the polarization vector into the carthesian coordinate system of the machine
        # Now calculate rotation from Cartesian to reference frame of the wave and the polarization vector.
        # The polarization coefficients are given in a way so that
        # vec(N) = (vec(e_x) * sin(theta) + vec(e_z) * cos(theta)) * N_abs
        # vec(B) = B e_z
        # Hence, the unit vectors are given by:
        # vec(e_x) = vec(N) / N_abs - cos(theta) * vec(B) / B_abs
        # vec(e_y) = vec(e_x) x vec(e_z)
        # vec(e_z) = vec(B) / B_abs
        # The rotation is then given according to https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotations_and_motions
        N_abs_ray = np.sqrt(sum(N_vec ** 2))
        N_vec_norm = N_vec / N_abs_ray
        B_abs = np.sqrt(np.sum(B_vec ** 2))
        if(debug):
            print("omega", omega)
            print("X", X)
            print("Y", Y)
            print("x_vec", x_vec)
            print("N_vec", N_vec)
            print("B_vec", B_vec)
            print("theta", np.rad2deg(np.arccos(cos_theta)))
        if(mode > 0 and debug):
            print("X-mode")
        if(mode < 0 and debug):
            print("O-mode")
        # N_vec already points towards the antenna when its is copied into svec
        e_x = N_vec_norm - cos_theta * B_vec / B_abs
        e_x = e_x / np.sqrt(sum(e_x ** 2))
        e_z = B_vec / B_abs
        # e_y = e_z x e_x
        e_y = np.zeros(3)
        e_y[0] = e_z[1] * e_x[2] - e_z[2] * e_x[1]
        e_y[1] = e_z[2] * e_x[0] - e_z[0] * e_x[2]
        e_y[2] = e_z[0] * e_x[1] - e_z[1] * e_x[0]
        e_y = e_y / np.sqrt(np.sum(e_y ** 2))  # not necessary because e_x and e_z perpendicular
        if(debug):
            print("e_x", e_x)
            print("e_y", e_y)
            print("e_z", e_z)
            print("e_x . e_y", np.sum(e_x * e_y))
            print("e_x . e_z", np.sum(e_x * e_y))
            print("e_y . e_z", np.sum(e_y * e_z))
            print("e_x.N_vec", np.sum(e_x * N_vec_norm))
            print("e_y.N_vec", np.sum(e_y * N_vec_norm))
            print("e_z.N_vec", np.sum(e_z * N_vec_norm))
        pol_vector_lab = pol_vec_real[0] * e_x + \
                            pol_vec_real[1] * e_y + \
                            pol_vec_real[2] * e_z
        if(debug):
            print("Polvec in laboratory frame before removal of longitudinal component", pol_vector_lab)
            print("Dot product N_vec pol_vec in lab frame", np.sum(N_vec * pol_vector_lab))
        # Now remove portion that points along N_vec
        pol_vector_lab = pol_vector_lab - N_vec_norm * np.sum(N_vec_norm * pol_vector_lab)
        pol_vector_lab = pol_vector_lab / np.sqrt(np.sum(np.abs(pol_vector_lab) ** 2))
        if(debug):
            print("Polvec in laboratory frame after removal of longitudinal component", pol_vector_lab)
            print("Dot product N_vec pol_vec in lab frame", np.sum(N_vec * pol_vector_lab))
        # Next the rotation of k by the quasi-optical system:
        # Normalized vector perpendicular to the filter
        N_filter = np.zeros(3)
        phi = np.arctan2(x_launch[1], x_launch[0])
        N_filter[0] = np.cos(phi)
        N_filter[1] = np.sin(phi)
        # We do not want a z component here
        N_filter[2] = 0.e0
        N_filter = N_filter / np.sqrt(np.sum(N_filter ** 2))
        if(debug):
            print("N_vec norm", N_vec_norm)
        if(debug):
            print("N_filter", N_filter)
        # Rotation matrix around angle sigma with axis N_filter x N_vec
        sigma = -np.arccos(np.sum(N_vec_norm * N_filter))
        if(debug):
            print("Sigma [deg.]", sigma / np.pi * 180.e0)
        norm_vec_N_rot_plane = np.zeros(3)
        # Rotate the polarization vector in the plane spanned by N_filter and N_vec_norm by the angle sigma
        norm_vec_N_rot_plane[0] = N_filter[1] * N_vec_norm[2] - N_filter[2] * N_vec_norm[1]
        norm_vec_N_rot_plane[1] = N_filter[2] * N_vec_norm[0] - N_filter[0] * N_vec_norm[2]
        norm_vec_N_rot_plane[2] = N_filter[0] * N_vec_norm[1] - N_filter[1] * N_vec_norm[0]
        norm_vec_N_rot_plane = norm_vec_N_rot_plane / np.sqrt(np.sum(norm_vec_N_rot_plane ** 2))
        if(debug):
            print("Axis of rotation", norm_vec_N_rot_plane)
        # First index selects column second index row
        # Compare my rotation against the one from stackoverflow
        pol_vec_perp = rotate_vec_around_axis(pol_vector_lab, norm_vec_N_rot_plane, sigma)
        N_vec_perp_test = rotate_vec_around_axis(N_vec_norm, norm_vec_N_rot_plane, sigma)
        if(debug):
            print("pol vec perp filter", pol_vec_perp)
        if(debug):
            print("dot product rotated N_vec and N_filter vec - should be one", np.sum(N_vec_perp_test * N_filter))
        if(debug):
            print("dot product rotated polarization vector and N_filter vec - should be 0", np.sum(pol_vec_perp * N_filter))
        R_vec = np.zeros(3)
        R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        R_vec[1] = func_calc_phi(x_vec[0], x_vec[1])
        R_vec[2] = x_vec[2]
        cos_phi = np.cos(R_vec[1])
        sin_phi = np.sin(R_vec[1])
        Jones_vector = np.zeros(2, dtype=np.complex)
        Jones_vector[0] = sin_phi * pol_vec_perp[0] + cos_phi * pol_vec_perp[1]
        Jones_vector[1] = pol_vec_perp[2]
        if(debug):
            print("Jones_vector:" , Jones_vector)
        filter_mat = np.zeros((2, 2))
        filter_mat[:, :] = 0.e0
        filter_mat[1, 1] = 1.e0
        filtered_Jones_vector = np.zeros(2, dtype=np.complex)
        for i in range(2):
            filtered_Jones_vector[i] = np.sum(filter_mat[i, :] * Jones_vector[:])
        if(debug):
            print("Filtered Jones_vector:" , filtered_Jones_vector)
        get_filter_transmittance = np.sum(np.abs(filtered_Jones_vector) ** 2)
        if(debug):
            print("Transmittance", get_filter_transmittance)
        if(get_filter_transmittance != get_filter_transmittance or get_filter_transmittance < 0.0 or \
           get_filter_transmittance > 1.e0):
            print("pol_vec in original coordinates", pol_vec)
            print("pol_vec in carth. coordinates", pol_vec)
            print("phi", R_vec[2])
            print("cos_theta", cos_theta)
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("Transmittance", get_filter_transmittance)
        return get_filter_transmittance

    def get_filter_transmittance_new(self, omega, X, Y, mode, x_vec, N_vec, B_vec):
        # Do not USE - WRONG
        # Calculates the filtered intensity for a polarization filter aligned with e_phi.
        # Seven steps are required:
        #    1. Calculate polarization vector e in Stix coordinate system
        #    2. Express the polarization vector in the carthesian coordinate system of the ray.
        #    3. The opitcal system of the diagnostic rotates the normalized wave vector N and the polarization vector
        #       such that N -> N' with N' perpendicular to the filter. Hence, step 3 is to determine N' and e'.
        #    4. Express e' in the coordinate system spanned by the passing and the blocking direction of the filter, to determine the Jones Vector.
        #    5. Apply the linear polarizer on the Jones vector.
        #    6. Calculate the Jones vector behing the filter.
        #    7. Calculate the passing intensity.
        debug = False
        cos_theta = np.sum(N_vec * B_vec) / np.sqrt(np.sum(N_vec ** 2)) / np.sqrt(np.sum(B_vec ** 2))
        sin_theta = np.sin(np.arccos(cos_theta))
        if(debug):
            print("omega", omega)
            print("X", X)
            print("Y", Y)
            print("x_vec", x_vec)
            print("N_vec", N_vec)
            print("B_vec", B_vec)
            print("theta", np.rad2deg(np.arccos(cos_theta)))
        if(mode > 0 and debug):
            print("X-mode")
        if(mode < 0 and debug):
            print("O-mode")
        # Passing direction of the polarizer
        # For an polarizer set to X-mode this is the z-direction of the torus
        f_pass = np.array([0.0, 0.0, 1.0])
        # The filter blocks in the toroidal direction, which has to be expressed in the ray coordinate system
        R_vec = np.zeros(3)
        R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        R_vec[1] = func_calc_phi(x_vec[0], x_vec[1])
        R_vec[2] = x_vec[2]
        f_block = np.array([-np.sin(R_vec[1]), np.cos(R_vec[1]), 0.0])
        # This is the direction of -N'
        f_perp = np.cross(f_pass, f_block)
        if(debug):
            print("Passing direction of the filter", f_pass)
            print("Blocking direction of the filter", f_block)
            print("Vector perpendicular to the plane of the filter", f_perp)
        # Step 1:
        N_abs, pol_vec = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)
        pol_vec[:] /= np.linalg.norm(pol_vec)  # Normalized to 1 for calculation of X/O-mode fraction
        if(N_abs == 0.e0):
            return 0.e0
        N_perp = N_abs * sin_theta
        if(np.any(pol_vec != pol_vec)):
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("N_perp", N_perp)
            print("omega", omega)
            return None
        if(debug):
            print("pol_vec", pol_vec)
        # Step 2:
        # First calculate the unit vector in which the e is expressed in
        N_abs_ray = np.sqrt(sum(N_vec ** 2))
        N_vec_norm = N_vec / N_abs_ray
        B_abs = np.sqrt(np.sum(B_vec ** 2))
        e_x = N_vec_norm - cos_theta * B_vec / B_abs  # Propagation in x, z plane
        e_x = e_x / np.sqrt(sum(e_x ** 2))
        e_z = B_vec / B_abs  # Magnetic field direction
        e_y = np.cross(e_x, e_z)  # NxB -> e_x, e_y, e_z form right handed coordinate system
        if(debug):
            print("e_x", e_x)
            print("e_y", e_y)
            print("e_z", e_z)
            print("e_x . e_y", np.sum(e_x * e_y))
            print("e_x . e_z", np.sum(e_x * e_y))
            print("e_y . e_z", np.sum(e_y * e_z))
            print("e_x.N_vec", np.sum(e_x * N_vec_norm))
            print("e_y.N_vec", np.sum(e_y * N_vec_norm))
            print("e_z.N_vec", np.sum(e_z * N_vec_norm))
        # Polarization vector in the lab coordinate system
        pol_vector_lab = pol_vec[0] * e_x + \
                         pol_vec[1] * e_y + \
                         pol_vec[2] * e_z
        # Now remove longitudinal because it cannot exist in vacuum
        if(debug):
            print("Longitudinal component", np.dot(N_vec_norm, pol_vector_lab))
            print("Polvec in laboratory frame before removal of longitudinal component", pol_vector_lab)
            print("Normalization", np.linalg.norm(pol_vector_lab))
            print("Dot product N_vec pol_vec in lab frame", np.sum(N_vec_norm * pol_vector_lab))
        # Now remove portion that points along N_vec
        pol_vector_lab = pol_vector_lab - N_vec_norm * np.sum(N_vec_norm * pol_vector_lab)
        pol_vector_lab = pol_vector_lab / np.sqrt(np.sum(pol_vector_lab ** 2))
        pol_vector_lab = pol_vector_lab - N_vec_norm * np.dot(N_vec_norm, pol_vector_lab)
        # Renormalize - this is incorrect, because the longitudinal component does not redistribute like this onto the to transverse component
        # At the moment this is the best we can do, unless we just assume X << 1 from the start.
        # In general this should not be a large error as the longitudinal component should be very small.
        pol_vector_lab = pol_vector_lab / np.linalg.norm(pol_vector_lab)
        if(debug):
            print("Transversal Polvec in laboratory frame", pol_vector_lab)
            print("Dot product N_vec and transversal pol_vec in lab frame", np.dot(N_vec_norm, pol_vector_lab))
        # Step 3. Rotate e_lab and N_vec according to the lense system of the ECE diagnostic
        # This step assumes that the lense system rotates N such that N' is perpendicular to the filter
        sigma = -np.arccos(np.sum(N_vec_norm * -f_perp))
        if(debug):
            print("Sigma [deg.]", sigma / np.pi * 180.e0)
        norm_vec_N_rot_plane = np.cross(-f_perp, N_vec_norm)
        if(debug):
            print("Axis of rotation", norm_vec_N_rot_plane)
        pol_vec_perp = rotate_vec_around_axis(pol_vector_lab, norm_vec_N_rot_plane, sigma)
        N_vec_perp_test = rotate_vec_around_axis(N_vec_norm, norm_vec_N_rot_plane, sigma)
        if(debug):
            print("pol vec perp filter", pol_vec_perp)
            print("Normalization", np.linalg.norm(pol_vec_perp))
        if(debug):
            print("dot product rotated N_vec and N_filter vec - should be one", np.sum(N_vec_perp_test * -f_perp))
        if(debug):
            print("dot product rotated polarization vector and N_filter vec - should be 0", np.sum(pol_vec_perp * -f_perp))
        Jones_vector = np.zeros(2, dtype=np.complex)
        # Express the polarization vector in terms of the two directions of the polarizer
        Jones_vector[0] = np.dot(f_pass, pol_vec_perp)
        Jones_vector[1] = np.dot(f_block, pol_vec_perp)
        if(debug):
            print("Jones_vector:" , Jones_vector)
            print("Normalization", np.linalg.norm(Jones_vector))
        # Polarization matrix for a linear polarizer which is passing for the x-direction.
        filter_mat = np.zeros((2, 2))
        filter_mat[0, 0] = 1.e0
        filtered_Jones_vector = np.zeros(2, np.complex)
        for i in range(2):
            filtered_Jones_vector[i] = np.sum(filter_mat[i, :] * Jones_vector[:])
        if(debug):
            print("Filtered Jones_vector:" , filtered_Jones_vector)
        transmittance = np.sum(np.abs(filtered_Jones_vector) ** 2)
        if(debug):
            print("Transmittance", transmittance)
        if(transmittance != transmittance or transmittance < 0.0 or \
           transmittance > 1.e0):
            print("pol_vec in original coordinates", pol_vec)
            print("pol_vec in carth. coordinates", pol_vec)
            print("phi", R_vec[2])
            print("cos_theta", cos_theta)
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("Transmittance", transmittance)
        return transmittance

    def get_filter_transmittance_correct_filter(self, omega, X, Y, mode, x_vec, N_vec, B_vec, x_launch):
        # Calculates the filtered intensity for a polarization filter aligned with e_phi.
        # Seven steps are required:
        #    1. Calculate polarization vector e in Stix coordinate system
        #    2. Express the polarization vector in the carthesian coordinate system of the ray.
        #    3. The opitcal system of the diagnostic rotates the normalized wave vector N and the polarization vector
        #       such that N -> N' with N' perpendicular to the filter. Hence, step 3 is to determine N' and e'.
        #    4. Express e' in the coordinate system spanned by the passing and the blocking direction of the filter, to determine the Jones Vector.
        #    5. Apply the linear polarizer on the Jones vector.
        #    6. Calculate the Jones vector behing the filter.
        #    7. Calculate the passing intensity.
        debug = False
        cos_theta = np.sum(N_vec * B_vec) / np.sqrt(np.sum(N_vec ** 2)) / np.sqrt(np.sum(B_vec ** 2))
        sin_theta = np.sin(np.arccos(cos_theta))
        if(debug):
            print("omega", omega)
            print("X", X)
            print("Y", Y)
            print("x_vec", x_vec)
            print("N_vec", N_vec)
            print("B_vec", B_vec)
            print("theta", np.rad2deg(np.arccos(cos_theta)))
        if(mode > 0 and debug):
            print("X-mode")
        if(mode < 0 and debug):
            print("O-mode")
        # Passing direction of the polarizer
        # For an polarizer set to X-mode this is the z-direction of the torus
        f_pass = np.array([0.0, 0.0, 1.0])
        # The filter blocks in the toroidal direction, which has to be expressed in the ray coordinate system
        R_vec = np.zeros(3)
        R_vec[0] = np.sqrt(x_launch[0] ** 2 + x_launch[1] ** 2)
        R_vec[1] = func_calc_phi(x_launch[0], x_launch[1])
        R_vec[2] = x_vec[2]
        f_block = np.array([-np.sin(R_vec[1]), np.cos(R_vec[1]), 0.0])
        # This is the direction of -N'
        f_perp = np.cross(f_pass, f_block)
        if(debug):
            print("Passing direction of the filter", f_pass)
            print("Blocking direction of the filter", f_block)
            print("Vector perpendicular to the plane of the filter", f_perp)
        # Step 1:
        N_abs, pol_vec = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)
        pol_vec[:] /= np.linalg.norm(pol_vec)  # Normalized to 1 for calculation of X/O-mode fraction
        if(N_abs == 0.e0):
            return 0.e0
        N_perp = N_abs * sin_theta
        if(np.any(pol_vec != pol_vec)):
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("N_perp", N_perp)
            print("omega", omega)
            return None
        if(debug):
            print("pol_vec", pol_vec)
        # Step 2:
        # First calculate the unit vector in which the e is expressed in
        N_abs_ray = np.sqrt(sum(N_vec ** 2))
        N_vec_norm = N_vec / N_abs_ray
        B_abs = np.sqrt(np.sum(B_vec ** 2))
        e_x = N_vec_norm - cos_theta * B_vec / B_abs  # Propagation in x, z plane
        e_x = e_x / np.sqrt(sum(e_x ** 2))
        e_z = B_vec / B_abs  # Magnetic field direction
        e_y = np.cross(e_x, e_z)  # NxB -> e_x, e_y, e_z form right handed coordinate system
        if(debug):
            print("e_x", e_x)
            print("e_y", e_y)
            print("e_z", e_z)
            print("e_x . e_y", np.sum(e_x * e_y))
            print("e_x . e_z", np.sum(e_x * e_y))
            print("e_y . e_z", np.sum(e_y * e_z))
            print("e_x.N_vec", np.sum(e_x * N_vec_norm))
            print("e_y.N_vec", np.sum(e_y * N_vec_norm))
            print("e_z.N_vec", np.sum(e_z * N_vec_norm))
        # Polarization vector in the lab coordinate system
        pol_vector_lab = pol_vec[0] * e_x + \
                         pol_vec[1] * e_y + \
                         pol_vec[2] * e_z
        # Now remove longitudinal because it cannot exist in vacuum
        if(debug):
            print("Longitudinal component", np.dot(N_vec_norm, pol_vector_lab))
            print("Polvec in laboratory frame before removal of longitudinal component", pol_vector_lab)
            print("Normalization", np.linalg.norm(pol_vector_lab))
            print("Dot product N_vec pol_vec in lab frame", np.sum(N_vec_norm * pol_vector_lab))
        # Now remove portion that points along N_vec
        pol_vector_lab = pol_vector_lab - N_vec_norm * np.sum(N_vec_norm * pol_vector_lab)
        pol_vector_lab = pol_vector_lab / np.sqrt(np.sum(pol_vector_lab ** 2))
        pol_vector_lab = pol_vector_lab - N_vec_norm * np.dot(N_vec_norm, pol_vector_lab)
        # Renormalize - this is incorrect, because the longitudinal component does not redistribute like this onto the to transverse component
        # At the moment this is the best we can do, unless we just assume X << 1 from the start.
        # In general this should not be a large error as the longitudinal component should be very small.
        pol_vector_lab = pol_vector_lab / np.linalg.norm(pol_vector_lab)
        if(debug):
            print("Transversal Polvec in laboratory frame", pol_vector_lab)
            print("Dot product N_vec and transversal pol_vec in lab frame", np.dot(N_vec_norm, pol_vector_lab))
        # Step 3. Rotate e_lab and N_vec according to the lense system of the ECE diagnostic
        # This step assumes that the lense system rotates N such that N' is perpendicular to the filter
        sigma = -np.arccos(np.sum(N_vec_norm * -f_perp))
        if(debug):
            print("Sigma [deg.]", sigma / np.pi * 180.e0)
        norm_vec_N_rot_plane = np.cross(-f_perp, N_vec_norm)
        if(debug):
            print("Axis of rotation", norm_vec_N_rot_plane)
        pol_vec_perp = rotate_vec_around_axis(pol_vector_lab, norm_vec_N_rot_plane, sigma)
        N_vec_perp_test = rotate_vec_around_axis(N_vec_norm, norm_vec_N_rot_plane, sigma)
        if(debug):
            print("pol vec perp filter", pol_vec_perp)
            print("Normalization", np.linalg.norm(pol_vec_perp))
        if(debug):
            print("dot product rotated N_vec and N_filter vec - should be one", np.sum(N_vec_perp_test * -f_perp))
        if(debug):
            print("dot product rotated polarization vector and N_filter vec - should be 0", np.sum(pol_vec_perp * -f_perp))
        Jones_vector = np.zeros(2, dtype=np.complex)
        # Express the polarization vector in terms of the two directions of the polarizer
        Jones_vector[0] = np.dot(f_pass, pol_vec_perp)
        Jones_vector[1] = np.dot(f_block, pol_vec_perp)
        if(debug):
            print("Jones_vector:" , Jones_vector)
            print("Normalization", np.linalg.norm(Jones_vector))
        # Polarization matrix for a linear polarizer which is passing for the x-direction.
        filter_mat = np.zeros((2, 2))
        filter_mat[0, 0] = 1.e0
        filtered_Jones_vector = np.zeros(2, np.complex)
        for i in range(2):
            filtered_Jones_vector[i] = np.sum(filter_mat[i, :] * Jones_vector[:])
        if(debug):
            print("Filtered Jones_vector:" , filtered_Jones_vector)
        transmittance = np.sum(np.abs(filtered_Jones_vector) ** 2)
        if(debug):
            print("Transmittance", transmittance)
        if(transmittance != transmittance or transmittance < 0.0 or \
           transmittance > 1.e0):
            print("pol_vec in original coordinates", pol_vec)
            print("pol_vec in carth. coordinates", pol_vec)
            print("phi", R_vec[2])
            print("cos_theta", cos_theta)
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("Transmittance", transmittance)
        return transmittance

    def get_filter_transmittance_reverse(self, omega, X, Y, mode, x_vec, N_vec, B_vec, polarizer_angle=0.0):
        # Do not USE - WRONG
        # Calculates the filtered intensity for a polarization filter aligned with e_phi.
        # In this routine the passing vector of the polarizer f_p is translated into the Stix reference frame f''_p
        # The transmittance is then given by abs(dot(f''p, e))^2.
        debug = False
        cos_theta = np.sum(N_vec * B_vec) / np.sqrt(np.sum(N_vec ** 2)) / np.sqrt(np.sum(B_vec ** 2))
        sin_theta = np.sin(np.arccos(cos_theta))
        if(debug):
            print("omega", omega)
            print("X", X)
            print("Y", Y)
            print("x_vec", x_vec)
            print("N_vec", N_vec)
            print("B_vec", B_vec)
            print("theta", np.rad2deg(np.arccos(cos_theta)))
        if(mode > 0 and debug):
            print("X-mode")
        if(mode < 0 and debug):
            print("O-mode")
        # Passing direction of the polarizer
        # For an polarizer set to X-mode this is the z-direction of the torus
        # Minus sign, because we want a right handed coordinate system, for which the coordinate f_pass x f_block points into the R direction
        f_pass = -np.array([0.0, 0.0, 1.0])
        if(np.dot(x_vec / np.linalg.norm(x_vec), N_vec / np.linalg.norm(N_vec)) < 0.e0):
            # ECRH case - wave travels from outside the torus into the plasma -  minus R direction for x
            # Polarizer should then also point outwards and not inwards
            f_pass *= -1.0
        # The filter blocks in the toroidal direction, which has to be expressed in the ray coordinate system
        R_vec = np.zeros(3)
        R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        R_vec[1] = func_calc_phi(x_vec[0], x_vec[1])
        R_vec[2] = x_vec[2]
        f_block = np.array([-np.sin(R_vec[1]), np.cos(R_vec[1]), 0.0])
        # This is the direction of -N'
        f_perp = np.cross(f_pass, f_block)
        # Rotate polarizer according to polarizer_angle
        f_pass = rotate_vec_around_axis(f_pass, f_perp, polarizer_angle)
        f_block = rotate_vec_around_axis(f_pass, f_block, polarizer_angle)
        # Now we need to rotate f_pass by the angle between f_perp and N_vec
        if(debug):
            print("Passing direction of the filter", f_pass)
            print("Blocking direction of the filter", f_block)
            print("Vector perpendicular to the plane of the filter", f_perp)
        # First we need to calculate N_vec
        N_abs, pol_vec = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)
        pol_vec[:] /= np.linalg.norm(pol_vec)  # Normalized to 1 for calculation of X/O-mode fraction
        if(N_abs == 0.e0):
            return 0.e0
        N_perp = N_abs * sin_theta
        if(np.any(pol_vec != pol_vec)):
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("N_perp", N_perp)
            print("omega", omega)
            return None
        if(debug):
            print("pol_vec", pol_vec)
        # No rotate f_pass so that it is perpendicular to N_vec
        N_abs_ray = np.sqrt(sum(N_vec ** 2))
        N_vec_norm = N_vec / N_abs_ray
        sigma = -np.arccos(np.sum(N_vec_norm * f_perp))
        if(debug):
            print("Sigma [deg.]", sigma / np.pi * 180.e0)
        norm_vec_N_rot_plane = np.cross(N_vec_norm, f_perp)
        f_pass_perp = rotate_vec_around_axis(f_pass, norm_vec_N_rot_plane, sigma)
        f_perp_perp_test = rotate_vec_around_axis(f_perp, norm_vec_N_rot_plane, sigma)
        if(debug):
            print("f_pass_perp", f_pass_perp)
            print("f_perp_test dot N_vec - should be 1", np.dot(f_perp_perp_test, N_vec_norm))
        # Now express f_pass in the Stix coordinate system
        # First express the unit vectors of the Stix coordiante system in carthesian coordinates
        B_abs = np.sqrt(np.sum(B_vec ** 2))
        e_x = N_vec_norm - cos_theta * B_vec / B_abs  # Propagation in x, z plane
        e_x = e_x / np.sqrt(sum(e_x ** 2))
        e_z = B_vec / B_abs  # Magnetic field direction
        e_y = np.cross(e_x, e_z)  # NxB -> e_x, e_y, e_z form right handed coordinate system
        if(debug):
            print("e_x", e_x)
            print("e_y", e_y)
            print("e_z", e_z)
            print("e_x . e_y", np.sum(e_x * e_y))
            print("e_x . e_z", np.sum(e_x * e_y))
            print("e_y . e_z", np.sum(e_y * e_z))
            print("e_x.N_vec", np.sum(e_x * N_vec_norm))
            print("e_y.N_vec", np.sum(e_y * N_vec_norm))
            print("e_z.N_vec", np.sum(e_z * N_vec_norm))
        # Now express f_pass in the stix reference system
        f_pass_stix = np.zeros(3)
        f_pass_stix[0] = np.dot(e_x, f_pass_perp)
        f_pass_stix[1] = np.dot(e_y, f_pass_perp)
        f_pass_stix[2] = np.dot(e_z, f_pass_perp)
        if(debug):
            print("f_pass_stix", f_pass_stix)
        # Now calcualte projection of polarization vector on f_pass
        transmittance = np.abs(np.vdot(f_pass_stix, pol_vec)) ** 2
        if(debug):
            print("Transmittance", transmittance)
        if(transmittance != transmittance or transmittance < 0.0 or \
           transmittance > 1.e0):
            print("pol_vec in original coordinates", pol_vec)
            print("pol_vec in carth. coordinates", pol_vec)
            print("phi", R_vec[2])
            print("cos_theta", cos_theta)
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("Transmittance", transmittance)
        return transmittance

    def get_filter_transmittance_reverse_correct_filter(self, omega, X, Y, mode, x_vec, N_vec, B_vec, x_launch, polarizer_angle=0.0):
        # Calculates the filtered intensity for a polarization filter aligned with e_phi.
        # In this routine the passing vector of the polarizer f_p is translated into the Stix reference frame f''_p
        # The transmittance is then given by abs(dot(f'', e))^2.
        # Allows for titled polarizer
        debug = False
        cos_theta = np.sum(N_vec * B_vec) / np.sqrt(np.sum(N_vec ** 2)) / np.sqrt(np.sum(B_vec ** 2))
        sin_theta = np.sin(np.arccos(cos_theta))
        if(debug):
            print("omega", omega)
            print("X", X)
            print("Y", Y)
            print("x_vec", x_vec)
            print("N_vec", N_vec)
            print("B_vec", B_vec)
            print("theta", np.rad2deg(np.arccos(cos_theta)))
        if(mode > 0 and debug):
            print("X-mode")
        if(mode < 0 and debug):
            print("O-mode")
        # Passing direction of the polarizer
        # For an polarizer set to X-mode this is the z-direction of the torus
        # Minus sign, because we want a right handed coordinate system, for which the coordinate f_pass x f_block points into the R direction
        f_pass = -np.array([0.0, 0.0, 1.0])
        if(np.dot(x_vec / np.linalg.norm(x_vec), N_vec / np.linalg.norm(N_vec)) < 0.e0):
            # ECRH case - wave travels from outside the torus into the plasma -  minus R direction for x
            # Polarizer should then also point outwards and not inwards
            f_pass *= -1.0
        # The filter blocks in the toroidal direction, which has to be expressed in the ray coordinate system
        R_vec = np.zeros(3)
        R_vec[0] = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        R_vec[1] = func_calc_phi(x_vec[0], x_vec[1])
        R_vec[2] = x_vec[2]
        f_block = np.array([-np.sin(R_vec[1]), np.cos(R_vec[1]), 0.0])
        # This is the direction of -N'
        f_perp = np.cross(f_pass, f_block)
        # Rotate polarizer according to polarizer_angle
        f_pass = rotate_vec_around_axis(f_pass, f_perp, polarizer_angle)
        f_block = rotate_vec_around_axis(f_pass, f_block, polarizer_angle)
        # Now we need to rotate f_pass by the angle between f_perp and N_vec
        if(debug):
            print("Passing direction of the filter", f_pass)
            print("Blocking direction of the filter", f_block)
            print("Vector perpendicular to the plane of the filter", f_perp)
        # First we need to calculate N_vec
        N_abs, pol_vec = N_with_pol_vec(X, Y, sin_theta, cos_theta, mode)
        pol_vec[:] /= np.linalg.norm(pol_vec)  # Normalized to 1 for calculation of X/O-mode fraction
        if(N_abs == 0.e0):
            return 0.e0
        N_perp = N_abs * sin_theta
        if(np.any(pol_vec != pol_vec)):
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("N_perp", N_perp)
            print("omega", omega)
            return None
        if(debug):
            print("pol_vec", pol_vec)
        # No rotate f_pass so that it is perpendicular to N_vec
        N_abs_ray = np.sqrt(sum(N_vec ** 2))
        N_vec_norm = N_vec / N_abs_ray
        sigma = -np.arccos(np.sum(N_vec_norm * f_perp))
        if(debug):
            print("Sigma [deg.]", sigma / np.pi * 180.e0)
        norm_vec_N_rot_plane = np.cross(N_vec_norm, f_perp)
        f_pass_perp = rotate_vec_around_axis(f_pass, norm_vec_N_rot_plane, sigma)
        f_perp_perp_test = rotate_vec_around_axis(f_perp, norm_vec_N_rot_plane, sigma)
        if(debug):
            print("f_pass_perp", f_pass_perp)
            print("f_perp_test dot N_vec - should be 1", np.dot(f_perp_perp_test, N_vec_norm))
        # Now express f_pass in the Stix coordinate system
        # First express the unit vectors of the Stix coordiante system in carthesian coordinates
        B_abs = np.sqrt(np.sum(B_vec ** 2))
        e_x = N_vec_norm - cos_theta * B_vec / B_abs  # Propagation in x, z plane
        e_x = e_x / np.sqrt(sum(e_x ** 2))
        e_z = B_vec / B_abs  # Magnetic field direction
        e_y = np.cross(e_x, e_z)  # NxB -> e_x, e_y, e_z form right handed coordinate system
        if(debug):
            print("e_x", e_x)
            print("e_y", e_y)
            print("e_z", e_z)
            print("e_x . e_y", np.sum(e_x * e_y))
            print("e_x . e_z", np.sum(e_x * e_y))
            print("e_y . e_z", np.sum(e_y * e_z))
            print("e_x.N_vec", np.sum(e_x * N_vec_norm))
            print("e_y.N_vec", np.sum(e_y * N_vec_norm))
            print("e_z.N_vec", np.sum(e_z * N_vec_norm))
        # Now express f_pass in the stix reference system
        f_pass_stix = np.zeros(3)
        f_pass_stix[0] = np.dot(e_x, f_pass_perp)
        f_pass_stix[1] = np.dot(e_y, f_pass_perp)
        f_pass_stix[2] = np.dot(e_z, f_pass_perp)
        if(debug):
            print("f_pass_stix", f_pass_stix)
        # Now calcualte projection of polarization vector on f_pass
        transmittance = np.abs(np.vdot(f_pass_stix, pol_vec)) ** 2
        if(debug):
            print("Transmittance", transmittance)
        if(transmittance != transmittance or transmittance < 0.0 or \
           transmittance > 1.e0):
            print("pol_vec in original coordinates", pol_vec)
            print("pol_vec in carth. coordinates", pol_vec)
            print("phi", R_vec[2])
            print("cos_theta", cos_theta)
            print("X", X)
            print("Y", Y)
            print("N_abs", N_abs)
            print("Transmittance", transmittance)
        return transmittance

    def em_Hutch_approx(self, svec, omega):
        # Fast routine that gives an upper limit for the emissivity at a given point
        omega_c = svec.freq_2X * np.pi
        omega_p = e0 * np.sqrt(svec.ne / (eps0 * mass_e))
        omega_bar = omega / omega_c
        j = 0.e0
        c_abs = 0.e0
        max_harmonic = 3
        for m_sum in range(2, max_harmonic + 1):
            if((float(m_sum) / omega_bar) ** 2 > 1.e0 - svec.cos_theta ** 2):
                j_m, c_abs_m = self.em_Hutch_approx_integral_nume(svec, omega, m_sum)
                j += j_m
                c_abs += c_abs_m
        if(j == 0.e0):
            return j
        j *= (cnst.e) ** 2.e0 * omega * svec.ne / (cnst.epsilon_0 * cnst.c)
        c_abs = c_abs * 2.e0 * np.pi ** 2 * omega_p ** 2 / (omega_c * cnst.c)
        return j, c_abs

    def em_Hutch_approx_integral_nume(self, svec, omega, m):
        # Fast routine that gives an upper limit for the emissivity at a given point
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        omega_bar = omega / (svec.freq_2X * np.pi)
        m_0 = np.sqrt(1.e0 - svec.cos_theta ** 2) * omega_bar
        t , weights = leggauss(120)
        u_par = 1.e0 / np.sqrt(1.e0 - svec.cos_theta ** 2) * (float(m) / m_0 * svec.cos_theta + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * t)
        gamma = m / omega_bar + svec.cos_theta * u_par
        u_perp = np.sqrt(gamma ** 2 - 1.e0 - u_par ** 2)
        fac = 1
        # (m - 1)!
        for i in range(2, m):
            fac *= i
        j = (u_perp / gamma / 2.e0) ** (2 * m)
        j = j * self.dist(u_par[i], u_perp[i], mu, svec) * (gamma ** 2) * weights
        j_sum = 0
        c_abs_sum = 0
        c_abs = j
        c_abs = c_abs * self.Rdist(u_par[i], u_perp[i], m / omega_bar, svec.cos_theta, mu, svec)
        for i in range(len(t)):
            j_sum += j(i)
            c_abs_sum += c_abs
        j_sum = j_sum * m ** (2 * (m - 1)) / fac ** 2 * svec.sin_theta ** (2 * (m - 1)) * (svec.cos_theta ** 2 + 1.e0)
        c_abs_sum = c_abs_sum * m ** (2 * (m - 1)) / fac ** 2 * svec.sin_theta ** (2 * (m - 1)) * (svec.cos_theta ** 2 + 1.e0)
        return j_sum, c_abs_sum

    def em_Hutch_along_res(self, svec, omega, m):
        # Implementation of Hutchinson emissivity - only use for small Te
        # NOT Valdiated!
        omega_c = svec.freq_2X * np.pi
        omega_bar = omega / omega_c
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        m_0 = np.sqrt(1.e0 - svec.cos_theta ** 2) * omega_bar
        if((float(m) / omega_bar) ** 2 < 1.e0 - svec.cos_theta ** 2):
            return [], [], [] , []
        m = 2
        u_par = 1.e0 / np.sqrt(1.e0 - svec.cos_theta ** 2) * (float(m) / m_0 * svec.cos_theta + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0) * self.t)
        gamma = m / omega_bar + svec.cos_theta * u_par
        u_perp_sq = gamma ** 2 - 1.e0 - u_par ** 2
        u_perp_sq[u_perp_sq <= 0] += 1.e-7
        fac = 1
        # (m - 1)!
        for i in range(2, m):
            fac *= i
        j = u_perp_sq ** m * (gamma * 2.e0) ** (-2 * m + 2) / 4.e0
        c_abs = j
        j = j * self.dist(u_par, np.sqrt(u_perp_sq), mu, svec)
        c_abs = c_abs * self.Rdist(u_par, np.sqrt(u_perp_sq), m / omega_bar, svec.cos_theta, mu, svec)
        j = j * m ** (2 * (m - 1)) / fac ** 2 * svec.sin_theta ** (2 * (m - 1)) * (svec.cos_theta ** 2 + 1.e0)
        c_abs = c_abs * m ** (2 * (m - 1)) / fac ** 2 * svec.sin_theta ** (2 * (m - 1)) * (svec.cos_theta ** 2 + 1.e0)
        j *= (cnst.e) ** 2.e0 * omega * svec.ne / (cnst.epsilon_0 * cnst.c)
        I_bb = omega ** 2 * cnst.m_e / (2.0 * np.pi)
        c_abs *= -(cnst.e) ** 2.e0 * omega * svec.ne / (cnst.epsilon_0 * cnst.c * I_bb)
        return u_par, np.sqrt(u_perp_sq), c_abs, j

    def get_u_par_max(self, svec, omega, m):
        # ???
        omega_c = svec.freq_2X * np.pi
        omega_bar = omega / omega_c
        mu = mass_e * c0 ** 2 / (e0 * svec.Te)
        m_0 = np.sqrt(1.e0 - svec.cos_theta ** 2) * omega_bar
        u_par1 = 1.e0 / np.sqrt(1.e0 - svec.cos_theta ** 2) * (float(m) / m_0 * svec.cos_theta - \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0))
        u_par2 = 1.e0 / np.sqrt(1.e0 - svec.cos_theta ** 2) * (float(m) / m_0 * svec.cos_theta + \
                           np.sqrt((float(m) / m_0) ** 2 - 1.e0))
        dist = u_par2 - u_par1
        u_par1 += dist * 1.e-5
        u_par2 -= dist * 1.e-5
        print("upar12", u_par1, u_par2)
        u_par_max = brentq(func_dj_du_par_no_gamma, u_par1, u_par2, args=[m / omega_bar, svec.cos_theta, mu])
        return u_par_max

def func_dj_du_par(u_par, args):
    # ???
    m_omega_bar = args[0]
    N_par = args[1]
    mu = args[2]
    return   (-1 + (-1 + N_par ** 2) * u_par ** 2 + 2 * N_par * u_par * m_omega_bar + m_omega_bar ** 2) * \
        (N_par ** 4 * u_par ** 3 * mu + 4 * u_par * m_omega_bar + N_par ** 3 * u_par ** 2 * (-2 + 3 * mu * m_omega_bar) - \
         N_par * (2 + mu * m_omega_bar + 2 * m_omega_bar ** 2 - mu * m_omega_bar ** 3 + u_par ** 2 * (-2 + mu * m_omega_bar)) - \
         N_par ** 2 * u_par * (4 * m_omega_bar + mu * (1 + u_par ** 2 - 3 * m_omega_bar ** 2)))

def func_dj_du_par_no_gamma(u_par, args):
    # ???
    m_omega_bar = args[0]
    N_par = args[1]
    mu = args[2]
    m = 2
    return -(N_par * mu) + (m * (-2 * u_par + 2 * N_par * (N_par * u_par + m_omega_bar))) / \
                (-1 - u_par ** 2 + (N_par * u_par + m_omega_bar) ** 2)

def func_calc_phi(x, y):
    # Differentiable arctan2
    if(y != 0.e0):
        return 2.e0 * np.arctan((np.sqrt(x ** 2 + y ** 2) - x) / y)
    elif(x > 0.e0):
        return 0.e0
    elif(x < 0.e0):
        return np.pi
    else:
        print("encountered atan(0/0)")
        return None

# test()
if __name__ == "__main__":
    theta = np.deg2rad(80)
    alb_obj = EmAbsAlb()
    omega = 205.e9 * 2 * np.pi
    omega_c = 70.e9 * np.pi * 2
    ne1 = 11.e19
    Te1 = 3.9e3
    ne2 = 8.3e19
    Te2 = 2.8e3
    svec1 = SVec(0.1, Te1, ne1, omega_c/ np.pi, theta)
    abs1, j1 = alb_obj.abs_Albajar(svec1, omega, 1)
    svec2 = SVec(0.1, Te2, ne2, omega_c/ np.pi, theta)
    abs2, j2 = alb_obj.abs_Albajar(svec2, omega, 1)
    print(abs1, abs2)
    print(3.8 * abs2/abs1)
    print(1 - np.exp(-3.8 * abs2/abs1))


#    N_abs = []
#    omega = 15.0e9 * np.pi
#    n_e = np.linspace(1.e17, 6.e18, 100)
#    svec = s_vec(1.2, 10.e0, 1.e18, 15.e9, np.deg2rad(theta))
#    Y = svec.freq_2X * np.pi * mass_e / cnst.m_e / omega
#    omega_p = e0 * np.sqrt(svec.ne / (eps0 * cnst.m_e))
#    X = omega_p ** 2 / omega ** 2
#    print(alb_obj.refr_index(svec, omega, -1))
#    omega_ps = np.zeros(len(n_e))
#    freqs = np.zeros(len(n_e))
#    freqs[:] = 15.5e9
#    for i in range(len(n_e)):
#        svec = s_vec(1.2, 10.e0, n_e[i], 15.e9, np.deg2rad(theta))
#        omega_ps[i] = e0 * np.sqrt(svec.ne / (eps0 * cnst.m_e))
#    plt.plot(n_e, omega_ps / np.pi / 2.e9, label="$\omega_\mathrm{p}$")
#    plt.plot(n_e, freqs / 1.e9, "--", label="$\omega$")
#    plt.show()
#    for theta in thetas:
#        svec = s_vec(1.2, 10.e0, 5.e18, 216.e9, np.deg2rad(theta))
#        N_abs.append(alb_obj.refr_index(svec, 288.e9, 1))
#    plt.plot(thetas, np.array(N_abs))
#    plt.show()
#    bornat = Bornatici_abs()
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111)
#    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111)
#    bornat.test_Dnestrovskii_recursion(ax1, ax2)
#    bornat.test_Dnestrovskii(ax1, ax2)
#    plt.show()
#    abs_Te()
