'''
Created on Jan 09, 2018

@author: sdenk
'''
from plotting_configuration import *
import numpy as np
import scipy.constants as cnst
from scipy.special import jn
from scipy.interpolate import InterpolatedUnivariateSpline


def phase_plot_Landau():
    t = np.linspace(0, np.pi, 200)
    zeros = np.zeros(t.shape)
    omegas = np.linspace(0, 1, 5)
    for omega in omegas:
        plt.plot(t, np.cos(omega * t), label=r"$\omega_\mathrm{particle} = " + r"{0:1.2f}".format(omega) + r"$")
    plt.plot(t, zeros, "--k")
    plt.gca().set_xlabel(r"$t$ [\si{\second}]")
    plt.gca().set_ylabel(r"$E$ [\si{\volt\metre}]")
    leg = plt.gca().legend()
    leg.draggable()
    plt.show()

def power_plot_cyclotron():
    t = np.linspace(0, 2 * np.pi, 200)
    n_max = 3
    zeros = np.zeros(t.shape)
    omega = 140.e9
    t /= omega / n_max
    k = 2.0 * np.pi * omega / cnst.c
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for n in [1, 2, 3]:
        rho_l = 0.1 * cnst.c * n / omega
        ax1.plot(t * 1.e12, np.cos(t * omega / float(n)) * np.cos(-omega * t), label=r"$\rho_\mathrm{l} = 0$, $n = " + r"{0:d}".format(n) + r"$")
        ax1.plot(t * 1.e12, np.cos(t * omega / float(n)) * np.cos(k * rho_l * np.sin(t * omega / float(n)) - omega * t), "--", label=r"$\rho_\mathrm{l} = \SI{" + r"{0:1.2f}".format(rho_l * 1.e3) + r"}{\milli\metre}$ , $n = " + r"{0:d}".format(n) + r"$")
    ax1.plot(t * 1.e12, zeros, "--k")
    ax1.set_xlabel(r"$t$ [\si{\pico\second}]")
    ax1.set_ylabel(r"$E$ [\si{\volt\metre}]")
    leg = ax1.legend()
    leg.draggable()
    Y = np.linspace(0.01, 1.0, 1000)
    E0 = 5.e5
    v0 = 0.1 * cnst.c
    Py = -E0 * cnst.e / cnst.m_e
    harmonic_sum = 0.0
    zero_rho = ((-Y + Y * np.cos(2 * omega * np.pi) * np.cos((2 * omega * np.pi) / Y) +
                              np.sin(2 * omega * np.pi) * np.sin((2 * omega * np.pi) / Y))) / (omega * (-1 + Y ** 2))
    for m in np.linspace(-10, 10, 11, dtype=np.int):
        harmonic_sum += (jn(m, v0 / cnst.c) * \
                         (-2.0 * Y + (-1.0 + Y + m * Y) * np.cos((2 * omega * np.pi * (-1 + (-1 + m) * Y)) / Y) + \
                         (1 + Y - m * Y) * np.cos((2 * omega * np.pi * (-1 + Y + m * Y)) / Y))) / \
                         (2.*omega * (-1 + (-1 + m) * Y) * (-1 + Y + m * Y))
    ax2.plot(Y, Py * harmonic_sum, "-", label=r"$P_\mathrm{y,el}$ finite $\rho_\mathrm{l}$")
    ax2.plot(Y, Py * zero_rho, "--", label=r"$P_\mathrm{el}$ zero $\rho_\mathrm{l}$")
    leg = ax2.legend()
    leg.draggable()
    ax2.set_xlabel(r"$\frac{\omega_\mathrm{c}}{\omega}$")
    ax2.set_ylabel(r"$P_\mathrm{el}$ [\si{\watt}]")
    plt.show()

def rx(t, v_perp, omega_c):
    return v_perp / omega_c * np.sin(omega_c * t)

def ry(t, v_perp, omega_c):
    return v_perp / omega_c * np.cos(omega_c * t)

def rxdt(t, v_perp, omega_c):
    return v_perp * np.cos(omega_c * t)

def rydt(t, v_perp, omega_c):
    return -v_perp * np.sin(omega_c * t)

def E_field(x, t, E_0, k, omega, phi0):
    return E_0 * np.cos(k * x - omega * t + phi0)

def cyclotron_damping():
    t = np.linspace(0, 8 * np.pi, 800)
    n_sum = 10
    zeros = np.zeros(t.shape)
    omega = 140.e9
    t /= omega
    k = 2.0 * np.pi * omega / cnst.c
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)
    # , bottom='off', top='off', left="off", right="off"
#    plt.tick_params('both', which='both', \
#                    labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax1_sub = plt.twinx(ax1)
    fig2 = plt.figure(tight_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1)
    omega_c = omega / 2.0
    v_perp = 2.5e-2 * cnst.c
    if(v_perp > 1.e-4 * cnst.c):
        velocity_prefix = "\mega"  # ""  # "\mega" #
        energy_prefix = ""  # "\micro"
        velocity_scaling = 1.e-6  # 1.0  #
        energy_scaling = 1.0  # 1.0e6 #
    else:
        velocity_prefix = ""  # ""  # "\mega" #
        energy_prefix = "\micro"
        velocity_scaling = 1.0
        energy_scaling = 1.0e6
    phi0 = -np.pi / 2.0
    E_0 = 1.e6
    ax1.plot(t * 1.e12, E_field(rx(t, v_perp, omega_c), t, E_0, k, omega, phi0) / 1.e6, "-r")
    ax1_sub.plot(t * 1.e12, rydt(t, v_perp, omega_c) * velocity_scaling, "--b")
    ax1.plot(t * 1.e12, zeros, ":k")
    ax1.set_xlabel(r"$t\,[\si{\pico\second}]$")
    ax1.set_ylabel(r"$E\,[\si{\mega\volt\per\meter}]$")
    ax1_sub.set_ylabel(r"$v_\mathrm{y}\,[\si{" + velocity_prefix + r"\meter\per\second}]$")
    for phi0, linestyle in zip([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2 ], ["-", "--", "-.", "--", "-"]):
        dW_dt = cnst.e * E_field(rx(t, v_perp, omega_c), t, E_0, k, omega, phi0) * rydt(t, v_perp, omega_c)
        dW_dt_spl = InterpolatedUnivariateSpline(t, dW_dt)
        W_spl = dW_dt_spl.antiderivative()
        ax2.plot(t * 1.e12, W_spl(t) / cnst.e * energy_scaling, linestyle, label=r"$\phi_0 = \ang{" + r"{0:1.2f}".format(np.rad2deg(phi0)) + r"}$")
        ax2.plot(t * 1.e12, zeros, ":k")
#    leg = ax2.legend()
#    leg.draggable()
    ax2.set_xlabel(r"$t\,[\si{\pico\second}]$")
    ax2.set_ylabel(r"$\Delta W\,[\si{" + energy_prefix + r"\electronvolt}]$")
    plt.show()

def phase_plot_cyclotron():
    t = np.linspace(0, 2 * np.pi, 800)
    n_max = 3
    zeros = np.zeros(t.shape)
    omega = 140.e9
    t *= n_max * 2
    t /= omega
    k = 2.0 * np.pi * omega / cnst.c
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    for n in [1, 1.5, 2, 2.5, 3, 3.5]:
        rho_l = 0.1 * cnst.c / omega * float(n)
        ax1.plot(t * omega / (2.0 * np.pi), np.arccos(np.cos((t * omega / float(n) - k * rho_l * np.sin(t * omega / float(n)) - omega * t))) / (2.0 * np.pi), label=r"$\phi_\mathrm{l} = 0$, $n = " + r"{0:1.1f}".format(n) + r"$")
#    ax1.plot(t * 1.e12, zeros, "--k")
    ax1.set_xlabel(r"$\phi_\mathrm{c}$")
    ax1.set_ylabel(r"$\phi_\mathrm{w}$")
    leg = ax1.legend()
    leg.draggable()
    plt.show()

def phase_plot_cyclotron_helical():
    t = np.linspace(0, 2 * np.pi, 200)
    zeros = np.zeros(t.shape)
    omega_c = np.linspace(0.1, 1.0, 100)  # np.array([0.2, 0.25, 0.33, 0.5, 1.0])
    omega = 2.0 * np.pi * 140.e9
    omega_c *= omega
    t /= omega
    k = 2.0 * np.pi * cnst.c / omega
    for n in range(-4, 4):
        rho_l = 0.1 * cnst.c / omega_c
        plt.semilogy(omega_c, jn(n, rho_l * k), label=r"$n = " + r"{0:d}".format(n) + r"$")
    plt.plot(t, zeros, "--k")
    plt.gca().set_xlabel(r"$\omega_\mathrm{c}$ [\si{\per\second}]")
    plt.gca().set_ylabel(r"$E$ [\si{\volt\metre}]")
    leg = plt.gca().legend()
    leg.draggable()
    plt.show()

def v_dot_and_power_gain_cycl():
    t = np.linspace(0, 2 * np.pi, 200)
    omega_c = np.linspace(0.1, 1.0, 100)
    omega = 2.0 * np.pi * 140.e9
    omega_c *= omega
    t /= omega
    k = 2.0 * np.pi * cnst.c / omega
    v0_perp = 0.1 * cnst.c
    rho_l = v0_perp / omega_c
    m = 0
    E0y = 5.e05
    plt.plot(omega_c / omega, -cnst.e / cnst.m_e * v0_perp * jn(m, k * rho_l) * E0y)
    plt.show()

if __name__ == "__main__":
    cyclotron_damping()
