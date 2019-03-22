'''
Created on Jul 18, 2018

@author: sdenk
'''
import numpy as np
from plotting_configuration import *
from scipy.io import readsav
from scipy.interpolate import UnivariateSpline

def Damping():
    u = np.linspace(0.0, 0.5, 100)
    f_0 = np.exp(-511.0 / 10.0 * u ** 2)
    f_1 = np.copy(f_0)
    f_2 = np.copy(f_0)
    u_res = 0.2
    delta_res = 0.05
    i_res_start = np.argmin(np.abs(u - u_res + delta_res))
    i_res = np.argmin(np.abs(u - u_res))
    i_res_stop = np.argmin(np.abs(u - u_res - delta_res))
    f_1[i_res_stop:len(f_1)] = f_1[i_res_start:len(f_1) - (i_res_stop - i_res_start)]
    f_1[i_res_start:i_res_stop] = f_1[i_res_start]
    i_res_half_stop = np.argmin(np.abs(u - u_res - 0.5 * delta_res))
    f_2[:] += f_0[i_res_start] * np.exp(-(u - u_res - 0.5 * delta_res) ** 2 / (0.8 * delta_res) ** 2)
    plt.plot(u, f_0, ":")
    plt.plot(u, f_1, "--")
    plt.plot(u, f_2, "-")
    plt.vlines(u_res, 0, 1.2 * np.max(f_0), linestyle="-.")
    plt.gca().set_xlabel("$v$")
    plt.gca().set_ylabel("$f$ [a.u.]")
    plt.tight_layout()
    plt.show()

def PowerDepo(filename, shotstr, p_dict):
    sav = readsav(filename)
    rho_pol = np.insert(sav["plflx_xb"], 0, 0.0)
    xb = np.insert(sav["xb"], 0, 0.0)
    xb_spl = UnivariateSpline(np.sqrt(rho_pol / np.max(rho_pol)), xb, s=0)
#    plt.plot(np.sqrt(rho_pol / np.max(rho_pol)), xb)
#    plt.show()
    profs = {}
    rhop_new = np.linspace(0.0, 0.99, 200)
    fig = plt.figure(figsize=(8.5, 4.5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    for quant in ["ecrh", "icrh", "nbi", "oh", "te"]:
        quant_org = sav[quant]
        quant_org[quant_org < 1.e-12] = 1.e-12
        quant_spl = UnivariateSpline(sav["rhotor"], np.log(quant_org), s=np.abs(np.log(np.max(sav[quant]))) / 1.e4)
        quant_arr_new = np.exp(quant_spl(xb_spl(rhop_new)))
        if(quant != "te"):
            quant_arr_new[:] /= p_dict[quant]
#            quant_spl_2 = UnivariateSpline(rhop_new, quant_arr_new, s=0)
#            quant_arr_new[:] /= quant_spl_2.integral(0.0, np.max(rhop_new))
        profs[quant] = np.copy(quant_arr_new)
    for quant, label, marker, multi in zip(["ecrh", "icrh", "nbi", "oh"], ["ECRH", r"IRCH  $\times$ 2", r"NBI $\times$ 3", "Ohmic"], \
                                  ["-", "--", "-.", ":"], [1.0, 2.0, 3.0, 1.0]):
        ax.plot(rhop_new, profs[quant] * multi, linestyle=marker, label=label)
    ax2.plot(rhop_new, profs["te"] / 1.e3, ":", label="$T_\mathrm{e}$")
    lns = ax.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    leg = ax.legend(lns, labs)
    leg.draggable()
    ax.set_ylabel(r"$\mathrm{d} P^*/\mathrm{d} V^*$")
    ax2.set_ylabel(r"$T_\mathrm{e}\,[\si{\kilo\electronvolt}]$")
    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax.text(0.7, 0.8, shotstr,
                verticalalignment='center', horizontalalignment='center',
                transform=ax.transAxes,
                color='black', fontsize=plt.rcParams['axes.titlesize'])
    ax.get_yaxis().set_major_locator(MaxNLocator(nbins=4))
    ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=4))
    ax2.get_yaxis().set_major_locator(MaxNLocator(nbins=4))
    ax2.get_yaxis().set_minor_locator(MaxNLocator(nbins=4))
    plt.tight_layout(pad=1.0)  # , rect=[0.05, 0.05, 0.95, 0.9]
    plt.show()



if(__name__ == "__main__"):
    p_dict = {}
    p_dict["ecrh"] = 720.e-3
    p_dict["icrh"] = 5.e0
    p_dict["nbi"] = 2.4e0
    p_dict["oh"] = 155.e-3
    PowerDepo("/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/heatingcomp_data.sav", r"\# 29783, $t = \SI{4.5}{\second}$", p_dict)

