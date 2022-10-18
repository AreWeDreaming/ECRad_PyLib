'''
Created on Jun 19, 2019
@author: Severin Denk
'''
# Tools to generate the ASCII distribution files required by ECRad and load the distribution class from .mat files or hdf5 files in case of GENE distributions
# Also includes a function to generate a thermal distribution on a grid and a distribution  resulting form the force balance between collisions and the radiation reaction force

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.integrate import simps
import os
from scipy.io import loadmat
from Distribution_Classes import Distribution, Beam, Gene
from Distribution_Helper_Functions import fill_zeros_with_thermal,check_distribution, \
                                          get_dist_moments_non_rel, get_dist_moments
from scipy import constants as cnst
import h5py
from Distribution_Functions import BiMaxwell2DV,Maxwell2D_beta,BiMaxwellJuettner2DV,Juettner2D,SynchrotonDistribution


def export_fortran_friendly(args):
    # Writes the ASCII distribution files needed by ECRad for a given distribution object
    # Note that the routine below should be used for GENE distributions
    # print(uxx, ull)
    dist_obj = args[0]
    wpath = args[1]
    print(dist_obj)
    print("Fe", np.shape(dist_obj.f))
    print("rhop", np.shape(dist_obj.rhop))
    print("pn", np.shape(dist_obj.u))
    print("mu", np.shape(dist_obj.pitch))
    rhopfile = open(wpath + "/frhop.dat", "w")
    pitchfile = open(wpath + "/pitch.dat", "w")
    ufile = open(wpath + "/u.dat", "w")
    pitchfile.write("{0: 5d}\n".format(len(dist_obj.pitch)))
    for i in range(len(dist_obj.pitch)):
        try:
            pitchfile.write("{0: 1.12e}\n".format(dist_obj.pitch[i]))
        except ValueError as e:
            print(e)
            print(dist_obj.pitch[i])
    pitchfile.flush()
    pitchfile.close()
    ufile.write("{0: 5d}\n".format(len(dist_obj.u)))
    for i in range(len(dist_obj.u)):
        ufile.write("{0: 1.12e}\n".format(dist_obj.u[i]))
    ufile.flush()
    ufile.close()
    rhopfile.write("{0: 5d}\n".format(len(dist_obj.rhop)))
    for i in range(len(dist_obj.rhop)):
        rhopfile.write("{0: 1.12e}\n".format(dist_obj.rhop[i]))
        thefile = open(wpath + "/fu{0:0>3}.dat".format(i), "w")
        for j in range(len(dist_obj.u)):
            for k in range(len(dist_obj.pitch)):
                thefile.write("{0: 1.8E} ".format(np.log(dist_obj.f[i, j, k])))
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()
    print("Distribution ready")

def export_gene_fortran_friendly(wpath, rhop, beta_par, mu_norm, ne, f, f0, B0):
    # Same as a bove but for GENE distribution
    # There are several routines for creating artificial GENE data based on (relativistic) Bi-Maxwellians (see below)
    f = f / ne
    f0 = f0 / ne
    f[f < 1.e-20] = 1.e-20
    f0[f0 < 1.e-20] = 1.e-20
    rhopfile = open(wpath + "/grhop.dat", "w")
    mufile = open(wpath + "/mu.dat", "w")  # mu, actually mu normalized
    vparfile = open(wpath + "/vpar.dat", "w") # vpas, actually beta_par
    B0_file = open(wpath + "/B0.dat", "w")
    B0_file.write("{0: 1.12e}\n".format(B0))
    B0_file.close()
    mufile.write("{0: 5d}\n".format(len(mu_norm)))
    for i in range(len(mu_norm)):
        try:
            mufile.write("{0: 1.12e}\n".format(mu_norm[i]))
        except ValueError as e:
            print(e)
            print(mu_norm[i])
    mufile.flush()
    mufile.close()
    vparfile.write("{0: 5d}\n".format(len(beta_par)))
    for i in range(len(beta_par)):
        vparfile.write("{0: 1.12e}\n".format(beta_par[i]))
    vparfile.flush()
    vparfile.close()
    rhopfile.write("{0: 5d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        rhopfile.write("{0: 1.12e}\n".format(rhop[i]))
        thefile = open(wpath + "/gvpar{0:0>3}.dat".format(i), "w")
        for j in range(len(beta_par)):
            for k in range(len(mu_norm)):
                thefile.write("{0: 1.8E} ".format(np.log(f[i, j, k])))  # / ne_prof[i]
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()
    thefile = open(wpath + "/f0.dat", "w")
    for j in range(len(beta_par)):
        for k in range(len(mu_norm)):
            thefile.write("{0: 1.8E} ".format(np.log(f0[j, k])))
        thefile.write("\n")
    thefile.flush()
    thefile.close()
    print("Gene distribution ready")

# def generate_single_thermal(rhop, Te):
#     u = np.linspace(0.0, 3.0, 200)
#     pitch = np.linspace(0, np.pi,400)
#     f = np.zeros((len(u), len(pitch)))
#     u_assist = np.zeros(len(pitch))
#     for i,u_single in enumerate(u):
#         u_assist[:] = u_single
#         f[i] = Juettner2D_cycl(u_assist, Te)
#     return distribution(None, np.array([rhop]), u, pitch, np.array([f]), \
#                         np.array([rhop]), np.array([rhop]), np.array([Te]), np.array([1.0]))
#     

def load_f_from_mat(filename=None, mdict = None, use_dist_prefix=False):
    # Load distribution object from .mat file created by AECM GUI
    if(mdict is None):
        mdict = loadmat(filename, squeeze_me=True)
    dist_prefix = ""
    if(use_dist_prefix is None):
        for key in mdict:
            if(key.startswith("dist")):
                dist_prefix = "dist_"
                break
    elif(use_dist_prefix):
        dist_prefix = "dist_"
    dist_obj = Distribution()
    if(dist_prefix + "rhot_prof" not in mdict):
        dist_obj.set(None, mdict[dist_prefix + "rhop_prof"], mdict[dist_prefix + "u"], mdict[dist_prefix + "pitch"], mdict[dist_prefix + "f"], \
                            mdict[dist_prefix + "rhot_1D_profs"], mdict[dist_prefix + "rhop_1D_profs"], mdict[dist_prefix + "Te_init"], \
                            mdict[dist_prefix + "ne_init"])
    else:
        dist_obj.set(mdict[dist_prefix + "rhot_prof"], mdict[dist_prefix + "rhop_prof"], \
                            mdict[dist_prefix + "u"], mdict[dist_prefix + "pitch"], mdict[dist_prefix + "f"], 
                            mdict[dist_prefix + "rhot_1D_profs"], mdict[dist_prefix + "rhop_1D_profs"], \
                            mdict[dist_prefix + "Te_init"], mdict[dist_prefix + "ne_init"])
    return dist_obj

def read_waves_mat_to_beam(waves_mat, EQSlice, use_wave_prefix=False):
    # Load waves object from .mat file created by AECM GUI (GRAY or TORAYFOM)
    wave_prefix = ""
    if(use_wave_prefix is None):
        for key in waves_mat:
            if(key.startswith("wave")):
                wave_prefix = "wave_"
                break
    elif(use_wave_prefix):
        wave_prefix = "wave_"
    rho_prof = waves_mat[wave_prefix + "rhop_prof"]
    j = waves_mat[wave_prefix + "j_prof"]
    PW = waves_mat[wave_prefix + "PW_prof"]
    PW_tot = waves_mat[wave_prefix + "PW_tot"]
    j_tot = waves_mat[wave_prefix + "j_tot"]
    rays = []
    B_tot = np.sqrt(EQSlice.Br ** 2 + EQSlice.Bt ** 2 + EQSlice.Bz ** 2)
    B_tot_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, B_tot)
    PW_beam = []
    j_beam = []
    for key in ["s", "R", "phi", "z", "rhop", "PW", "Npar"]:
        waves_mat[wave_prefix + key] = np.atleast_3d(waves_mat[wave_prefix + key])
        if(waves_mat[wave_prefix + key].shape[-1] == 1):
            waves_mat[wave_prefix + key] = np.swapaxes(waves_mat[wave_prefix + key].T, 1, 2)
    for key in ["PW_beam", "j_beam"]:
        waves_mat[wave_prefix + key] = np.atleast_2d(waves_mat[wave_prefix + key])
        if(waves_mat[wave_prefix + key].shape[-1] == 1):
            waves_mat[wave_prefix + key] = waves_mat[wave_prefix + key].T
    for ibeam in range(len(waves_mat[wave_prefix + "R"])):
        print("Processing beam: " + str(ibeam + 1))
        PW_beam.append(waves_mat[wave_prefix + "PW_beam"][ibeam])
        j_beam.append(waves_mat[wave_prefix + "j_beam"][ibeam])
        rays.append([])
        for iray in range(len(waves_mat[wave_prefix + "R"][ibeam])):
            print("Processing ray: " + str(iray + 1))
            rays[-1].append({})
            rays[-1][-1]["s"] = waves_mat[wave_prefix + "s"][ibeam][iray]
            rays[-1][-1]["R"] = waves_mat[wave_prefix + "R"][ibeam][iray]
            rays[-1][-1]["phi"] = waves_mat[wave_prefix + "phi"][ibeam][iray]
            rays[-1][-1]["z"] = waves_mat[wave_prefix + "z"][ibeam][iray]
            rays[-1][-1]["rhop"] = waves_mat[wave_prefix + "rhop"][ibeam][iray]
            rays[-1][-1]["PW"] = waves_mat[wave_prefix + "PW"][ibeam][iray]
            rays[-1][-1]["Npar"] = waves_mat[wave_prefix + "Npar"][ibeam][iray]
            rays[-1][-1]["omega_c"] = cnst.e * B_tot_spl(rays[-1][-1]["R"], rays[-1][-1]["z"], grid=False) / cnst.m_e
    PW_beam = np.array(PW_beam)
    j_beam = np.array(j_beam)
    return Beam(waves_mat[wave_prefix + "rhot_prof"], rho_prof, PW, j, PW_tot, j_tot, PW_beam, j_beam, rays)

def read_dist_mat_to_beam(dist_mat, use_dist_prefix=True):
    # Load waves object from .mat file created by AECM GUI (RELAX)
    dist_prefix = ""
    if(use_dist_prefix is None):
        for key in dist_mat:
            if(key.startswith("dist")):
                dist_prefix = "dist_"
                break
    elif(use_dist_prefix):
        dist_prefix = "dist_"
    rho_prof = dist_mat[dist_prefix + "rhop_prof"]
    j = dist_mat[dist_prefix + "j_prof"]
    PW = dist_mat[dist_prefix + "PW_prof"]
    PW_tot = dist_mat[dist_prefix + "PW_tot"]
    j_tot = dist_mat[dist_prefix + "j_tot"]
    return Beam(dist_mat[dist_prefix + "rhot_prof"], rho_prof, PW, j, PW_tot, j_tot, None, None, None)

def load_f_from_ASCII(path, rhop_in=None, Gene=False):
# Directly loads the ascii distribution files that are used by ECRad
    if(not Gene):
        x = np.loadtxt(os.path.join(path, "u.dat"), skiprows=1)
        y = np.loadtxt(os.path.join(path, "pitch.dat"), skiprows=1)
        ne_data = np.loadtxt(os.path.join(path, "..", "ne_file.dat"), skiprows=1)
        Te_data = np.loadtxt(os.path.join(path, "..", "Te_file.dat"), skiprows=1)
        rhop_ne = ne_data.T[0]
        ne = ne_data.T[1]
        Te = Te_data.T[1]
        rhop = np.loadtxt(os.path.join(path, "frhop.dat"), skiprows=1)
        rhop_B_min, B_min = np.loadtxt(os.path.join(path, "..", "B_min.dat"), unpack=True)
        B_min_spline = InterpolatedUnivariateSpline(rhop_B_min, B_min)
        if(rhop_in is not None):
            irhop = np.argmin(np.abs(rhop - rhop_in))
            rhop = np.array([rhop[irhop]])
            Fe = np.array([np.loadtxt(os.path.join(path, "fu{0:03d}.dat".format(irhop)))])
            B_min = B_min_spline
        else:
            Fe = np.zeros((len(rhop), len(x), len(y)))
            for irhop in range(len(rhop)):
                Fe[irhop, :, :] = np.loadtxt(os.path.join(path, "fu{0:03d}.dat".format(irhop)))
            B_min = B_min_spline
        return Distribution(None, rhop, x, y, np.exp(Fe), None, rhop_ne, Te, ne, B_min=B_min)
    else:
        x = np.loadtxt(os.path.join(path, "vpar.dat"), skiprows=1)
        y = np.loadtxt(os.path.join(path, "mu.dat"), skiprows=1)
        rhop = np.loadtxt(os.path.join(path, "grhop.dat"), skiprows=1)
        B0 = np.float64(np.loadtxt(os.path.join(path, "B0.dat")))
        if(rhop_in is not None):
            irhop = np.argmin(np.abs(rhop - rhop_in))
            Fe = np.loadtxt(os.path.join(path, "gvpar{0:03n}.dat".format(irhop)))
            return rhop[irhop], x, y, Fe, B0
        else:
            Fe = np.zeros((len(rhop), len(x), len(y)))
            for irhop in range(len(rhop)):
                Fe[irhop, :, :] = np.loadtxt(os.path.join(path, "gvpar{0:03n}.dat".format(irhop)))
        return rhop, x, y, Fe, B0

def read_LUKE_data(path, rhop_max=1.5, no_preprocessing=True, Flip=False):
# Returns a distribution object from LUKE input
# See also read_LUKE_profiles for this
    try:
        LUKE_f = loadmat(os.path.join(path, "LUKE.mat"))
        ne_filename = os.path.join(path, "ne_file.dat")
        ne_data = np.loadtxt(ne_filename, skiprows=1)
        rhop_vec_ne = ne_data.T[0]
        ne = ne_data.T[1]
        Te_filename = os.path.join(path, "Te_file.dat")
        Te_data = np.loadtxt(Te_filename, skiprows=1)
        rhop_vec_Te = Te_data.T[0]
        Te = Te_data.T[1]
        Fe = LUKE_f["f"] / LUKE_f["betath_ref"][0] ** 3
        print("pn", np.shape(Fe))
        x = LUKE_f["pn"][0] * LUKE_f["betath_ref"][0]
        print("pn", np.shape(x))
        y = LUKE_f["mhu"][0]
        print("mu", np.shape(y))
        y = np.arcsin(y)
        u = x
        Fe = np.swapaxes(Fe, 2, 0)
        Fe = np.swapaxes(Fe, 1, 2)
        if(Flip):
            Fe = Fe[:, :, ::-1]
        rhop = LUKE_f['xrhoP'][0]
        rhop = rhop[rhop < rhop_max]
        Fe = Fe[rhop < rhop_max]
        print("rhop", np.shape(rhop))
        ne_spl = InterpolatedUnivariateSpline(rhop_vec_ne, ne)
        Fe[:] = (Fe[:].T * LUKE_f["ne_ref"][0] / ne_spl(rhop)).T
        if(not no_preprocessing):
            print("Preprocessing LUKE distribution")
            Fe = fill_zeros_with_thermal(Fe, LUKE_f['xrhoP'][0], rhop_vec_Te, Te, u)
            if(not check_distribution(rhop, x, y, Fe)):
                print("Distribution in bad shape - output only for diagnostics !")
                # raise ValueError
        dist_obj = Distribution(None, rhop, x, y, Fe, None, rhop_vec_ne, Te, ne)
        return dist_obj
    except IOError as e:
        print(e)
        print("Could not find LUKE.mat for distribution at ", os.path.join(path, "LUKE.mat"))
        return []

def read_LUKE_profiles(path):
    # Could be extended to also read the rays for advanced plots like 3D Power deposition profile
    try:
        LUKE_mat = loadmat(os.path.join(path, "LUKE.mat"), struct_as_record=False, squeeze_me=True)
        scalar = LUKE_mat["data_proc"].scalar
        radial = LUKE_mat["data_proc"].radial
        waves = LUKE_mat["data_proc"].wave
        quasi_linear_beam = Beam(radial.xrhoT, radial.xrhoP, radial.P_tot * 1.e6, radial.J_tot * 1.e6, \
                                 scalar.p_rf_2piRp * 1.e6, scalar.I_tot * 1.e6)
        linear_beam = Beam(radial.xrhoT, radial.xrhoP, (waves.wxP_rf_lin[0] + waves.wxP_rf_lin[1]) * 1.e6, np.zeros(len(radial.xrhoP)), \
                                 scalar.p_rf_2piRp_lin * 1.e6, 0.0, \
                                 PW_beam=[waves.wxP_rf[0], waves.wxP_rf[1]], j_beam=[np.zeros(len(radial.xrhoP)), np.zeros(len(radial.xrhoP))])
        return quasi_linear_beam, linear_beam
    except IOError as e:
        print(e)
        print("Could not find LUKE.mat for beams at ", os.path.join(path, "LUKE_DATA.mat"))
        return [], []

def make_dist_from_Gene_input(path, shot, time, EQObj, debug=False):
    h5_fileID = h5py.File(path, 'r')
    it = 0
    beta_max = 0.5
    R = np.array(h5_fileID["axes"]["Rpos_m"]).flatten()
    z = np.array(h5_fileID["axes"]["Zpos_m"]).flatten()
    dR_down = R[1] - R[0]
    dR_up = R[-1] - R[-2]
    dz_down = z[1] - z[0]
    dz_up = z[-1] - z[-2]
    R = np.concatenate([[R[0] - dR_down], R, [R[-1] + dR_up]])
    z = np.concatenate([[z[0] - dz_down], z, [z[-1] + dz_up]])
    beta_par = np.array(h5_fileID["axes"]['vpar_m_s']).flatten() / cnst.c
    mu_norm = np.array(h5_fileID["axes"]['mu_Am2']).flatten() / (cnst.m_e * cnst.c ** 2)
    g = np.array(h5_fileID['delta_f'][u'{0:04d}'.format(it)]) * cnst.c ** 3
    f0 = np.array(h5_fileID['misc']['F0']) * cnst.c ** 3
    print(f0.shape, beta_par.shape, mu_norm.shape)
    Te = float(h5_fileID['general information']["Tref,eV"].value.replace(",", "."))
    ne = float(h5_fileID['general information']["nref,m^-3"].value.replace(",", "."))
    EQSlice = EQObj.GetSlice(time)
    rhop_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, EQSlice.rhop)
    rhop = rhop_spl(R, z, grid=False)
    Btot_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, np.sqrt(EQSlice.Br ** 2 + EQSlice.Bt ** 2 + EQSlice.Bz ** 2))
    B_local = np.float64(Btot_spl(R[int(len(R) / 2.0 + 0.5)], z[int(len(z) / 2.0 + 0.5)], grid=False))
    beta_perp = np.sqrt(mu_norm * 2.0 * B_local)
    f0 = f0[:, beta_perp < beta_max]
    f0 = f0[np.abs(beta_par) < beta_max, :]
    g = g[:, :, beta_perp < beta_max]
    g = g[:, np.abs(beta_par) < beta_max, :]
    mu_norm = mu_norm[beta_perp < beta_max]
    beta_par = beta_par[np.abs(beta_par) < beta_max]
    print(f0.shape, beta_par.shape, mu_norm.shape)
    f = np.concatenate([[f0], g + f0, [f0]])  # g +
    g_test = np.zeros(f.shape)
    g = np.concatenate([[np.zeros(np.shape(f0))], g, [np.zeros(np.shape(f0))]])
    plot = debug
    f_int = np.zeros(len(beta_par))
    f_test_int = np.zeros(len(beta_par))
    if(debug):
        from Plotting_Configuration import plt, MaxNLocator
        for i in range(1, len(rhop)):
    #        plt.contourf(vpar, mu, np.log10(f[i].T))
    #        plt.colorbar()
            # B = Btot_spl(R[i], z[i], grid=False)
            g_test[i] = make_test_GENE_input(Te, ne, beta_par, beta_perp)
    #        plt.figure()
    #        plt.contourf(vpar, mu, np.log10(g_test[i].T))
    #        plt.colorbar()
    #        plt.show()
            f_int[:] = 0.e0
            f_test_int[:] = 0.e0
            for j in range(len(beta_par)):
                gamma = np.sqrt(1.e0 / (1.e0 - beta_par[j] ** 2 - beta_perp ** 2))
                f_int[j] = simps(gamma ** 5 * beta_perp * 2 * np.pi * f[i, j, :], beta_perp)
                f_test_int[j] = simps(gamma ** 5 * beta_perp * 2 * np.pi * g_test[i, j, :], beta_perp)
                if(plot and np.all(np.abs(g_test[i, j, :]) > 0.0) and j % 5 == 0 and i == int(len(rhop) / 2)):
                    plt.plot(beta_perp, f[i, j, :] / g_test[i, j, :] , label=r"$\beta_\parallel = " + "{0:1.2f}".format(beta_par[j]) + "$")
            print("Rho {0:1.3f} ne_ref {1:1.4e} ne GENE {2:1.4e} ne test {3:1.4e} ne_Gene/ne_test {4:1.4e}".format(rhop[i], ne, simps(f_int, beta_par), simps(f_test_int, beta_par), simps(f_int, beta_par) / simps(f_test_int, beta_par)))
                    #  * 2.e0 * B / (np.sqrt(beta_perp) * cnst.c ** 3 * cnst.m_e)
        plt.gca().set_xlabel(r"$\beta_\perp$")
        plt.gca().set_ylabel(r"$f_\mathrm{GENE}/f_\mathrm{thermal}$")
        plt.title(r"$\rho_\mathrm{pol}=$" + "{0:1.3f}".format(rhop[int(len(rhop) / 2)]))
        plt.legend()
        plt.show()
    rhopindex = np.argsort(rhop)
    gene_data = {}
    gene_data["rhop"] = rhop[rhopindex]
    gene_data["R"] = R
    gene_data["z"] = z
    gene_data["beta_par"] = beta_par
    gene_data["mu_norm"] = mu_norm
    gene_data["f"] = f[rhopindex]
    gene_data["f9"] = f0
    gene_data["g"] = g[rhopindex]
    gene_data["Te"] = Te
    gene_data["ne"] = ne
    gene_data["B_local"] = B_local
    return gene_data

def browse_gene_dists(path, shot, time, it):
    from Plotting_Configuration import plt, MaxNLocator
    # Broken atm -> GENE requires EQSlice now
    gene_obj = Gene(path, shot, time=time, it=it)
    f = np.copy(gene_obj.f)
    f /= gene_obj.ne
    f[f < 1.e-20] = 1.e-20
    f = np.log10(f)
    beta_perp_dense = np.linspace(np.min(gene_obj.beta_perp), np.max(gene_obj.beta_perp), len(gene_obj.beta_par) / 2)
    beta_perp_dense = np.concatenate([beta_perp_dense[::-1], beta_perp_dense])
    f_map = []
    for ixx in range(len(beta_perp_dense)):
        ixx_nearest = np.argmin(np.abs(gene_obj.beta_perp - beta_perp_dense[ixx]))
        beta_perp_dense[ixx] = gene_obj.beta_perp[ixx_nearest]
        f_map.append([ixx, ixx_nearest])
    f_map = np.array(f_map)
    levels = np.linspace(-13, 5, 20)
    for f_slice in f:
        fig1 = plt.figure()
        fig2 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax2 = fig2.add_subplot(111)
        cont1 = ax1.contourf(gene_obj.beta_par, gene_obj.beta_perp, f_slice.T, cmap=plt.cm.get_cmap("plasma"), \
                     levels=levels)
        cont2 = ax1.contour(gene_obj.beta_par, gene_obj.beta_perp, f_slice.T, levels=levels, colors='k',
                            hold='on', alpha=0.25, linewidths=1)
        for c in cont2.collections:
            c.set_linestyle('solid')
        cb = fig1.colorbar(cont1, ax=ax1, ticks=[-10, -5, 0, 5])  # ticks = levels[::6] #,
        cb.set_label(r"$\log_\mathrm{10}\left(f\right)$")  # (R=" + "{:1.2}".format(R_arr[i]) + r" \mathrm{m},u_\perp,u_\Vert))
        # cb.ax.get_yaxis().set_major_locator(MaxNLocator(nbins = 3, steps=np.array([1.0,5.0,10.0])))
        cb.ax.get_yaxis().set_minor_locator(MaxNLocator(nbins=3, steps=np.array([1.0, 5.0, 10.0])))
        cb.ax.minorticks_on()
        ax1.plot(gene_obj.beta_par, beta_perp_dense)
        ax2.plot(gene_obj.beta_par, f_slice[f_map.T[0], f_map.T[1]])
        ax1.set_xlabel(r"$\beta_\parallel$")
        ax1.set_ylabel(r"$\beta_\perp$")
        ax2.set_xlabel(r"$\beta_\parallel$")
        ax2.set_ylabel(r"$\log_{10}(f_\mathrm{GENE})$")
        plt.show()

def make_test_GENE_input(Te, ne, beta_par, beta_perp, turb=False):
    g = np.zeros((len(beta_par), len(beta_perp)))
    if(turb):
        fluct_amp = np.random.rand()
        print("fluct amp", fluct_amp)
        Te_par = Te * (1.0 + (np.random.rand() - 0.5) / 50.0)
        Te_perp = Te * (1.0 + (np.random.rand() - 0.5) / 10.0)
    for ill in range(len(beta_par)):
        if(turb):
            g[ill, :] = ne * BiMaxwell2DV(beta_par[ill], beta_perp, Te_par, Te_perp)
        else:
            g[ill, :] = ne * Maxwell2D_beta(beta_par[ill], beta_perp, Te)
            # g[ill, :] = ne * Juettner2D_beta(beta_par[ill], beta_perp, Te)
        # print(g[ill])
    return g

def export_art_gene_f_fortran_friendly(args):
    # print(uxx, ull)
    wpath = args[0]
    rhop = args[1]
    beta_par = args[2]
    mu_norm = args[3]
    f = args[4]
    f0 = args[5]
    ne = args[6]
    B0 = args[7]
    rhopfile = open(wpath + "/grhop.dat", "w")
    mufile = open(wpath + "/mu.dat", "w")
    vparfile = open(wpath + "/vpar.dat", "w")
    B0_file = open(wpath + "/B0.dat", "w")
    B0_file.write("{0: 1.12e}\n".format(B0))
    B0_file.close()
    mufile.write("{0: 5d}\n".format(len(mu_norm)))
    f = f / ne
    f0 = f0 / ne
    f[f < 1.e-20] = 1.e-20
    f0[f0 < 1.e-20] = 1.e-20
    for i in range(len(mu_norm)):
        try:
            mufile.write("{0: 1.12e}\n".format(mu_norm[i]))
        except ValueError as e:
            print(e)
            print(mu_norm[i])
    mufile.flush()
    mufile.close()
    vparfile.write("{0: 5d}\n".format(len(beta_par)))
    for i in range(len(beta_par)):
        vparfile.write("{0: 1.12e}\n".format(beta_par[i]))
    vparfile.flush()
    vparfile.close()
    rhopfile.write("{0: 5d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        rhopfile.write("{0: 1.12e}\n".format(rhop[i]))
        thefile = open(wpath + "/gvpar{0:0>3}.dat".format(i), "w")
        for j in range(len(beta_par)):
            for k in range(len(mu_norm)):
                thefile.write("{0: 1.8E} ".format(np.log(f[i, j, k])))
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()
    thefile = open(wpath + "/f0.dat", "w")
    for j in range(len(beta_par)):
        for k in range(len(mu_norm)):
            thefile.write("{0: 1.8E} ".format(np.log(f0[j, k])))
        thefile.write("\n")
    thefile.flush()
    thefile.close()
    print("Gene distribution ready")

def make_bimax_from_GENE(path, shot, time, wpath_parent, subdir_list, wrong=False, write=False):
    from Plotting_Configuration import plt, MaxNLocator
    # For comparison of GENE Trad vs Bimax Trad
    rhop, R, z, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(path, shot, time, debug=False)
    Te_perp, Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f, Te, ne, B0, slices=1)
    Te_prof = np.zeros(len(rhop))
    Te_prof[:] = Te
    f_perp_low_max = np.zeros(f.shape)
    f_par_low_max = np.zeros(f.shape)
    f_perp_par_low_max = np.zeros(f.shape)
    beta_perp = np.sqrt(mu_norm * 2.0 * B0)
    for i in range(len(Te_perp)):
        for j in range(len(beta_par)):
            f_perp_low_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te, Te_perp[i])
            f_par_low_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te)
            f_perp_par_low_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te_perp[i])
    if(wrong):
        Te_perp_low_max_Te_perp, Te_perp_low_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_low_max, Te, ne, B0, slices=1)
        Te_par_low_max_Te_perp, Te_par_low_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_par_low_max, Te, ne, B0, slices=1)
        Te_perp_Te_par_low_max_Te_perp, T_perp_Te_par_low_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_par_low_max, Te, ne, B0, slices=1)
    else:
        Te_perp_low_max_Te_perp, Te_perp_low_max_Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f_perp_low_max, Te, ne, B0, slices=1)
        Te_par_low_max_Te_perp, Te_par_low_max_Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f_par_low_max, Te, ne, B0, slices=1)
        Te_perp_Te_par_low_max_Te_perp, T_perp_Te_par_low_max_Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f_perp_par_low_max, Te, ne, B0, slices=1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(rhop, Te_prof / 1.e3, "-k", label=r"$T_{\mathrm{e},0}$[GENE]")
    ax1.plot(rhop, Te_perp / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\perp$[GENE]")
    ax1.plot(rhop, Te_perp_low_max_Te_perp / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$]")
    ax1.plot(rhop, Te_par_low_max_Te_perp / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\parallel$]")
    ax1.plot(rhop, Te_perp_Te_par_low_max_Te_perp / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$, $T_\parallel$]")
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"$\tilde{T}_{\mathrm{e},\perp}$ [keV]")
    ax1.legend()
    fig1.suptitle("Maxwell")
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(rhop, Te_prof / 1.e3, "-k", label=r"$T_{\mathrm{e},0}$[GENE]")
    ax2.plot(rhop, Te_par / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\parallel$}[GENE]")
    ax2.plot(rhop, Te_perp_low_max_Te_par / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$]")
    ax2.plot(rhop, Te_par_low_max_Te_par / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\parallel$]")
    ax2.plot(rhop, T_perp_Te_par_low_max_Te_par / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$, $T_\parallel$]")
    ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax2.set_ylabel(r"$\tilde{T}_{\mathrm{e},\parallel}$ [keV]")
    ax2.legend()
    fig2.suptitle("Maxwell")
    if(write):
        subdir_index = 0
        wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
        if(not os.path.isdir(wpath)):
            os.mkdir(wpath)
        args = [wpath, rhop, beta_par, mu_norm, f_perp_low_max, f0, ne, B0]
        export_art_gene_f_fortran_friendly(args)
        subdir_index += 1
        wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
        if(not os.path.isdir(wpath)):
            os.mkdir(wpath)
        args = [wpath, rhop, beta_par, mu_norm, f_par_low_max, f0, ne, B0]
        export_art_gene_f_fortran_friendly(args)
        subdir_index += 1
        wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
        if(not os.path.isdir(wpath)):
            os.mkdir(wpath)
        args = [wpath, rhop, beta_par, mu_norm, f_perp_par_low_max, f0, ne, B0]
        export_art_gene_f_fortran_friendly(args)
        subdir_index += 1
    N = 100
    beta_par = np.linspace(-0.7, 0.7, 2 * N)
    beta_perp = np.linspace(0, 0.7, N)
    mu_norm = beta_perp ** 2 / (2.0 * B0)
    f0 = np.zeros((len(beta_par), len(beta_perp)))
    f_perp_large_max = np.zeros((len(Te_perp), len(beta_par), len(beta_perp)))
    f_par_large_max = np.zeros((len(Te_perp), len(beta_par), len(beta_perp)))
    f_perp_par_large_max = np.zeros((len(Te_perp), len(beta_par), len(beta_perp)))
    for i in range(len(Te_perp)):
        for j in range(len(beta_par)):
            if(i == 0):
                f0[j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te, Te)
            f_perp_large_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te, Te_perp[i])
            f_par_large_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te)
            f_perp_par_large_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te_perp[i])
    if(wrong):
        Te_perp_large_max_Te_perp, Te_perp_large_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_large_max, Te, ne, B0, slices=1)
        Te_par_large_max_Te_perp, Te_par_large_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_par_large_max, Te, ne, B0, slices=1)
        Te_perp_Te_par_large_max_Te_perp, T_perp_Te_par_large_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_par_large_max, Te, ne, B0, slices=1)
    else:
        Te_perp_large_max_Te_perp, Te_perp_large_max_Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f_perp_large_max, Te, ne, B0, slices=1)
        Te_par_large_max_Te_perp, Te_par_large_max_Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f_par_large_max, Te, ne, B0, slices=1)
        Te_perp_Te_par_large_max_Te_perp, T_perp_Te_par_large_max_Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f_perp_par_large_max, Te, ne, B0, slices=1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(rhop, Te_prof / 1.e3, "-k", label=r"$T_{\mathrm{e},0}$[GENE]")
    ax3.plot(rhop, Te_perp / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\perp}$[GENE]")
    ax3.plot(rhop, Te_perp_large_max_Te_perp / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$]")
    ax3.plot(rhop, Te_par_large_max_Te_perp / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\parallel$]")
    ax3.plot(rhop, Te_perp_Te_par_large_max_Te_perp / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$, $T_\parallel$]")
    ax3.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax3.set_ylabel(r"$\tilde{T}_{\mathrm{e},\perp}$ [keV]")
    ax3.legend()
    fig3.suptitle("Maxwell")
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(rhop, Te_prof / 1.e3, "-k", label=r"$T_{\mathrm{e},0}$[GENE]")
    ax4.plot(rhop, Te_par / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\parallel$}[GENE]")
    ax4.plot(rhop, Te_perp_large_max_Te_par / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$]")
    ax4.plot(rhop, Te_par_large_max_Te_par / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\parallel$]")
    ax4.plot(rhop, T_perp_Te_par_large_max_Te_par / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$, $T_\parallel$]")
    ax4.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax4.set_ylabel(r"$\tilde{T}_{\mathrm{e},\parallel}$ [keV]")
    ax4.legend()
    fig4.suptitle("Maxwell")
    plt.show()
    if(write):
        wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
        if(not os.path.isdir(wpath)):
            os.mkdir(wpath)
        args = [wpath, rhop, beta_par, mu_norm, f_perp_large_max, f0, ne, B0]
        export_art_gene_f_fortran_friendly(args)
        subdir_index += 1
        wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
        if(not os.path.isdir(wpath)):
            os.mkdir(wpath)
        args = [wpath, rhop, beta_par, mu_norm, f_par_large_max, f0, ne, B0]
        export_art_gene_f_fortran_friendly(args)
        subdir_index += 1
        wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
        if(not os.path.isdir(wpath)):
            os.mkdir(wpath)
        args = [wpath, rhop, beta_par, mu_norm, f_perp_par_large_max, f0, ne, B0]
        export_art_gene_f_fortran_friendly(args)
        print("All 6 artifical GENE data sets ready")
    return

def make_bimaxjuett_from_GENE(path, shot, time, wpath_parent, subdir_list):
    from Plotting_Configuration import plt, MaxNLocator
    # For comparison of GENE Trad vs relativistic Bimax Trad
    rhop, R, z, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(path, shot, time, debug=False)
    Te_perp, Te_par = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f, Te, ne, B0, slices=1)
    f_perp_low_max = np.zeros(f.shape)
    f_par_low_max = np.zeros(f.shape)
    f_perp_par_low_max = np.zeros(f.shape)
    beta_perp = np.sqrt(mu_norm * 2.0 * B0)
    for i in range(len(Te_perp)):
        for j in range(len(beta_par)):
#            f_perp_low_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te, Te_perp[i])
#            f_par_low_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te)
#            f_perp_par_low_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te_perp[i])
            f_perp_low_max[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te, Te_perp[i])
            f_par_low_max[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te_par[i], Te)
            f_perp_par_low_max[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te_par[i], Te_perp[i])
    Te_perp_low_max_Te_perp, Te_perp_low_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_low_max, Te, ne, B0, slices=1)
    Te_par_low_max_Te_perp, Te_par_low_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_par_low_max, Te, ne, B0, slices=1)
    Te_perp_Te_par_low_max_Te_perp, T_perp_Te_par_low_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_par_low_max, Te, ne, B0, slices=1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(rhop, Te_perp / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\perp$[GENE]")
    ax1.plot(rhop, Te_perp_low_max_Te_perp / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$]")
    ax1.plot(rhop, Te_par_low_max_Te_perp / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\parallel$]")
    ax1.plot(rhop, Te_perp_Te_par_low_max_Te_perp / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$, $T_\parallel$]")
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylabel(r"$\tilde{T}_{\mathrm{e},\perp}$ [keV]")
    ax1.legend()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(rhop, Te_par / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\parallel$}[GENE]")
    ax2.plot(rhop, Te_perp_low_max_Te_par / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$]")
    ax2.plot(rhop, Te_par_low_max_Te_par / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\parallel$]")
    ax2.plot(rhop, T_perp_Te_par_low_max_Te_par / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$, $T_\parallel$]")
    ax2.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax2.set_ylabel(r"$\tilde{T}_{\mathrm{e},\parallel}$ [keV]")
    ax2.legend()
    subdir_index = 0
    wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
    if(not os.path.isdir(wpath)):
        os.mkdir(wpath)
    args = [wpath, rhop, beta_par, mu_norm, f_perp_low_max, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    subdir_index += 1
    wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
    if(not os.path.isdir(wpath)):
        os.mkdir(wpath)
    args = [wpath, rhop, beta_par, mu_norm, f_par_low_max, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    subdir_index += 1
    wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
    if(not os.path.isdir(wpath)):
        os.mkdir(wpath)
    args = [wpath, rhop, beta_par, mu_norm, f_perp_par_low_max, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    subdir_index += 1
    N = 100
    beta_par = np.linspace(-0.7, 0.7, 2 * N)
    beta_perp = np.linspace(0, 0.7, N)
    mu_norm = beta_perp ** 2 / (2.0 * B0)
    f0 = np.zeros((len(beta_par), len(beta_perp)))
    f_perp_large_max = np.zeros((len(Te_perp), len(beta_par), len(beta_perp)))
    f_par_large_max = np.zeros((len(Te_perp), len(beta_par), len(beta_perp)))
    f_perp_par_large_max = np.zeros((len(Te_perp), len(beta_par), len(beta_perp)))
    for i in range(len(Te_perp)):
        for j in range(len(beta_par)):
            if(i == 0):
                f0[j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te, Te)
#            f_perp_large_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te, Te_perp[i])
#            f_par_large_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te)
#            f_perp_par_large_max[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par[i], Te_perp[i])
            f_perp_large_max[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te, Te_perp[i])
            f_par_large_max[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te_par[i], Te)
            f_perp_par_large_max[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te_par[i], Te_perp[i])
    Te_perp_large_max_Te_perp, Te_perp_large_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_large_max, Te, ne, B0, slices=1)
    Te_par_large_max_Te_perp, Te_par_large_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_par_large_max, Te, ne, B0, slices=1)
    Te_perp_Te_par_large_max_Te_perp, T_perp_Te_par_large_max_Te_par = get_dist_moments(rhop, beta_par, mu_norm, f_perp_par_large_max, Te, ne, B0, slices=1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(rhop, Te_perp / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\perp}$[GENE]")
    ax3.plot(rhop, Te_perp_large_max_Te_perp / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$]")
    ax3.plot(rhop, Te_par_large_max_Te_perp / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\parallel$]")
    ax3.plot(rhop, Te_perp_Te_par_large_max_Te_perp / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\perp}$[$T_\perp$, $T_\parallel$]")
    ax3.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax3.set_ylabel(r"$\tilde{T}_{\mathrm{e},\perp}$ [keV]")
    ax3.legend()
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(rhop, Te_par / 1.e3, "-b", label=r"$\tilde{T}_{\mathrm{e},\parallel$}[GENE]")
    ax4.plot(rhop, Te_perp_large_max_Te_par / 1.e3, "--r", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$]")
    ax4.plot(rhop, Te_par_large_max_Te_par / 1.e3, "+k", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\parallel$]")
    ax4.plot(rhop, T_perp_Te_par_large_max_Te_par / 1.e3, ":g", label=r"$\tilde{T}_{\mathrm{e},\parallel}$[$T_\perp$, $T_\parallel$]")
    ax4.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax4.set_ylabel(r"$\tilde{T}_{\mathrm{e},\parallel}$ [keV]")
    ax4.legend()
    plt.show()
    wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
    if(not os.path.isdir(wpath)):
        os.mkdir(wpath)
    args = [wpath, rhop, beta_par, mu_norm, f_perp_large_max, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    subdir_index += 1
    wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
    if(not os.path.isdir(wpath)):
        os.mkdir(wpath)
    args = [wpath, rhop, beta_par, mu_norm, f_par_large_max, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    subdir_index += 1
    wpath = os.path.join(wpath_parent, subdir_list[subdir_index], "ECRad_data", "fGe")
    if(not os.path.isdir(wpath)):
        os.mkdir(wpath)
    args = [wpath, rhop, beta_par, mu_norm, f_perp_par_large_max, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    print("All 6 artifical GENE data sets ready")
    return






def make_test_f(rpath):
    Te_vec = np.loadtxt(os.path.join(rpath, "Te_file.dat"), skiprows=1, unpack=True)[1]
    rhop_prof, ne_vec = np.loadtxt(os.path.join(rpath, "ne_file.dat"), skiprows=1, unpack=True)
    rhop = np.linspace(0.001, 0.5, 60)
    m = 200
    n = 100
    Fe = np.zeros([len(rhop), m, n])
    u = np.linspace(0, 1.5, m)
    pitch = np.linspace(0, np.pi, n)
    Te_spl = InterpolatedUnivariateSpline(rhop_prof, Te_vec)
    for i in range(len(rhop)):
        for j in range(len(u)):
                Fe[i, j, :] = Juettner2D(u[j], 0, Te_spl(rhop[i]))
    # plt.plot(np.arange(0,1.5,0.01),Fe[2,:,0])
    # plt.show()
    return Distribution(None, rhop, u, pitch, Fe, None, rhop_prof, Te_vec, ne_vec)

def make_synchroton_f(rpath, B):
    Te_vec = np.loadtxt(os.path.join(rpath, "Te_file.dat"), skiprows=1, unpack=True)[1]
    rhop_prof, ne_vec = np.loadtxt(os.path.join(rpath, "ne_file.dat"), skiprows=1, unpack=True)
    rhop = np.linspace(0.001, 0.5, 60)
    m = 200
    n = 100
    u = np.linspace(0.0, 1.5, m)
    pitch = np.linspace(0.0, 2.0 * np.pi, n)
    zeta = np.cos(pitch)
    Fe = np.zeros([len(rhop), m, n])
    Te_spl = InterpolatedUnivariateSpline(rhop_prof, Te_vec)
    ne_spl = InterpolatedUnivariateSpline(rhop_prof, ne_vec)
    for i in range(len(rhop)):
        for j in range(len(u)):
                    Fe[i, j, :] = Juettner2D(u[j], 0, Te_spl(rhop[i]))
                    if(u[j] > 0.2):
                        f = SynchrotonDistribution(u[j], zeta, Te_spl(rhop[i]), ne_spl(rhop[i]), B, 1.0)[2]
                        Fe[i, j, :] *= (1.0 + f)
    return Distribution(None, rhop, u, pitch, Fe, None, rhop_prof, Te_vec, ne_vec)     

def plot_dist_moments(path, shot, time, eq_exp='AUGD', eq_diag='EQH', eq_ed=0):
    from Plotting_Configuration import plt, MaxNLocator
    rhop_Gene, R, z, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(path, shot, time, debug=False)
    Te_perp, Te_par = get_dist_moments_non_rel(rhop_Gene, beta_par, mu_norm, f, Te, ne, B0, slices=1)
    ax = plt.gca()
    ax.plot(rhop_Gene, 1.e2 * (1.0 - Te_perp / Te), "-", label=r"$1 - \tilde{T}_\mathrm{e,\perp} / T_\mathrm{e}$")
    ax.plot(rhop_Gene, 1.e2 * (1.0 - Te_par / Te), "--", label=r"$1 - \tilde{T}_\mathrm{e,\Vert} / T_\mathrm{e}$")
    Te_perp_h5 = h5py.File(os.path.join(path, "AUG33585_Tperp_t1612.7_1c.h5"), 'r')
    Te_par_h5 = h5py.File(os.path.join(path, "AUG33585_Tpar_t1612.7_1c.h5"), 'r')
    Te_perp_spl = RectBivariateSpline(np.array(Te_perp_h5["R"]), np.array(Te_perp_h5["Z"]), np.array(Te_perp_h5["/data"]["0001"]))
    Te_par_spl = RectBivariateSpline(np.array(Te_par_h5["R"]), np.array(Te_par_h5["Z"]), np.array(Te_par_h5["/data"]["0001"]))
    ax.plot(rhop_Gene, 1.e2 * (1.0 - Te_perp_spl(R, z, grid=False)) / Te, "-", label=r"$1 - \tilde{T}_\mathrm{e,\perp,GENE} / T_\mathrm{e}$")
    ax.plot(rhop_Gene, 1.e2 * (1.0 - Te_par_spl(R, z, grid=False)) / Te, "--", label=r"$1 - \tilde{T}_\mathrm{e,\Vert,GENE} / T_\mathrm{e}$")
    ax.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax.set_ylabel(r"$1 - \tilde{T}_\mathrm{e} / T_\mathrm{e} \left[\si{\percent}\right]$")
#    ax2 = ax.twinx()
#    ax2.plot(rhop, (1.0 - Te_perp / Te_spl(rhop)) / Te_perp_spl(R, z, grid=False) * Te_spl(rhop), "+", label=r"$\tilde{T}_\mathrm{e,\perp} / \tilde{T}_\mathrm{e,\perp,GENE}$")
#    ax2.plot(rhop, (1.0 - Te_par / Te_spl(rhop)) / Te_par_spl(R, z, grid=False) * Te_spl(rhop), "^", label=r"$\tilde{T}_\mathrm{e,\Vert} / \tilde{T}_\mathrm{e,\Vert,GENE}$")
#    ax2.plot(rhop, (Te_perp / Te) / Te_perp_spl(R, z, grid=False) * Te, "+", label=r"$\tilde{T}_\mathrm{e,\perp} / \tilde{T}_\mathrm{e,\perp,GENE}$")
#    ax2.plot(rhop, (Te_par / Te) / Te_par_spl(R, z, grid=False) * Te, "^", label=r"$\tilde{T}_\mathrm{e,\Vert} / \tilde{T}_\mathrm{e,\Vert,GENE}$")
#    ax2.set_ylabel(r"$\tilde{T}_\mathrm{e} / \tilde{T}_\mathrm{e,GENE}$")
#     ax.plot(rhop, Te_spl(rhop), "+")
#    ax.plot(rhop, Te_perp_spl(R, z, grid=False), "-g", label=r"$1 - \tilde{T}_\mathrm{e,\perp,GENE} / T_\mathrm{e}$")
#    ax.plot(rhop, Te_par_spl(R, z, grid=False), "--k", label=r"$1 - \tilde{T}_\mathrm{e,\Vert,GENE} / T_\mathrm{e}$")
#    plt.plot(rhop, 1.e2 * (1.0 - Te / Te_spl(rhop)), ":k", label=r"$1 - \tilde{T}_\mathrm{e} / T_\mathrm{e}$")
    lns = ax.get_lines()  # + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    plt.show()

def calculate_and_export_gene_bimax_fortran_friendly(args):
    rpath = args[0]
    shot = args[1]
    time = args[2]
    eq_exp = args[3]
    eq_diag = args[4]
    eq_ed = args[5]
    wpath = args[6]
    relativistic = args[7]
    rhop_Gene, R, z, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(rpath, shot, time, eq_exp=eq_exp, eq_diag=eq_diag, eq_ed=eq_ed)
    Te_perp, Te_par, ne = get_dist_moments_non_rel(rhop_Gene, beta_par, mu_norm, f, Te, ne, B0, slices=1, ne_out=True)
    return export_gene_bimax_fortran_friendly(wpath, rhop_Gene, beta_par, mu_norm, Te, ne, Te_perp, Te_par, B0, relativistic)

def export_gene_bimax_fortran_friendly(wpath, rhop_Gene, beta_par, mu_norm, Te, ne, Te_perp, Te_par, B0, relativistic=True):
    Te_perp_inter = Te_perp
    Te_par_inter = Te_par
    rhop_inter = rhop_Gene
#    N = 100
#    beta_par = np.linspace(-0.7, 0.7, 2 * N)
#    beta_perp = np.linspace(0, 0.7, N)
#    mu_norm = beta_perp ** 2 / (2.0 * B0)
    beta_perp = np.sqrt(mu_norm * 2.e0 * B0)
    f0 = np.zeros((len(beta_par), len(beta_perp)))
    f = np.zeros((len(Te_perp_inter), len(beta_par), len(beta_perp)))
    for i in range(len(Te_perp_inter)):
        for j in range(len(beta_par)):
            if(i == 0):
                if(relativistic):
                    f0[j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te, Te)
                else:
                    f0[j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te, Te)
            if(relativistic):
                f[i, j, :] = ne * BiMaxwellJuettner2DV(beta_par[j], beta_perp, Te_par_inter[i], Te_perp_inter[i])
            else:
                f[i, j, :] = ne * BiMaxwell2DV(beta_par[j], beta_perp, Te_par_inter[i], Te_perp_inter[i])
    args = [wpath, rhop_inter, beta_par, mu_norm, f, f0, ne, B0]
    export_art_gene_f_fortran_friendly(args)
    return rhop_inter, Te_par_inter, Te_perp_inter

def load_and_export_fortran_friendly(args):
    rpath = args[0]
    shot = args[1]
    time = args[2]
    eq_exp = args[3]
    eq_diag = args[4]
    eq_ed = args[5]
    wpath = args[6]
    rhop, R, z, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input(rpath, shot, time, eq_exp=eq_exp, eq_diag=eq_diag, eq_ed=eq_ed)
    #    Te_perp, Te_par, ne_prof = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f, Te, ne, B0, slices=1, ne_out=True)
    export_gene_fortran_friendly(wpath, rhop, beta_par, mu_norm, f, f0, B0)  
