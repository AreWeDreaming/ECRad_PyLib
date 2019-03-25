# Import statements
import numpy as np
import fmtout as fmt
import SpitzerHaerm as sh
import os
import scipy.odr as odr
import scipy.optimize as scopt
try:
    import h5py
    h5py_ready = True
except ImportError:
    print("Warning! h5py not loaded - h5 files will not be readable")
    h5py_ready = False
from scipy import __version__ as scivers
from subprocess import call, Popen
from scipy.special import jn
from  threading import Lock
# oot = "C:\\nssf\\"
# libjuettner = ct.cdll.LoadLibrary(root + "F90/IDA_GUI_Ext/libeJp.so")
# Build the parallel and perpendicular momenta on the equatorial plane
from plotting_configuration import *
# from mayavi import mlab
from scipy.interpolate import RectBivariateSpline, splev, splrep, InterpolatedUnivariateSpline
import scipy.constants as cnst
from scipy.integrate import simps
from time import sleep
try:
    from scipy.integrate import nquad
except ImportError:
    print('Old scipy version detected! Cannot load nquad!')
from scipy.special import erf
from scipy.io import loadmat
from scipy.special import kve
try:
    from scipy.optimize import least_squares
except ImportError:
    print("Could not load least_sqarues method from scipy")
    print("This method requires scipy 17.X or higher")
    print("Some fitting routines might not be available")
from em_Albajar import distribution_interpolator, s_vec, em_abs_Alb
from wxEvents import *

def momenta_on_equatorial_plane(x, y):

    """ 
    Compute the normalized parallel ull and perpendicular uxx momentum
    on the points of the numerical grid of RELAX.
    
    Remark: the grid in RELAX is Cartesian in x = |u| y = arccos(ull/|u|),
    therefore points in the (ull, uxx) plane are located on a polar grid.
    """

    # Extract number of grid points
    nptx = np.size(x)
    npty = np.size(y)

    # Build the grid in pll (parallel momentum) and in
    # pxx (perpendicular momentum)
    pll = np.empty([nptx, npty])
    pxx = np.empty([nptx, npty])
    for ix in range(0, nptx):
        for iy in range(0, npty):
            # normalized parallel momentum
            pll[ix, iy] = x[ix] * np.cos(y[iy])
            # normalized perpendicular momentum
            pxx[ix, iy] = x[ix] * np.sin(y[iy])

    # Exit
    return pll, pxx

# Read the distribution function from the RELAX data set
def read_Fe(working_dir):

    """ 
    Usage:
      
      ipsi, psi, ull, uxx = read_Fe(wroking_dir)

    Arguments:

      1. wroking_dir = string with path to directory of the 
         file relax_output.dat.

    Return:
    
      1. ipsi = int, index of the surface of maximum deposition.
    
      2. psi = ndarray, dim 1, normalized poloidal flux of magnetic
         surfaces where the distribution function is known.
         
      3. ull = ndarray, dim 1, normalized relativistic parallel speed.

      4. uxx = ndarray, dim 1, normalized relativistic perpendicular speed.

      5. Fe = ndarray, dim 2, distribution funtion Fe(psi, ull, uxx),
         where Fe is the electron distribution funtion computed by RELAX. 
         
    Remark: Fe is not remapped, i.e., it must be considered the distribution
            of ull, uxx on the outer equatorial plane.
    Remark: The grid is polar in the (ull, uxx) plane.
    """

    # Load RELAX results
    relax_results = fmt.fmtout(working_dir + 'relax_output.dat')
    print("Successfully imported: " + working_dir + 'relax_output.dat')
    # For a table of the content of the output from RELAX
    # uncomment the following line
    # print(relax_results.table())

    # Index of the surface of maximum deposition
    ipsi = np.argmax(relax_results.dPec_dV.val)

    # Extract grid points from RELAX results
    x = relax_results.x.val
    y = relax_results.y.val
    psi = relax_results.psi.val
    Fe = relax_results.Fe.val
    ull, uxx = momenta_on_equatorial_plane(x, y)
    rho_relax = np.sqrt(relax_results.psi.val)

    # Return
    return ipsi, psi, x, y, Fe

def plot_Relax_Te_ne(folder):
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax3 = fig2.add_subplot(211)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
    ax2 = fig1.add_subplot(212, sharex=ax1)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
    ax4 = fig2.add_subplot(212, sharex=ax3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
    relax_results = fmt.fmtout(folder + 'relax_output.dat')
    rho_relax = np.sqrt(relax_results.psi.val)
    ax1.plot(rho_relax, relax_results.Tini.val, 'b--')
    ax1.plot(rho_relax, relax_results.Te.val, 'r-')
    ax1.set_xlim([min(rho_relax), max(rho_relax)])
    ax2.plot(rho_relax, 1.e-13 * relax_results.Nini.val, 'b--')
    ax2.plot(rho_relax, 1.e-13 * relax_results.Ne.val, 'r-')
    ax2.legend(('Intitial profile', 'RELAX profile'), 'lower left')
    ax2.set_ylabel(r'$N_\mathrm{e}$ [$10^{19} \mathrm{m}^{-3}$]')
    ax1.legend(('Intitial profile', 'RELAX profile'), 'lower left')
    ax1.set_ylabel(r'$T_\mathrm{e}$ [$\mathrm{keV}$]')
    ax1.set_xlabel(r'$\rho_\mathrm{poloidal}$')
    ax2.set_xlabel(r'$\rho_\mathrm{poloidal}$')
    ax3.set_xlabel(r'$\rho_\mathrm{poloidal}$')
    ax4.set_xlabel(r'$\rho_\mathrm{poloidal}$')
    ax3.plot(rho_relax, 1.e-3 * relax_results.dPec_dV.val, 'r-')
    ax4.plot(rho_relax, 1.e-6 * relax_results.Je_fsa.val, 'r-')
    ax3.set_ylabel(r'$dP/dV$ [$\mathrm{MW}/\mathrm{m}^3$]')
    ax4.set_ylabel(\
        r'$\langle J_{\mathrm{e}} \rangle$ [$\mathrm{MA}/\mathrm{m}^2$]')
    ax4.set_xlim([min(rho_relax), max(rho_relax)])
    cur = 0.0
    for i in range(len(rho_relax)):
        if(i == 0):
            cur += 0.5 * relax_results.Je_fsa.val[i] * (rho_relax[1] - rho_relax[0])
        elif(i == len(rho_relax) - 1):
            cur += 0.5 * relax_results.Je_fsa.val[i] * \
              (rho_relax[len(rho_relax) - 1] - rho_relax[len(rho_relax) - 2])
        else:
            cur += relax_results.Je_fsa.val[i] * \
              (rho_relax[i] - rho_relax[i - 1])
    print(cur)
    plt.show()



def clean_string(i):
    while True:
        if(i.startswith(" ")):
            i = i[1:len(i)]
        elif("  " in i):
            i = i.replace("  ", " ")
        else:
            break
    return i


def read_file(filename, maxlines=0, coloumn=1):
    afile = open(filename, "r")
    astring = afile.readlines()
    afile.close()
    if(maxlines == 0):
        maxlines = len(astring)
    x = []  # np.zeros(maxlines)
    y = []  # np.zeros(maxlines)
    for i in range(maxlines):
        astring[i] = clean_string(astring[i])
        try:
            x.append(float(astring[i].split(" ")[0]))
            y.append(float(astring[i].split(" ")[coloumn]))
        except ValueError:
            pass
    return np.array(x), np.array(y)

def interpolate(x, x1, x2, y1, y2):
    return y1 + (y2 - y1) \
           / (x2 - x1) * (x - x1)

def extrapolate(x, x1, x2, y1, y2):
    return y1 + (y2 - y1) \
           / (x2 - x1) * (x - x1)

def spline_interpolate(x, y, x_new, y_new):
    tck = splrep(x, y, s=0)  # s= 0 -> no smothing
    y_new = splev(x_new, tck, der=0)
    return y_new

def search_interpolate(x, x_vec, y_vec):
    i = 0
    try:
        if(x < x_vec[0]):
            return y_vec[0]
        if(x > x_vec[-1]):
            return y_vec[-1]
        while (x > x_vec[i]):
            i += 1
    except ValueError:
        print("Matrix encountered in x")
        print(type(x_vec[0]), type(x))
        print(x_vec[0], x)
    return interpolate(x, x_vec[i - 1], x_vec[i], y_vec[i - 1] , y_vec[i])

def reduce_file(path, name, new_name, lines, coloumn=1, scale=1.0, from_ida=False, skiprows=0):
    inputfile_name = os.path.join(path, name)
    outputfile_name = os.path.join(path, new_name)
    in_file = np.loadtxt(inputfile_name, skiprows=skiprows)
    x = in_file.T[0]
    y = in_file.T[1]
    # read_file(inputfile_name, coloumn = coloumn)
    # if(from_ida):
    # else:
    #    x_out = np.linspace(0.0, 1.06,lines)
    if(lines != len(x)):
        x_out = np.linspace(0.0, max(x), lines)
        spl = InterpolatedUnivariateSpline(x, y)
        y_out = spl(x_out) * scale
    else:
        x_out = x
        y_out = y * scale
    outputfile = open(outputfile_name, "w")
    if(not from_ida):
        outputfile.write("{0: 7n}\n".format(lines))
    for i in range(len(x_out)):
        if(not from_ida):
            outputfile.write("{0: 1.12E} {1: 1.12E}\n".format(x_out[i], y_out[i]))
        else:
            outputfile.write("{0: 1.12E} {1: 1.12E}\n".format(x_out[i], y_out[i]))
    outputfile.flush()
    outputfile.close()

def weighted_emissivity(folder, shot, time, ch, dstf, old=False, O_mode=False):
    Ich = "Ich" + dstf
    # folder = os.path.join(shotfolder, "30907","0.75")
    if(O_mode):
        filename_n = os.path.join(folder, "ECRad_data", Ich , "IrhoOch" + "{0:0>3}.dat".format(ch))
        filename_transparency = os.path.join(folder, "ECRad_data", Ich, "TrhoOch" + "{0:0>3}.dat".format(ch))
    else:
        filename_n = os.path.join(folder, "ECRad_data", Ich , "Irhopch" + "{0:0>3}.dat".format(ch))
        filename_transparency = os.path.join(folder, "ECRad_data", Ich, "Trhopch" + "{0:0>3}.dat".format(ch))
    rhop_Birth = []
    D = []
    if(old):
        R, j = read_file(filename_n, coloumn=4)
        R, T = read_file(filename_transparency, coloumn=2)
    else:
        R, j = read_file(filename_n, coloumn=3)
        R, T = read_file(filename_transparency, coloumn=1)
    if(O_mode):
        mode = "O"
    else:
        mode = "X"
    svec, freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch, mode=mode)
    # print(len(T*j), len(svec.T[0]))
    I = simps(T * j, svec.T[0])
    print("Trad", cnst.c ** 2 * I / (cnst.e * freq ** 2))
    for i in range(len(T)):
        if(T[i] * j[i] < 0.0):
            print("negative BDOP!", T[i], j[i], R[i])
    rhop_Birth.append([])
    D.append([])
    R_ax, z_ax = get_axis(shot, time)
    HFS = svec.T[1] < R_ax
    last_HFS = HFS[0]
    profile_cnt = 0
    for i in range(len(HFS)):
        if(last_HFS != HFS[i]):
            last_HFS = HFS[i]
            rhop_Birth.append([])
            D.append([])
            profile_cnt += 1
        if(HFS[i]):
            rhop_Birth[profile_cnt].append(-svec.T[3][i])
        else:
            rhop_Birth[profile_cnt].append(svec.T[3][i])
        D[profile_cnt].append(T[i] * j[i])
    for profile_cnt in range(len(rhop_Birth)):
        rhop_Birth[profile_cnt] = np.array(rhop_Birth[profile_cnt])
        D[profile_cnt] = np.array(D[profile_cnt]) / I
    return rhop_Birth, D


def weighted_emissivity_along_s(folder, shot, time, ch, dstf, old=False, mode='X'):
    Ich = "Ich" + dstf
    # folder = os.path.join(shotfolder, "30907","0.75")
    filename_n = os.path.join(folder, "ECRad_data", Ich , "Irhopch" + "{0:0>3}.dat".format(ch))
    filename_transparency = os.path.join(folder, "ECRad_data", Ich, "Trhopch" + "{0:0>3}.dat".format(ch))
    rhop_Birth = []
    D = []
    if(old):
        R, j = read_file(filename_n, coloumn=4)
        R, T = read_file(filename_transparency, coloumn=2)
    else:
        R, j = read_file(filename_n, coloumn=3)
        R, T = read_file(filename_transparency, coloumn=1)
    svec, freq = read_svec_from_file(os.path.join(folder, "ECRad_data"), ch, mode=mode)
    # print(len(T*j), len(svec.T[0]))
    I = simps(T * j, svec.T[0])
    print("Trad", cnst.c ** 2 * I / (cnst.e * freq ** 2))
    for i in range(len(T)):
        if(T[i] * j[i] < 0.0):
            print("negative BDOP!", T[i], j[i], R[i])
    return svec.T[0], T * j / I, svec.T[5] / 1.e3

def modify_ece_data(scale):
    ece_file = open(os.path.join(base, "residue_ece.res"), "r")
    ece_lines = ece_file.readlines()
    ece_file.close()
    ece = []
    for i in range(len(ece_lines)):
        cur_ece_array = ece_lines[i].replace("\n", "")
        final_ece_array = []
        cur_ece = np.zeros(4)
        if(len(cur_ece_array) == 48):
            for j in range(4):
                final_ece_array.append(cur_ece_array[j * 12 + 1:(j + 1) * 12 + 1])
            if(len(final_ece_array) != 4):
                print final_ece_array
                return -1
            for j in range(0, 4):
                try:
                    cur_ece[j] = np.double(final_ece_array[j])
                except ValueError:
                    print np.double(final_ece_array[j])
            ece.append(cur_ece)
    for i in range(len(ece)):
        if((i - 36) % 50 == 0):
            ece[i][1] = ece[i][1] / scale
    ece_file = open(os.path.join(base, "residue_ece_mod.res"), "w")
    for i in range(len(ece)):
        for j in range(4):
            ece_file.write(" {0: 1.4e}".format(ece[i][j]))
        ece_file.write("\n")
        if(i != 0 and (i + 1) % 50 == 0):
            ece_file.write(" \n")
    ece_file.flush()
    ece_file.close()
    return

def fix_residue_ece(target_folder):
    X = np.loadtxt(os.path.join(target_folder, "residue_ece.res"))
    rhop_cor = np.loadtxt(os.path.join(target_folder, "ECRad_data", "rhopres.dat"))
    diag_filename = os.path.join(target_folder, "ECRad_data", "diag.dat")
    diag_data = np.genfromtxt(diag_filename, dtype='str')
    for i in range(len(X.T[0])):
        X.T[0][i] = rhop_cor[i % len(rhop_cor[diag_data == "CEC"])]
    np.savetxt(os.path.join(target_folder, "residue_ece.res"), X, "% 1.5e", " ")

def scale_TRAD():
    t , Ich46 = read_file(os.path.join(base, "ECRad_data", "TRAD_46_47_44"))
    t , Ich47 = read_file(os.path.join(base, "ECRad_data", "TRAD_46_47_44"), coloumn=2)
    t , Ich44 = read_file(os.path.join(base, "ECRad_data", "TRAD_46_47_44"), coloumn=3)
    t_0 = 0
    while t_0 < len(t):
        if(t[t_0] > 1.0):
            break;
        t_0 += 1
    t_1 = t_0
    while t_1 < len(t):
        if(t[t_1] > 1.5):
            break;
        t_1 += 1
    t = t[t_0:t_1]
    Ich46 = Ich46[t_0:t_1]
    Ich47 = Ich47[t_0:t_1]
    Ich44 = Ich44[t_0:t_1]
    scale = 0.0
    diff = np.zeros(len(t))
    for i in range(len(t)):
        diff[i] = 2 * abs(Ich46[i]) / abs(Ich47[i] + Ich44[i])
        scale += diff[i] / (float(len(t)))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax_diff = fig.add_subplot(212)
    ax.plot(t, Ich46, label="ch46")
    ax.plot(t, Ich46 / scale, label="ch46 scaled")
    ax.plot(t, Ich47, label="ch47")
    ax.plot(t, Ich44, label="ch44")
    ax_diff.plot(t, diff)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="best")
    print(scale)  # 0.917818031846
    # modify_ece_data(scale)
    plt.show()

def remap_f(x, y, Fe_rhop, uxx, ull):
    spline = RectBivariateSpline(x, y, Fe_rhop, kx=5, ky=5)
    Fe_remapped = np.empty([len(ull), len(uxx)])
    for i in range(len(ull)):
        for j in range(len(uxx)):
            cur_x = np.sqrt(uxx[j] ** 2 + ull[i] ** 2)
            cur_y = np.arctan2(uxx[j], ull[i])
            # print cur_x, cur_y
            Fe_remapped[i, j] = spline.ev(cur_x, cur_y)
            # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped


def fit_TRad(args):
    Callee = args[0]
    path = args[1]
    beta = np.copy(args[2])
    fit = args[3]
    dstf = args[4]
    exec_ECRad_model = args[5]
    ECE_data = args[6]
    ECE_y_err = args[7]
    thread_Lock = Lock()
    ifixb = np.zeros(len(beta), np.int)
    sd_beta = np.zeros(len(beta))
    sd_beta[:] = -1.0  # No fit
    x = np.linspace(0, 1, len(ECE_data))
    if(dstf == "BM" or dstf == "BJ"):
        model_func = evaluate_bi_max
        if(dstf == "BJ"):
            Trad_filename = os.path.join(path, "TRadM_BiMnJ.dat")
            res_filename = os.path.join(path, "bi_maxj.res")
        else:
            Trad_filename = os.path.join(path, "TRadM_BiMax.dat")
            res_filename = os.path.join(path, "bi_max.res")
        parameter_filename = "bi_max.dat"
        model_func = evaluate_bi_max
        beta_bounds = np.array([[1.e-7, 0.0, 1.e-3, 5.e3, 5.e3], [1.0, 1.0, np.inf, 5.e5, 5.e5]])
    elif(dstf == "DM"):
        Trad_filename = os.path.join(path, "TRadM_Drift.dat")
        parameter_filename = "drift_m.dat"
        res_filename = os.path.join(path, "Drift_max.res")
    elif(dstf == "MS"):
        model_func = evaluate_multi_slope
        Trad_filename = os.path.join(path, "TRadM_MultS.dat")
        parameter_filename = "multi_s.dat"
        res_filename = os.path.join(path, "multi_s.res")
        beta_bounds = np.array([[-100, 0.9999], [1.0, 1.5]])
    elif(dstf == "RA"):
        model_func = evaluate_runaway
        Trad_filename = os.path.join(path, "TRadM_RunAw.dat")
        res_filename = os.path.join(path, "run_away.res")
        parameter_filename = "runaway.dat"
    else:
        print("Invalid value for dstf", dstf)
        raise(ValueError)
    parameter_filename = os.path.join(path, parameter_filename)
    fun_args = {}
    fun_args["exec_ECRad_model"] = exec_ECRad_model
    fun_args["parameter_filename"] = parameter_filename
    fun_args["Trad_filename"] = Trad_filename
    os.environ['OMP_NUM_THREADS'] = "24"
    os.environ['OMP_STACKSIZE'] = "{0:d}".format(int(np.ceil(10000 * 3.125) * 3))
    if(fit):
        thread_Lock.acquire()
        res = least_squares(model_func, beta, bounds=beta_bounds, \
                            args=[exec_ECRad_model, Trad_filename, parameter_filename, ECE_data, ECE_y_err])
        print(res.message)
        thread_Lock.release()
        beta = res.x
        sd_beta = np.zeros(len(beta))
        if(res.success):
            print("Fit successfull")
        else:
            print("Fit failed")
            print("status: ", res.status)
    else:
        if(dstf == "BM"):
            state = make_bi_max(beta, parameter_filename)
        elif(dstf == "BJ"):
            state = make_bi_max(beta, parameter_filename)
        elif(dstf == "DM"):
            state = make_drift_m(beta, parameter_filename)
        elif(dstf == "MS"):
            state = make_multi_slope(beta, parameter_filename)
        elif(dstf == "RA"):
            state = make_runaway(beta, parameter_filename)
        if(state):
            print("Parametrized distribution ready")
        else:
            print("Error when preparing parametrization")
    res_file = open(res_filename, "w")
    for i in range(len(beta)):
        res_file.write("{0:1.5e} \t {1:1.5e} \n".format(beta[i], sd_beta[i]))
    res_file.flush()
    res_file.close()
    evt_out = ThreadFinishedEvt(Unbound_EVT_FIT_FINISHED, Callee.GetId())
    wx.PostEvent(Callee, evt_out)

def evaluate_bi_max(beta, exec_ECRad_model, trad_filename, bi_max_filename, ECE_data, ECE_y_err):
    if(make_bi_max(beta, bi_max_filename)):
        ECRad = Popen(exec_ECRad_model)
        sleep(0.1)
        os.system("renice -n 10 -p " + "{0:d}".format(ECRad.pid))
        stderr_log = []
        while(ECRad.poll() is None):
            stdout, stderr = ECRad.communicate(None)
            stderr_log.append(stderr)
            print(stdout)
            sleep(0.25)
        for stderr in stderr_log:
            print(stderr)
    else:
        print("Fit failed")
        return
    Trad = np.loadtxt(trad_filename)
    print("beta", beta)
    print(("Trad", Trad.T[1]))
    print("residues", (Trad.T[1] - ECE_data) / ECE_y_err)
    print("Sum of squares", np.sum((Trad.T[1] - ECE_data) ** 2 / ECE_y_err) ** 2)
    return (Trad.T[1] - ECE_data) / ECE_y_err

def make_bi_max(beta, bi_max_filename):
    bi_file = open(bi_max_filename, "w")
    Te_par = beta[3]
    Te_perp = beta[4]
    rhop = np.linspace(0.0, 1.0, 100)
    bi_file.write("{0: 1.8E}{1: 1.8E}\n".format(Te_par, Te_perp))
    bi_file.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        bi_file.write("{0: 1.8E}{1: 1.8E}\n"\
            .format(rhop[i], Gauss_norm(rhop[i], \
                beta)))  # j[i]
        if(Gauss_norm(rhop[i], beta) > 1.0 or Gauss_norm(rhop[i], beta) < 0.0):
            return False
    bi_file.flush()
    bi_file.close()
    return True

def evaluate_drift_m(beta, x, exec_efcm_model, trad_filename, drift_m_filename):
    make_drift_m(beta, x, drift_m_filename)
    call(exec_efcm_model)
    return read_file(trad_filename)[1]

def make_drift_m(beta, drift_m_filename):
    drift_m = open(drift_m_filename, "w")
    rhop = np.linspace(0.0, 1.0, 100)
    drift_m.write("{0: 1.8E}{1: 1.8E}{2: 1.8E}{3: 1.8E}"\
            .format(beta[3], beta[4], beta[5], beta[6]))
    drift_m.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        drift_m.write("{0: 1.8E}{1: 1.8E}"\
            .format(rhop[i], Gauss_norm(rhop[i], beta)))
        if(Gauss_norm(rhop[i], beta) > 1.0 or Gauss_norm(rhop[i], beta) < 0.0):
            return False
    drift_m.flush()
    drift_m.close()
    return True

def evaluate_multi_slope(beta, exec_ECRad_model, trad_filename, multi_slope_filename, ECE_data, ECE_y_err):
    if(make_multi_slope(beta, multi_slope_filename)):
        ECRad = Popen(exec_ECRad_model)
        sleep(0.1)
        os.system("renice -n 10 -p " + "{0:d}".format(ECRad.pid))
#        stderr_log = []
        while(ECRad.poll() is None):
#            stdout, stderr = ECRad.communicate(None)
#            stderr_log.append(stderr)
#            print(stdout)
            sleep(0.25)
#        for stderr in stderr_log:
#            print(stderr)
    else:
        print("Fit failed")
        return
    Trad = np.loadtxt(trad_filename)
    return  (Trad.T[1] - ECE_data) / ECE_y_err

def make_multi_slope(beta, multi_slope_filename):
    multi_slope_file = open(multi_slope_filename, "w")
    rhop = np.linspace(0.0, 1.0, 100)
    rhop_Te, Te = np.loadtxt(os.path.join(os.path.dirname(multi_slope_filename), "Te_file.dat"), skiprows=1, unpack=True)
    Te_spline = InterpolatedUnivariateSpline(rhop_Te, Te, k=1)
    scale = Te_spline(rhop) / np.max(Te)
    multi_slope_file.write("{0: 1.8E}\n".format(beta[1]))
    multi_slope_file.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        multi_slope_file.write("{0: 1.8E}{1: 1.8E}\n"\
                          .format(rhop[i], (1.0 - beta[0] * scale[i] ** 2) * Te_spline(rhop[i])))
    multi_slope_file.flush()
    multi_slope_file.close()
    return True

def evaluate_runaway(beta, x, invoke_ECRad, trad_filename, runaway_filename):
    # print("param set", beta)
    if(make_runaway(beta, runaway_filename)):
        call(invoke_ECRad)
    Trad = np.loadtxt(trad_filename)
    return Trad.T[1]

def make_runaway(beta, runaway_filename):
    run_file = open(runaway_filename, "w")
    rhop = np.linspace(0.0, 1.0, 100)
    run_file.write("{0: 1.8E}{1: 1.8E}\n"\
            .format(beta[3], beta[4]))  # j[i]
    run_file.write("{0:3d}\n".format(len(rhop)))
    for i in range(len(rhop)):
        run_file.write("{0: 1.8E}{1: 1.8E}\n"\
            .format(rhop[i], Gauss_norm(rhop[i], \
                beta)))  # j[i]
        if(Gauss_norm(rhop[i], beta) < 0.0):  # Gauss_norm(rhop[i], beta) > 1.0 or
            return False
    run_file.flush()
    run_file.close()
    return True

def cyc_momentum(cos_theta):
    u_par = np.linspace(0, 1.5, 200)
    gamma = np.sqrt(1 + u_par ** 2)
    ratio_mass = 1.0 / gamma
    ratio_doppl = 1.0 / (1.0 - cos_theta * u_par / gamma)
    ratio_both = 1.0 / (gamma - cos_theta * u_par)
    limit = (cos_theta * np.sqrt((1.0 - cos_theta) * (1.0 + cos_theta))) / (1.0 - cos_theta ** 2)
    return u_par, ratio_mass, ratio_doppl, ratio_both, limit

def cycl_distribution(Te, cos_theta):
    mu = (cnst.c ** 2 * cnst.m_e) / (cnst.e * Te)
    int_limit = 10.e0 / np.sqrt(mu * np.pi)
    print("int limit", int_limit)
    gamma_limit = np.sqrt(1.0 + int_limit ** 2)
    omega_mass = np.linspace(1.0 / gamma_limit, 1.0 - 1.e-8, 600)
    f_omega_mass = []
    for omega in omega_mass:
        u_par = np.linspace(-np.sqrt(1.0 / omega ** 2 - 1.e0) * (1.0 - 1.e-8), \
                        np.sqrt(1.0 / omega ** 2 - 1.e0) * (1.0 - 1.e-5), 400)
        u_perp = np.sqrt(1.e0 / omega ** 2 - 1.e0 - u_par ** 2)
        jacobi = 2.0 * np.pi * np.abs(1 + u_par ** 2 + u_perp ** 2) ** 1.5
        f_omega_mass.append(simps(Juettner2D(u_par, u_perp, Te) * jacobi, u_par))
    f_omega_mass = np.array(f_omega_mass)
    print("Norm mass", simps(f_omega_mass, omega_mass))
    print("std dev mass", simps(f_omega_mass * omega_mass ** 2, omega_mass))
    omega_doppler = np.linspace(np.max([(1.0 + 1.e-8) / (1.0 + np.abs(cos_theta)), 1.0 / (1.0 + np.abs(cos_theta * int_limit / gamma_limit))]), 1.0 - 1.e-8, 600)
    f_omega_doppler = []
    u_perp = np.linspace(1.e-5, int_limit, 400)
    for omega in omega_doppler:
        u_par = -np.sqrt(((1.e0 + u_perp ** 2) * (-1.e0 + omega) ** 2) / (-1.e0 + 2.e0 * omega + (-1.e0 + cos_theta ** 2) * omega ** 2))
        jacobi = (2.e0 * np.pi * u_perp * np.sqrt(1.e0 + u_par ** 2 + u_perp ** 2) * \
                   (-(cos_theta * u_par) + np.sqrt(1.e0 + u_par ** 2 + u_perp ** 2)) ** 2) / \
                   (cos_theta * (1 + u_perp ** 2))
        f_omega_doppler.append(simps(Juettner2D(u_par, u_perp, Te) * jacobi, u_perp))
    f_omega_doppler = np.array(f_omega_doppler)
    print("Norm Dppler", 2.e0 * simps(f_omega_doppler, omega_doppler))
    print("std dev Doppler", 2.e0 * simps(f_omega_doppler * omega_doppler ** 2, omega_doppler))
    omega_doppler = np.concatenate([omega_doppler, 2.0 - omega_doppler[::-1]])
    # np.linspace(np.max([(1.0 + 1.e-8)/(1.0 -  np.abs(cos_theta)),1.0/(1.0 - np.abs(cos_theta * int_limit/gamma_limit))]),1.0 - 1.e-8,600)
    f_omega_doppler = np.concatenate([f_omega_doppler, f_omega_doppler[::-1]])
    omega_total = np.linspace(1.0 / gamma_limit, 1.0 / (1.0 - cos_theta ** 2), 600)
    f_total = []
    omega_red = []
    for omega in omega_total:
        res = get_resonance(1.0 / omega, cos_theta, points=600)
        if(res is not None):
            f_total.append(simps(Juettner2D(res[0], res[1], Te) * 2.0 * np.pi * (1.e0 + res[0] ** 2 + res[1] ** 2) / omega, res[0]))  # np.sqrt(1 + res[1]**2 + res[0]**2)
            omega_red.append(omega)
    f_total = np.array(f_total)
    omega_red = np.array(omega_red)
    print("Norm total", simps(f_total, omega_red))
    print("std dev Total", simps(f_total * omega_red ** 2, omega_red))
    min_omega = np.min([np.min(omega_mass), np.min(omega_doppler), np.min(omega_red)])
    max_omega = np.max([np.max(omega_mass), np.max(omega_doppler), np.max(omega_red)])
    print("Range", min_omega, max_omega)
    # Extend min
    if(min_omega < np.min(omega_mass)):
        omega_mass = np.concatenate([np.linspace(min_omega, np.min(omega_mass), 50), omega_mass])
        f_omega_mass = np.concatenate([np.zeros(50), f_omega_mass])
    if(min_omega < np.min(omega_doppler)):
        omega_doppler = np.concatenate([np.linspace(min_omega, np.min(omega_doppler), 50), omega_doppler])
        f_omega_doppler = np.concatenate([np.zeros(50), f_omega_doppler])
    if(min_omega < np.min(omega_red)):
        omega_red = np.concatenate([np.linspace(min_omega, np.min(omega_red), 50), omega_red])
        f_total = np.concatenate([np.zeros(50), f_total])
    # Extend max
    if(max_omega > np.max(omega_mass)):
        omega_mass = np.concatenate([omega_mass, np.linspace(np.max(omega_mass), max_omega, 50)])
        f_omega_mass = np.concatenate([f_omega_mass, np.zeros(50)])
    if(max_omega > np.max(omega_doppler)):
        omega_doppler = np.concatenate([omega_doppler, np.linspace(np.max(omega_doppler), max_omega, 50)])
        f_omega_doppler = np.concatenate([f_omega_doppler, np.zeros(50)])
    if(max_omega > np.max(omega_red)):
        omega_red = np.concatenate([omega_red, np.linspace(np.max(omega_red), max_omega, 50)])
        f_total = np.concatenate([f_total, np.zeros(50)])
    return omega_mass, omega_doppler, omega_red, f_omega_mass, f_omega_doppler, f_total, 1.0 / np.sqrt(1.0 - cos_theta ** 2)



def rel_thermal_beta(mu, u=[0]):
    return np.sqrt(1.e0 - (kve(1, mu) / kve(2, mu) + 3.e0 / mu) ** (-2)) - u[0]

def relax_time(E_kin, Te, ne):
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

def Coloumb_log(Te, ne):
    omega_p = np.sqrt(ne * cnst.e ** 2 / (cnst.m_e * cnst.epsilon_0))
#    print("Plasma freuquency [GHz]", omega_p / (2.0 * np.pi))
    theta_min = cnst.hbar * omega_p / (Te * cnst.e)
    Clog = -np.log(np.tan(theta_min / 2.0))
#    print("Coloumb log", Clog)
    return Clog

def Gauss_norm(x, beta):
    return np.sqrt(1.0 / (np.pi * beta[2] ** 2)) * beta[0] * np.exp(-((x - beta[1]) / beta[2]) ** 2)

def Gauss_not_norm(x, beta):
    return beta[0] * np.exp(-((x - beta[1]) / beta[2]) ** 2)


def Dreicer_field_and_E_crit(ne, Te, E, Log_Lambda=22.e0):
    # ne in units of 1.e19 m^-3 and Te in keV
    v_th = rel_thermal_beta(cnst.c ** 2 * cnst.m_e / (Te * 1.e3 * cnst.e)) * cnst.c
    nu = ne * 1.e19 * cnst.e ** 4 * Log_Lambda / (4.e0 * np.pi * cnst.epsilon_0 ** 2 * cnst.m_e ** 2 * cnst.c ** 3)
    print("nu", nu)
    E_D = nu * cnst.m_e ** 2 * cnst.c ** 3 / (Te * 1.e3 * cnst.e ** 2)
    v_c = v_th * np.sqrt(E_D / (2 * E))
    E_c = 1.e-3 * cnst.m_e * cnst.c ** 2 / cnst.e * (1.e0 / np.sqrt(1.0 - v_c ** 2 / cnst.c ** 2) - 1.e0)
    print("Dreicer Field [V/m]: {0:1.4e}, critical energy [keV]: {1:3.4e}".format(E_D, E_c))

def read_beta_from_file(beta_filename):
    beta_file = open(beta_filename, "r")
    beta_lines = beta_file.readlines()
    beta_file.close()
    beta = np.zeros([len(beta_lines), 10])
    for i in range(len(beta_lines)):
        cur_beta_array = beta_lines[i].replace("\n", "")
        final_beta_array = []
        for j in range(11):
            final_beta_array.append(cur_beta_array[j * 15:(j + 1) * 15])
        if(len(final_beta_array) != 11):
            print final_beta_array
            return -1
        for j in range(1, 11):
            try:
                beta[i][j - 1] = np.double(final_beta_array[j])
            except ValueError:
                print np.double(final_beta_array[j])
    return beta

def read_svec_from_file(folder, ich, mode="X"):  # ch no. starts from 1
    # ich is here channel nummer - i.e. channel 1 is the first channel -> add + 1 if ich comes from loop
    if(mode == "O"):
        ch_filename = os.path.join(folder, "chOdata{0:0>3}.dat".format(ich))
    else:
        ch_filename = os.path.join(folder, "chdata{0:0>3}.dat".format(ich))
    try:
        svec = np.loadtxt(ch_filename)
    except ValueError as e:
        print(e)
        print("Channel ", ich)
        return
    freq_filename = os.path.join(folder, "f_ECE.dat")
    freq_file = open(freq_filename)
    freq_data = freq_file.readlines()
    freq_file.close()
    freq = np.double(freq_data[ich - 1])
    # print(freq)
    return svec, freq  # Index 2 = rhop
    # Index 6 = theta
    # Index 7 = freq_2X

def read_svec_dict_from_file(folder, ich, mode="X"):  # ch no. starts from 1
    # ich is here channel nummer - i.e. channel 1 is the first channel -> add + 1 if ich comes from loop
    if(mode == "O"):
        ch_filename = os.path.join(folder, "chOdata{0:0>3}.dat".format(ich))
        mode = -1
    else:
        ch_filename = os.path.join(folder, "chdata{0:0>3}.dat".format(ich))
        mode = +1
    try:
        svec_block = np.loadtxt(ch_filename)
    except ValueError as e:
        print(e)
        print("Channel ", ich)
        return
    freqs = np.loadtxt(os.path.join(folder, "f_ECE.dat"))
    svec = {}
    svec["s"] = svec_block.T[0][svec_block.T[3] != -1.0]
    svec["R"] = svec_block.T[1][svec_block.T[3] != -1.0]
    svec["z"] = svec_block.T[2][svec_block.T[3] != -1.0]
    svec["rhop"] = svec_block.T[3][svec_block.T[3] != -1.0]
    svec["ne"] = svec_block.T[4][svec_block.T[3] != -1.0]
    svec["Te"] = svec_block.T[5][svec_block.T[3] != -1.0]
    svec["theta"] = svec_block.T[6][svec_block.T[3] != -1.0]
    Abs_obj = em_abs_Alb()
    svec["freq_2X"] = svec_block.T[-1][svec_block.T[3] != -1.0]
    svec["N_abs"] = []
    for i in range(len(svec["s"])):
        svec_cur = s_vec(svec["rhop"][i], svec["Te"][i], svec["ne"][i], svec["freq_2X"][i], svec["theta"][i])
        N = Abs_obj.refr_index(svec_cur, freqs[ich - 1] * 2.0 * np.pi, mode)
        svec["N_abs"].append(N)
    svec["N_abs"] = np.array(svec["N_abs"])
    return svec, freqs[ich - 1]

def read_ray_dict_from_file(folder, dist, ich, mode="X", iray=1):
    if(mode == "O"):
        ray_filename = os.path.join(folder, "Ich" + dist, "BPD_ray{0:03d}ch{1:03d}_O.dat".format(iray, ich))
    else:
        ray_filename = os.path.join(folder, "Ich" + dist, "BPD_ray{0:03d}ch{1:03d}_X.dat".format(iray, ich))
    ray_data = np.loadtxt(ray_filename)
    ray_dict = {}
    ray_dict["s"] = ray_data.T[0][ray_data.T[4] != -1.0]
    ray_dict["x"] = ray_data.T[1][ray_data.T[4] != -1.0]
    ray_dict["y"] = ray_data.T[2][ray_data.T[4] != -1.0]
    ray_dict["z"] = ray_data.T[3][ray_data.T[4] != -1.0]
    ray_dict["rhop"] = ray_data.T[4][ray_data.T[4] != -1.0]
    ray_dict["BPD"] = ray_data.T[5][ray_data.T[4] != -1.0]
    ray_dict["BPD_second"] = ray_data.T[6][ray_data.T[4] != -1.0]
    ray_dict["N_ray"] = ray_data.T[9][ray_data.T[4] != -1.0]
    ray_dict["N_cold"] = ray_data.T[10][ray_data.T[4] != -1.0]
    ray_dict["theta"] = ray_data.T[11][ray_data.T[4] != -1.0]
    if(len(ray_data[0]) <= 12):
        spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["x"])
        ray_dict["Nx"] = spl.derivative(1)(ray_dict["s"])
        spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["y"])
        ray_dict["Ny"] = spl.derivative(1)(ray_dict["s"])
        spl = InterpolatedUnivariateSpline(ray_dict["s"], ray_dict["z"])
        ray_dict["Nz"] = spl.derivative(1)(ray_dict["s"])
        norm = ray_dict["N_ray"] / np.sqrt(ray_dict["Nx"] ** 2 + ray_dict["Ny"] ** 2 + ray_dict["Nz"] ** 2)
        ray_dict["Nx"] *= norm
        ray_dict["Ny"] *= norm
        ray_dict["Nz"] *= norm
        ray_dict["Bx"] = None
        ray_dict["By"] = None
        ray_dict["Bz"] = None
    else:
        ray_dict["Nx"] = ray_data.T[12][ray_data.T[4] != -1.0]
        ray_dict["Ny"] = ray_data.T[13][ray_data.T[4] != -1.0]
        ray_dict["Nz"] = ray_data.T[14][ray_data.T[4] != -1.0]
        ray_dict["Bx"] = ray_data.T[15][ray_data.T[4] != -1.0]
        ray_dict["By"] = ray_data.T[16][ray_data.T[4] != -1.0]
        ray_dict["Bz"] = ray_data.T[17][ray_data.T[4] != -1.0]
#    plt.plot(ray_dict["s"], np.sqrt(ray_dict["Nx"] ** 2 + ray_dict["Ny"] ** 2 + ray_dict["Nz"] ** 2), "-")
#    plt.plot(ray_dict["s"], ray_dict["N_ray"], "--")
#    plt.show()
    return ray_dict

def get_B_min_from_file(folder):
    return np.loadtxt(os.path.join(folder, "B_min.dat"), unpack=True)

def find_value_occurence(array, value):
    indices = []
    i = 0
    while(i < len(array) - 1):
        if(array[i] < value and array[i + 1] > value):
            indices.append(i)
        if(array[i] > value and array[i + 1] < value):
            indices.append(i)
        if(array[i] == value):
            indices.append(i)
        i += 1
    if(array[i] == value):
        indices.append(i)
    return indices

def find_index(array, value, i_guess=0):
    if(value > np.max(array)):
        return None
    if(value < np.min(array)):
        return 0
    if(i_guess == 0):
        i = 0
    else:
        while array[i_guess] > value:
            i_guess = int(float(i_guess) / 2.0)
        i = i_guess
    while(i < len(array)):
        if(array[i] > value):
            return i
        i += 1

def save_log(f, f_out):
    for j in range(len(f)):
        for k in range(len(f[j])):
            if(np.log(f[j, k]) == -np.inf):
                f_out[j, k] = -1.e5
            elif(abs(f[j, k]) < 0.0005):
                f_out[j, k] = -1.e5
            else:
                f_out[j, k] = -np.log(f[j, k])
    return f_out

def is_ch_on_hfs(R_ax, rpath, ich, rel_res):
    rpath_data = os.path.join(rpath, "ECRad_data")
    svec, ece_freq = read_svec_from_file(rpath_data, ich)
    if(not rel_res):
        i_res = np.where(np.min(np.abs(ece_freq - svec.T[-1])) == np.abs(ece_freq - svec.T[-1]))[0][0]
        if(R_ax > svec.T[1][i_res]):
            return True
        else:
            return False
    else:
        R_rel = np.loadtxt(os.path.join(rpath_data, "sres_rel.dat")).T[1][ich - 1]
        # print(rhop_min, svec.T[7][I_rhop_min])
        if(R_ax > R_rel):
            return True
        else:
            return False

def identify_LFS_channels(time, rpath, ch_num, EQ_obj, rel_res=False):
    R_ax, z_ax = EQ_obj.get_axis(time)
    LFS_channel = np.zeros(ch_num, np.int)
    for ich in range(1, ch_num + 1):
        if(is_ch_on_hfs(R_ax, rpath, ich, rel_res)):
            LFS_channel[ich - 1] = 0
        else:
            LFS_channel[ich - 1] = 1
    return LFS_channel

def find_rel_rhop_res(rpath, ich, sres):
    rpath_data = os.path.join(rpath, "ECRad_data")
    svec, ece_freq = read_svec_from_file(rpath_data, ich)
    i_rhop = np.abs(svec.T[0] - sres).argmin()
    return svec.T[3][i_rhop]

def rel_rhop_res_all_ch(rpath):
    rpath_data = os.path.join(rpath, "ECRad_data")
    res = np.loadtxt(os.path.join(rpath_data, "sres_rel.dat"))
    return res.T[3]

def make_R_res(rpath, ch_num):
    rpath_data = os.path.join(rpath, "ECRad_data")
    R_res = np.zeros(ch_num, np.double)
    for ich in range(1, ch_num + 1):
        svec, ece_freq = read_svec_from_file(rpath_data, ich)
        for i in range(1, len(svec)):
            if(ece_freq > svec.T[8][i] and ece_freq > svec.T[8][i - 1]):
                R_res[ich - 1] = interpolate(ece_freq, svec.T[8][i - 1], svec.T[8][i], \
                                    svec.T[1][i - 1], svec.T[1][i])
    return R_res

def remap_rhop_R(rpath, rhop, R_mag, ich=1):
    rpath_data = os.path.join(rpath, "ECRad_data")
    svec, ece_freq = read_svec_from_file(rpath_data, ich)
    Hfs_rhop_max = svec.T[3][0]
    print(Hfs_rhop_max)
    Lfs_rhop_max = svec.T[3][-1]
    i_Hfs_rhop_max = len(rhop) - 1
    while Hfs_rhop_max < rhop[i_Hfs_rhop_max]:
        i_Hfs_rhop_max -= 1
    print(rhop[i_Hfs_rhop_max])
    i_Lfs_rhop_max = len(rhop) - 1
    while Lfs_rhop_max < rhop[i_Lfs_rhop_max]:
        i_Lfs_rhop_max -= 1
    rhop_min = np.min(svec.T[3])
    i_rhop_min = 0
    while rhop[i_rhop_min] < rhop_min:
        i_rhop_min += 1
    index_center = np.argmin(svec.T[3])
    R_Hfs = svec.T[1][index_center:0:-1]
    rhop_Hfs = svec.T[3][index_center:0:-1]
    R_Lfs = svec.T[1][index_center:len(svec.T[1])]
    rhop_Lfs = svec.T[3][index_center:len(svec.T[1])]
    s_Hfs = InterpolatedUnivariateSpline(rhop_Hfs, R_Hfs)
    s_Lfs = InterpolatedUnivariateSpline(rhop_Lfs, R_Lfs)
    R = np.zeros(i_Hfs_rhop_max + i_Lfs_rhop_max)
    R[0:i_Hfs_rhop_max - i_rhop_min] = s_Hfs(rhop[i_rhop_min:i_Hfs_rhop_max])[::-1]
    R[i_Hfs_rhop_max + i_rhop_min:i_Hfs_rhop_max + i_Lfs_rhop_max] = s_Lfs(rhop[i_rhop_min:i_Lfs_rhop_max])
    for irhop in range(i_rhop_min, 0, -1):
        R[i_Hfs_rhop_max - irhop] = interpolate(rhop[irhop], rhop_min, 0.e0, \
                                                R[i_Hfs_rhop_max - i_rhop_min - 1], R[i_Hfs_rhop_max + 2 * rhop_min - 1])
    for irhop in range(0, i_rhop_min):
        R[i_Hfs_rhop_max + irhop] = interpolate(rhop[irhop], 0.e0, rhop_min, \
                                                R[i_Hfs_rhop_max - i_rhop_min - 1], R[i_Hfs_rhop_max + 2 * rhop_min - 1])
    return R, i_Hfs_rhop_max, i_Lfs_rhop_max

def get_Te_ne_R(rpath, ich=1):
    rpath_data = os.path.join(rpath, "ECRad_data")
    svec, ece_freq = read_svec_from_file(rpath_data, ich)
    return svec.T[1][::10], svec.T[5][::10] * 1.e-3, svec.T[4][::10] * 1.e-19


def find_cold_res(rpath, ich, mode="X", harmonic_number=2):
    svec, ece_freq = read_svec_dict_from_file(rpath, ich, mode)
    s = svec["s"][svec["rhop"] != 0.0]
    f_c = svec["freq_2X"][svec["rhop"] != 0.0] / 2.0
    rhop = svec["rhop"][svec["rhop"] != 0.0]
    R = svec["R"][svec["rhop"] != 0.0]
    z = svec["z"][svec["rhop"] != 0.0]
    res_spl = InterpolatedUnivariateSpline(s, harmonic_number * f_c - ece_freq)
    roots = res_spl.roots()
    if(len(roots) == 0):
        print("No roots for selected resonance - returning ECRad resonances")
        s_res = np.loadtxt(os.path.join(rpath, "sres.dat"))
        return False, s_res[ich - 1][0], s_res[ich - 1][1], s_res[ich - 1][2], s_res[ich - 1][3]
    else:
        R_spl = InterpolatedUnivariateSpline(s, R)
        z_spl = InterpolatedUnivariateSpline(s, z)
        rhop_spl = InterpolatedUnivariateSpline(s, rhop)
        s_res = np.max(roots)
        return True, s_res, R_spl(s_res), z_spl(s_res), rhop_spl(s_res)
# def find_cold_res(rpath, ich, mode="X"):
#    # ich > 0
#    try:
#        svec, ece_freq = read_svec_from_file(rpath, ich, mode)
#        svec = svec[svec.T[3] != 0.0]
#        svec_ref = svec[::-1]
#        # Assumes lfs launch!
#        N_larger = np.where(svec_ref.T[-1] > ece_freq)[0][0]
#        s = svec_ref.T[0][0:N_larger + 1]
#        res = svec_ref.T[-1][0:N_larger + 1] - ece_freq
#        res_spl = InterpolatedUnivariateSpline(s[::-1], res[::-1])
#        s_res = res_spl.roots()[0]
#        R = interpolate(s_res, svec_ref.T[0][N_larger], svec_ref.T[0][N_larger - 1], \
#                        svec_ref.T[1][N_larger], svec_ref.T[1][N_larger - 1])
#        z = interpolate(s_res, svec_ref.T[0][N_larger], svec_ref.T[0][N_larger - 1], \
#                        svec_ref.T[2][N_larger], svec_ref.T[2][N_larger - 1])
#    except:
#        sres = np.loadtxt(os.path.join(rpath, "sres.dat"))
#        try:
#            R = sres.T[1][ich - 1]
#            z = sres.T[2][ich - 1]
#        except IndexError:
#            if(len(np.shape(sres)) > 1):
#                sres = sres.T[0]
#            i = np.argmin(np.abs(svec.T[0] - sres[ich - 1]))
#            if(i - 1 <= 0):
#                R = extrapolate(sres[ich - 1], svec.T[0][i], svec.T[0][i + 1], \
#                                svec.T[1][i - 1], svec.T[1][i])
#                z = extrapolate(sres[ich - 1], svec.T[0][i], svec.T[0][i + 1], \
#                                svec.T[2][i - 1], svec.T[2][i])
#            elif(i >= len(svec.T[0]) - 1):
#                R = extrapolate(sres[ich - 1], svec.T[0][i - 1], svec.T[0][i], \
#                                svec.T[1][i - 1], svec.T[1][i])
#                z = extrapolate(sres[ich - 1], svec.T[0][i - 1], svec.T[0][i], \
#                                svec.T[2][i - 1], svec.T[2][i])
#            elif(sres[ich - 1] > svec.T[0][i]):
#                R = interpolate(sres[ich - 1], svec.T[0][i - 1], svec.T[0][i], \
#                                svec.T[1][i - 1], svec.T[1][i])
#                z = interpolate(sres[ich - 1], svec.T[0][i - 1], svec.T[0][i], \
#                                svec.T[2][i - 1], svec.T[2][i])
#            else:
#                R = interpolate(sres[ich - 1], svec.T[0][i], svec.T[0][i + 1], \
#                                svec.T[1][i], svec.T[1][i + 1])
#                z = interpolate(sres[ich - 1], svec.T[0][i], svec.T[0][i + 1], \
#                                svec.T[2][i], svec.T[2][i + 1])
#    return R, z


def find_rel_res(rpath, ich):
    # ich > 0
    sres = np.loadtxt(os.path.join(rpath, "sres_rel.dat"))
    # return find_value_occurence(svec.T[0],sres[ich])
    return sres.T[1][ich - 1], sres.T[2][ich - 1]


def get_resonance_cnt(n_omega_bar, cos_theta, cnt):
    theta = np.arccos(cos_theta)
    if((n_omega_bar) ** 2 < np.sin(theta) ** 2):
        return None
    else:
        ull_min = (n_omega_bar * np.cos(theta) - np.sqrt((n_omega_bar) ** 2 - \
            np.sin(theta) ** 2)) / np.sin(theta) ** 2
        ull_max = (n_omega_bar * np.cos(theta) + np.sqrt((n_omega_bar) ** 2 - \
            np.sin(theta) ** 2)) / np.sin(theta) ** 2
        delta = (ull_max - ull_min) * 1.e-9
        ull_min += delta
        ull_max -= delta
        ull_res = np.linspace(ull_min, ull_max, cnt)
        uxx_res = np.sqrt((n_omega_bar + np.cos(theta) * ull_res) ** 2 - 1.0 - ull_res ** 2)
        # temp = ull_res / np.sqrt(1.0 + ull_res**2 + uxx_res**2)
        # uxx_res = uxx_res / np.sqrt(1.0 + ull_res**2 + uxx_res**2)
        # ull_res = temp
        return np.array([ull_res, uxx_res])

def get_resonance(n_omega_bar, cos_theta, int_range=None, points=200):
    theta = np.arccos(cos_theta)
    if((n_omega_bar) ** 2 < np.sin(theta) ** 2):
        # print("No Resonance")
        return None
    else:
        ull_min = (n_omega_bar * np.cos(theta) - np.sqrt((n_omega_bar) ** 2 - \
            np.sin(theta) ** 2)) / np.sin(theta) ** 2
        ull_max = (n_omega_bar * np.cos(theta) + np.sqrt((n_omega_bar) ** 2 - \
            np.sin(theta) ** 2)) / np.sin(theta) ** 2
        delta = (ull_max - ull_min) * 1.e-9
        ull_min += delta
        ull_max -= delta
        if(int_range is not None):
            ull_min = np.max([ull_min, int_range[0]])
            ull_max = np.min([ull_max, int_range[1]])
        ull_res = np.linspace(ull_min, ull_max, points)
        uxx_res = np.sqrt((n_omega_bar + np.cos(theta) * ull_res) ** 2 - 1.0 - ull_res ** 2)
        # temp = ull_res / np.sqrt(1.0 + ull_res**2 + uxx_res**2)
        # uxx_res = uxx_res / np.sqrt(1.0 + ull_res**2 + uxx_res**2)
        # ull_res = temp
        # print("Found resonance")
        return np.array([ull_res, uxx_res])

def get_resonance_N(n_omega_bar, N_par):
    if((n_omega_bar) ** 2 + N_par ** 2 <= 1.0):
        print("No Resonance")
        return None
    else:
        ull_min = (-n_omega_bar * N_par - np.sqrt(n_omega_bar ** 2 + N_par ** 2 - 1.e0)) / (N_par ** 2 - 1.e0)
        ull_max = (-n_omega_bar * N_par + np.sqrt(n_omega_bar ** 2 + N_par ** 2 - 1.e0)) / (N_par ** 2 - 1.e0)
        # print(ull_min, ull_max)
        if(ull_min > ull_max):
            temp = ull_min
            ull_min = ull_max
            ull_max = temp
        delta = (ull_max - ull_min) * 1.e-9
        ull_min += delta
        ull_max -= delta
        ull_res = np.linspace(ull_min, ull_max, 200)
        uxx_res = np.sqrt((N_par * ull_res + n_omega_bar) ** 2 - 1.e0 - ull_res ** 2)
        print("Found resonance")
        return np.array([ull_res, uxx_res])

def plot_B_along_los(fig, fig2, rpath, rhop_in, ich, shot, time, dstf, mode="X"):
    rpath_data = os.path.join(rpath, "ECRad_data")
    svec, ece_freq = read_svec_from_file(rpath_data, ich, mode="X")
    print("f_ECE", ece_freq)
    # B = svec.T[7] * cnst.m_e / cnst.e / np.pi
    f_c_2X = svec.T[8] * 1.e-9  # * 1.04
    f_c_3X = svec.T[8] * 1.e-9 * 1.5  # * 1.04
    # i = find_value_occurence(svec.T[3],min(svec.T[3]))[0] + 20
    freq = np.zeros(len(svec.T[0]))
    freq[:] = ece_freq * 1.e-9
    # for j in range(len(freq_center)):
    #    freq_center[j] = svec.T[8][i]* 1.e-9 * 1.5
    # k = find_value_occurence(f_c_2X,freq_center[0])[0]
    ax1 = fig.add_subplot(111)
    # freq_center[:] = ece_freq * 1.e-9
    ax1.plot(svec.T[1], f_c_2X, "--g", label=r"$f_\mathrm{2X}$",)
    ax1.plot(svec.T[1], f_c_3X, ":b" , label=r"$f_\mathrm{3X}$")
    ax1.plot(svec.T[1], freq, "-r", label=r"$f_\mathrm{ECE}$")
    # ax1.vlines(svec.T[3][k],-1e5,1e5)
    ax1.set_xlabel(r"$\rho_\mathrm{pol}$")
    ax1.set_ylim(0, 1.3 * max(f_c_3X))
    # ax1.set_ylabel(r"$\vert B \vert /$ T")
    ax1.set_ylabel(r"$f_\mathrm{c} /$ Ghz")
    handles, labels = ax1.get_legend_handles_labels()
    leg = ax1.legend(handles, labels, loc="upper left")
    return fig, fig2

def get_B(ECRad_data, ich):
    svec, ece_freq = read_svec_from_file(ECRad_data, ich)
    B = svec.T[8] * cnst.m_e / cnst.e * np.pi
    return svec.T[1], B

def get_omega_c_and_cutoff(ECRad_data, ich, mode):
    svec, ece_freq = read_svec_from_file(ECRad_data, ich, mode)
    f_c_1X = svec.T[-1][svec.T[3] != -1.0] * 5.e-10
    f_c_2X = svec.T[-1][svec.T[3] != -1.0] * 1.e-9  # * 1.04
    f_c_3X = svec.T[-1][svec.T[3] != -1.0] * 1.e-9 * 1.5  # * 1.04
    freq = np.zeros(len(svec.T[0][svec.T[3] != -1.0]))
    freq[:] = ece_freq * 1.e-9
    arr = np.array([svec.T[1][svec.T[3] != -1.0], f_c_1X, f_c_2X, f_c_3X, freq])
    omega_p = np.sqrt(svec.T[4][svec.T[3] != -1.0] * cnst.e ** 2 / (cnst.epsilon_0 * cnst.m_e)) * 1.e-9
    f_R = arr[1] * (np.sqrt(1.0 + 4.0 * omega_p ** 2 / (arr[1] * 2.0 * np.pi) ** 2) + 1.0) / 2.0
    return f_R, omega_p, arr


def load_f_from_mat(filename):
    mdict = loadmat(filename, squeeze_me=True)
    return distribution(mdict["rhot_prof"], mdict["rhop_prof"], mdict["u"], mdict["pitch"], mdict["f"], mdict["rhot_1D_profs"], mdict["rhop_1D_profs"], mdict["Te_init"], mdict["ne_init"])

def load_f_from_ASCII(path, rhop_in=None, Gene=False):
    if(not Gene):
        x = np.loadtxt(os.path.join(path, "u.dat"), skiprows=1)
        y = np.loadtxt(os.path.join(path, "pitch.dat"), skiprows=1)
        ne_data = np.loadtxt(os.path.join(path, "..", "ne_file.dat"), skiprows=1)
        Te_data = np.loadtxt(os.path.join(path, "..", "Te_file.dat"), skiprows=1)
        rhop_ne = ne_data.T[0]
        ne = ne_data.T[1]
        Te = Te_data.T[1]
        rhop = np.loadtxt(os.path.join(path, "frhop.dat"), skiprows=1)
        rhop_B_min, B_min = get_B_min_from_file(os.path.join(path, ".."))
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
        return distribution(None, rhop, x, y, np.exp(Fe), None, rhop_ne, Te, ne, B_min=B_min)
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



def Sf(u_par, u_perp, theta, n_omega_bar, f_res, ne, freq, m):
    gamma = n_omega_bar + np.cos(theta) * u_par
    zeta = n_omega_bar / float(m) * u_perp * np.sin(theta)
    zeta *= gamma
    Sf = ((np.cos(theta) - u_par / gamma) * jn(2, zeta) / np.sin(theta)) ** 2
    Sf = Sf + (u_perp / gamma) ** 2 * (jn(1, zeta) - jn(3, zeta)) ** 2 / 4.e0
    Sf = Sf * f_res * (gamma ** 2)
    Sf *= (cnst.e) ** 2.e0 * freq * ne / (cnst.epsilon_0 * cnst.c) * 2.0 * np.pi
    return Sf

def Sf_approx(u_par, u_perp, theta, n_omega_bar, f_res, ne, freq, m):
    gamma = n_omega_bar + np.cos(theta) * u_par
    fac = 1
    # (m - 1)!
    for i in range(2, m):
        fac *= i
    Sf = (u_perp / gamma / 2.e0) ** (2 * m)
    Sf = Sf * f_res * (gamma ** 2)
    Sf = Sf * m ** (2 * (m - 1)) / fac ** 2 * np.sin(theta) ** (2 * (m - 1)) * (np.cos(theta) ** 2 + 1.e0)
    Sf *= (cnst.e) ** 2.e0 * freq * ne / (cnst.epsilon_0 * cnst.c) * 2.0 * np.pi
    return Sf


# def Sf_wrong(u_par, u_perp, theta, n_omega_bar, f_res):
#    gamma = n_omega_bar + np.cos(theta) * u_par
#    zeta = 2.0 * n_omega_bar * u_perp * np.sin(theta)
#    Sf = ((np.cos(theta) - u_par / gamma) * jn(2, zeta / np.sin(theta))) ** 2
#    Sf = Sf + (u_perp / gamma) ** 2 * (jn(1, zeta) - jn(3, zeta)) ** 2 / 4.e0
#    Sf = Sf * f_res * (gamma ** 2)
#    return Sf

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

def Write_ped_data(args):
    rpath = args[0]
    distpath = args[1]
    mode = args[2]
    wpath = os.path.join(rpath, "ECRad_data")
    ipsi, psi, x, y, Fe = read_Fe(rpath + "/ECRad_data/")
    step_ll = 1.0 / len(x)
    step_xx = step_ll / 1.0 * 0.5
    rhop_vec_Te, Te_vec = read_file(rpath + "/te_ida.res")
    ull = np.arange(-0.5, 0.5, step_ll)
    uxx = np.arange(0.0, 0.5, step_xx)
    print(len(ull), len(uxx))
    u = np.zeros([2, len(ull)])
    u[0] = ull[:]
    u[1] = uxx[:]
    f = np.zeros([len(ull), len(uxx)])
    if(mode == 0):
        ull_spitzer = np.arange(-0.3, 0.3, step_ll)
        beta = np.zeros([len(psi), 10])
        beta = test_fit(ipsi, psi, x, y, Fe, ull, uxx, ull_spitzer, rhop_vec_Te, Te_vec, beta)
        betafile = open(os.path.join(wpath, "beta.dat"), "w")
        for i in range(len(psi)):
            betafile.write("{0: 1.8E}".format(np.sqrt(psi[i])))
            for j in range(len(beta[i])):
                betafile.write("{0: 1.8E}".format(beta[i][j]))
            betafile.write("\n")
        betafile.close()
    else:
        beta = read_beta_from_file(os.path.join(wpath, "beta.dat"))
    rhopfile = open(os.path.join(distpath, "frhop.dat"), "w")
    uxxfile = open(os.path.join(distpath, "u_perp.dat"), "w")
    ullfile = open(os.path.join(distpath, "u_par.dat"), "w")
    for i in range(len(uxx)):
        uxxfile.write("{0: 1.8f}\n".format(uxx[i]))
    for i in range(len(ull)):
        ullfile.write("{0: 1.8f}\n".format(ull[i]))
    ullfile.flush()
    ullfile.close()
    uxxfile.flush()
    uxxfile.close()
    for irhop in range(len(psi)):
        rhopfile.write("{0: 1.5f}\n".format(np.sqrt(psi[irhop])))
        f = Juettner2D_ped(beta[irhop], u, f, len(ull))
        thefile = open(os.path.join(distpath, "fu{0:0>3}.dat".format(irhop)), "w")
        for j in range(len(ull)):
            for k in range(len(uxx)):
                if(np.log(f[j, k]) == -np.inf):
                    thefile.write("{0: 1.8E}".format(-1.e5))
                elif(f[j, k] < 0.0):
                    thefile.write("{0: 1.8E}".format(-1.e5))
                else:
                    thefile.write("{0: 1.8E}".format(np.log(f[j, k])))
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()
    print("ECRH-Pedstal data ready")

def prep_data(rpath):
    ipsi, psi, x, y, Fe = read_Fe(rpath + "/ECRad_data/")
    step_ll = 1.0 / len(x) * 0.5
    step_xx = step_ll / 1.0
    ull = np.arange(-0.2, 0.2, step_ll)
    uxx = np.arange(0.0, 0.4, step_xx)
    u = np.zeros([2, len(ull)])
    u[0] = ull[:]
    u[1] = uxx[:]
    f_cur = np.empty([len(ull), len(uxx)])
    f_cur = remap_f(x, y, Fe[ipsi], uxx, ull)
    return u, f_cur

def test_paras(u, f_cur, beta):
    middle = np.floor(float(len(u[0])) / 2.0)
    f = np.zeros([len(u[0]), len(u[1])])
    f = Juettner2D_ped(beta, u, f, len(u[0]))
    fig = plt.figure(figsize=(16, 8), tight_layout=True)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    cont = ax1.contour(u[0], u[1], f.T, 50)
    cont = ax2.contour(u[0], u[1], f_cur.T, 50)
    ax3.plot(u[1], f[middle])
    ax3.plot(u[1], f_cur[middle], "x")
    plt.clabel(cont, inline=1, fontsize=10)
    plt.show()

def fix_Te_perp(Te_perp, Te_par, f_zero):
    sol = scopt.root(Juettner_BiNorm, Te_perp, (Te_par, f_zero))
    Te_perp = sol.x
    print(sol.x)
    print(sol.message)
    return Te_perp

# def Juettner_BiNorm(Te_perp, Te_par, f_zero):
#    f = ct.c_double(0.0)
#    libjuettner.Juettner_BiNorm(ct.c_double(Te_par), ct.c_double(Te_perp), \
#        ct.byref(f))
#    return f.value - f_zero

def test_fit(ipsi, psi, x, y, Fe, ull, uxx, ull_spitzer, rhop_vec_Te, Te_vec, beta):
    # overwrite = r"C:\Users\Severin\Documents\IPP-Job\IDA_GUI_Ext\\"
    # ipsi, psi, x, y, Fe = read_Fe(os.path.join(base ,"ECRad_data" + os.path.sep))
    # rhop_vec_Te, Te_vec = read_file(os.path.join(base, "te_ida.res"))
    middle = np.floor(float(len(ull)) / 2.0)
    f = np.zeros([len(ull), len(uxx)])  # [len(ull),len(uxx)]
    v_drift_fit = np.zeros(len(psi))
    ifixb = np.zeros(10, np.int)
    ifixb[0:9] = 1
    u = np.zeros([2, len(ull)])
    u[0] = ull[:]
    u[1] = uxx[:]
    ifixb[0] = 1
    # ifixb[6] = 0
    ifixb[9] = 0
    f_cur = np.zeros([len(ull), len(uxx)])
    f_cur_spitzer = np.zeros(len(ull_spitzer))
    f_spitzer = np.zeros(len(ull_spitzer))
    Te = search_interpolate(np.sqrt(psi[0]), rhop_vec_Te, Te_vec) * 1000
    # beta[ipsi] =  np.array([5.8062037e+03,6.0142311e+03, -1.2633224e-04, \
    #        1.008289e-00, 1.507261e-01,  3.434952e-02,1.0e-02,\
    #        1.5e-02,-0.188e00,1.301e00,0.0])
    beta[ipsi] = np.array([5.83630846e+03, 6.03443864e+03, -1.84568692e+01, \
                           1.74526182e+00, 1.57237785e-01, 7.52228365e-02, \
                           2.42768291e-02 , 2.88952643e-02, \
                           9.06966889e-01, -9.32832333e+06])
    beta[len(psi) - 1] = np.array([8.88815063E+02, 8.88975119E+02, 3.24025790E+14, \
                      - 1.69152974E+14, 4.72190823E-01, -1.86632425E+15, \
                      5.02845671E+00, 7.91547160E-05, \
                      1.62455082E+02, -5.57734810E+05])
    beta[0] = np.array([ 4.96041714E+03, 5.29865536E+03, \
                        1.16613816E+00, 3.83136593E-01, 3.12981221E-01, \
                        7.19918751E-02, 9.20217695E-02, 9.30554111E-03, \
                        - 1.17081950E-01, -1.01785877E+07])
    # ifixb[3] = 0
    ifixb[7] = 1
    # ifixb[9] = 0
    dir = -1
    fig1 = plt.figure(figsize=(16, 9), tight_layout=True)
    fig2 = plt.figure(figsize=(16, 9), tight_layout=True)
    fig3 = plt.figure(figsize=(16, 9), tight_layout=True)
    fig4 = plt.figure(figsize=(16, 9), tight_layout=True)
    for dir in [-1, 1]:
    # if(True):
        irhop_max = len(psi)  # len(psi)
        if(dir == -1):
            irhop_max = 0
        for irhop in range(ipsi, irhop_max + dir, dir):
        # for irhop in [0, len(psi) - 1]:
            f_cur = remap_f(x, y, Fe[irhop], uxx, ull, f_cur)
            f_cur_spitzer = remap_f1D(x, y, Fe[irhop], 0.0, ull_spitzer, f_cur_spitzer)
            ifixb[0:10] = 1
            ifixb[1:9] = 0
            beta[irhop][9] = -cnst.c * 0.03
            # if(irhop != ipsi):
            #    beta[irhop] = beta[irhop - dir]
            ax11 = fig4.add_subplot(211)
            f_spitzer = Juettner_biSpitzer(beta[irhop], ull_spitzer, \
                f_spitzer, len(ull_spitzer))
            ax11.plot(ull_spitzer, f_spitzer)
            ax11.plot(ull_spitzer, f_cur_spitzer, "x")
            beta[irhop] = fit_biSpitzer(f_cur_spitzer, ull_spitzer, 0.0, Te, \
                f_spitzer, len(ull), beta[irhop], ifixb)
            ax12 = fig4.add_subplot(212)
            f_spitzer = Juettner_biSpitzer(beta[irhop], ull_spitzer, \
                f_spitzer, len(ull_spitzer))
            ax12.plot(ull_spitzer, f_spitzer)
            ax12.plot(ull_spitzer, f_cur_spitzer, "x")
            ifixb[0:10] = 0
            ifixb[0:9] = 1
            beta[irhop][1] = fix_Te_perp(beta[irhop][0], beta[irhop][1],
                f_cur[middle][0])
            # if(abs(beta[irhop][2]) < 1.0 or abs(beta[irhop][3]) < 1.0):
            #    ifixb[0:8] = 0
            #    ifixb[1] = 1
            #    beta[irhop][2] = 0.0
            #    beta[irhop][3] = 0.0
            # else:
            #    ifixb[0:8] = 1
            #    ifixb[0] = 0
            #    ifixb[7] = 0
            # ax10 = fig4.add_subplot(211)
            # ax11 = fig4.add_subplot(212)
            # fig3.suptitle("Initial Values")
                # Gauss_not_norm(np.sqrt(psi[irhop]),[1.1,0.2,0.2])
            # print(beta)
            fig1.clf()
            fig3.clf()
            ax1 = fig1.add_subplot(311)
            ax2 = fig1.add_subplot(312)
            ax3 = fig1.add_subplot(313)
            ax7 = fig3.add_subplot(311)
            ax8 = fig3.add_subplot(312)
            ax9 = fig3.add_subplot(313)
            f = Juettner2D_ped(beta[irhop], u, f, len(ull))
            cont3 = ax3.contour(ull, uxx, f.T, 50)
            plt.clabel(cont3, inline=1, fontsize=10)
            # ax3.set_xlim(-0.3, 0.3)
            ax3.set_ylim(0.0, 0.33)
            ax1.plot(ull, uxx, "-g")
            ax7.plot(uxx, f_cur[middle, :], "x")
            ax7.plot(uxx, f[middle])
            ax8.plot(uxx, f_cur[middle + 3, :], "x")
            ax8.plot(uxx, f[middle + 3, :])
            ax9.plot(uxx, f_cur[middle + 6, :], "x")
            ax9.plot(uxx, f[middle + 6, :])
            # ax10.plot(uxx, f_cur[:,0],"x")
            # ax10.plot(uxx,f[:,0])
            f, beta[irhop] = fit_Pedestal(f_cur, u, Te, f, len(ull), beta[irhop], ifixb)
            amp = Gauss_not_norm(np.sqrt(psi[irhop]), [1.0, np.sqrt(psi[ipsi]), 0.2])
            pos = 0.12 + 0.32 * np.exp((np.sqrt(psi[ipsi]) - np.sqrt(psi[irhop])) / 0.085 - 2.0)
            if(irhop > 1):
            # and abs(beta[irhop][2]) > 1.0 and \
             #   abs(beta[irhop][3])  > 1.0):
                beta[irhop - 1] = beta[irhop]
            # elif(irhop < len(psi) - 1):
            #    beta[irhop + 1] =  np.array([Te_par,Te + 750,2.6e1*amp,1.3e1*amp, pos,\
            #    0.03,1.4e0,v_drift_fit[irhop]])
            cont1 = ax1.contour(ull, uxx, f_cur.T, 50)
            plt.clabel(cont1, inline=1, fontsize=10)
            cont2 = ax2.contour(ull, uxx, f.T, 50)
            plt.clabel(cont2, inline=1, fontsize=10)
            # ax1.set_xlim(-0.3, 0.3)
            ax1.set_ylim(0.0, 0.33)
            # ax2.set_xlim(-0.3, 0.3)
            ax2.set_ylim(0.0, 0.33)
            fig2.clf()
            ax4 = fig2.add_subplot(311)
            ax5 = fig2.add_subplot(312)
            ax6 = fig2.add_subplot(313)
            ax4.plot(uxx, f_cur[middle, :], "x")
            ax4.plot(uxx, f[middle])
            ax5.plot(uxx, f_cur[middle + 3, :], "x")
            ax5.plot(uxx, f[middle + 3, :])
            ax6.plot(uxx, f_cur[middle + 6 , :], "x")
            ax6.plot(uxx, f[middle + 6, :])
            # ax11.plot(uxx, f_cur[:,0],"x")
            # ax11.plot(uxx,f[:,0])
            fig1.suptitle("Rhop = {0:1.3f}".format(np.sqrt(psi[irhop])))
            # plt.pause(1)

            plt.show()
    # ax4.plot(uxx, f_cur[middle + 3,:],"x")
    # ax4.plot(uxx,f[middle + 3,:])
    # ax5.plot(uxx, f_cur[middle + 6,:],"x")
    # ax5.plot(uxx,f[middle + 6,:])
    # ax6.plot(uxx, f_cur[middle ,:],"x")
    # ax6.plot(uxx,f[middle,:])
    # fig3 = plt.figure(figsize = (16,9),tight_layout=True)
    # ax7 = fig3.add_subplot(311)
    # ax8 = fig3.add_subplot(312)
    # ax9 = fig3.add_subplot(313)
    # f = Juettner2D_ped( beta,u, f,len(ull))
    # beta =  np.array([Te + 1000,7.0e4,1.1e4, 0.16, 5.5e-2,1.0e-1,3.e-1])
    # u = np.zeros([2,len(ull)])
    # u[0,:] = ull[:]
    # u[1,:] = uxx[:]
    # ax3.contour(ull, uxx,f.T, 50)
    # ax3.set_xlim(-0.2, 0.2)
    # ax3.set_ylim(0.0, 0.35)
    # ax7.plot(uxx, f_cur[middle + 3,:],"x")
    # ax7.plot(uxx,f[middle + 3,:])
    # ax8.plot(uxx, f_cur[middle + 6,:],"x")
    # ax8.plot(uxx,f[middle + 6,:])
    # ax9.plot(uxx, f_cur[middle ,:],"x")
    # ax9.plot(uxx,f[middle,:])
    return beta

def Damped_Resonance(freq, width):
    x = np.arange(0.00, 0.3, 0.01)
    y = width / ((freq - x) ** 2 + width ** 2)
    plt.plot(x, y)
    plt.show()

def Arctan_step(freq, width, u_perp, u_res):
    x = np.arange(-0.3, 0.3, 0.01)
    y = (1.0 / np.pi * np.arctan((u_res + (u_perp - u_res) / freq - x) / width) + 0.5) * \
    (1.0 / np.pi * np.arctan((u_res + (u_perp - u_res) / freq + x) / width) + 0.5)
    plt.plot(x, y)
    plt.show()

def fit_biSpitzer(f, ull, uxx, Te , f_fit, m, beta, ifixb):
    biSpitzer = odr.Model(Juettner_biSpitzer, extra_args=[f_fit, m])
    relax_data = odr.Data(ull, f)
    ODR_Fit = odr.ODR(relax_data, biSpitzer, beta, ifixb=ifixb, iprint=1111)
    ODR_Fit.set_job(fit_type=2)
    results = ODR_Fit.run()
    print(beta)
    print(results.beta)
    return results.beta

def fit_Pedestal(f, u, Te , f_fit, m, beta, ifixb):
    # Te = 5000.0
    # params = [Te]
    # middle = np.floor(float(len(ull)) / 2.0) - 1
    # print f[:,0]
    # popt, pcov = curve_fit(Juettner1D, ull,f[:,0],params)
    # y = Juettner1D(ull, popt[0])
    # fig = plt.figure(figsize = (16,9),tight_layout=True)
    # ax = fig.add_subplot(111)
    # plt.title(r"#27764 t = 2.4s. Fit of thermal distribution at rhop = {:1.2f}.".format(rhop))
    # y_ida = Juettner1D(ull,Te_ida)
    # ax.plot(ull, y, "--g", label = r"Fit of thermal distribution $u_\bot = 0 $ with $T_e = ${0:4.1f}".format(popt[0]))
    # ax.plot(ull, f[:,0], "+r", label = r"$f(u_\bot = 0 ,u_\Vert)$ numerical Data")
    # ax.plot(ull, y_ida, "-b",label = r"Thermal distribution using IDA $T_e =  ${0:4.1f}".format(Te_ida))
    # print(libjuettner.simple_test())
    # exit(1)
    # u[1,:] = uxx[:]
    # u = np.array(uxx)
    # beta =  np.array([Te,7.0e1,1.1e1, 0.16, 5.5e-2,1.0e-02,5.8e-02])
    # f_fit = Juettner2D_ped( beta,u, f_fit,len(ull))
    # return f_fit
    pedestal = odr.Model(Juettner2D_ped, extra_args=[f_fit, m])
    relax_data = odr.Data(u, f)
    # f_fit = Juettner2D_ped( params,[ull,uxx])
    # return f_fit
    # if(ifixb[2] != 0):
    #    ifixb[2] = 0
    #    ifixb[3] = 0
    #    ODR_Fit = odr.ODR(relax_data, pedestal,beta,ifixb=ifixb,iprint = 1111)
    #    ODR_Fit.set_job(fit_type = 0)
    #    results = ODR_Fit.run()
    #    if('Numerical error detected' in results.stopreason):
    #        print(results.stopreason)
    #        f_fit = Juettner2D_ped( beta,u, f_fit,m)
    #        return f_fit, beta
    #    beta = results.beta
    #    ifixb[2] = 1
    #    ifixb[3] = 1
    ODR_Fit = odr.ODR(relax_data, pedestal, beta, ifixb=ifixb, iprint=1111)
    ODR_Fit.set_job(fit_type=2)
    results = ODR_Fit.run()
    beta_new = results.beta
    ODR_Fit = odr.ODR(relax_data, pedestal, beta_new, ifixb=ifixb, iprint=1111)
    ODR_Fit.set_job(fit_type=2)
    results = ODR_Fit.run()
    # ODR_Fit.set_iprint(init = 2,iter=2,final=2)



    if('Numerical error detected' in results.stopreason):
        print(results.stopreason)
        f_fit = Juettner2D_ped(beta, u, f_fit, m)
        return f_fit, beta
    print beta
    beta = results.beta
    print beta
    # y = Juettner1D_ped(uxx,  params[0], params[1], params[2], params[3],params[4])
    # ax.plot(uxx, y, "--m",label = r"$u_\Vert = 0$ with $T_e = ${0:4.1f}".format(params[0]))
    # ax2.plot(uxx, f[middle,:], "+r", label = r"$f(u_\bot,u_\Vert = 0)$ numerical Data".format(params[0]))
    # try:
    # popt, pcov = curve_fit(Juettner2D_ped, [uxx,ull],f[:,:],params)
    # retry = False
    # y = Juettner2D_ped([uxx,ull],  popt[0], popt[1], popt[2], popt[3],popt[4], popt[4])
    # ax.plot(uxx, y, "--")
    # ax.plot(uxx, f[middle,:])
    # if(params[-1] > 1000.0 or params[2] > 0.3 or params[2]  < 0.1):
    #    retry = True
    # else:
    # for i in range(len(ull)):
    f_fit = Juettner2D_ped(beta, u, f_fit, m)
    # print(popt)
    return f_fit, beta
    # except RuntimeError:
    #    retry = True
    # if(retry):
    #    for i in range(len(ull)):
    #        f_fit[i] =Juettner2D_fit(uxx,ull[i],Te)
    # y_ida =Juettner1D(uxx,Te_ida)
    # print(popt)
    # ax.plot(uxx, y, "--")
    # ax.plot(uxx, f[k,:])
    # ax2.plot(uxx, y_ida, "-b", label = r"Thermal distribution using IDA $T_e =  ${0:4.1f}".format(Te_ida))
    # handles, labels = ax.get_legend_handles_labels()
    # leg = ax.legend(handles, labels, loc = "best")
    # handles, labels = ax2.get_legend_handles_labels()
    # leg2 = ax2.legend(handles, labels, loc = "best")
    # plt.show()
    return f_fit

# def Juettner2D_ped(pars, u, f, m):
#    libjuettner.Juettner2D_ped(pars.ctypes.data_as(ct.POINTER(ct.c_double)), \
#        u.ctypes.data_as(ct.POINTER(ct.c_double)), f.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_long(m))
#    return f
#
# def Juettner1D_ped_res(pars, u, f, m):
#    libjuettner.Juettner1D_ped_res(pars.ctypes.data_as(ct.POINTER(ct.c_double)), \
#        u.ctypes.data_as(ct.POINTER(ct.c_double)), f.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_long(m))
#    return f
#
# def Juettner_biSpitzer(pars, u, f, m):
#    libjuettner.Juettner_BiSpitzer(pars.ctypes.data_as(ct.POINTER(ct.c_double)), \
#        u.ctypes.data_as(ct.POINTER(ct.c_double)), f.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_long(m))
#    return f
#
# def simple_lib_test():
#    ar = np.zeros([3, 4])
#    m = 3
#    n = 4
#    for i in range(m):
#        for j in range(n):
#            ar[i, j] = (-1) ** i * i * m + j
#    print(ar)
#    c_ar = ar.astype(np.double)
#    _c_ar = ar.ctypes.data_as(ct.POINTER(ct.c_double))
#    print(libjuettner.simple_test(_c_ar, ct.c_long(m), ct.c_long(n)))


def SpizerHaermDist1D(u, Te, sh_obj, Z, R, v_therm):
    v = u / (np.sqrt(1 + u ** 2))
    return (1.0 + np.sign(v) * sh_obj.deform_D(R, Z, abs(v / v_therm))) * \
        Juettner1D(u, Te)


def find_vd(args):
    rpath = args[0]
    wpath = args[1]
    rhop_vec_j = args[2]
    j_vec = args[3]
    rhop_vec_Te, Te_vec = read_file(rpath + "/te_ida.res")
    rhop_vec_ne, ne_vec = read_file(rpath + "/ne_ida.res")
    ipsi, psi, x, y, Fe = read_Fe(rpath + "/ECRad_data/")
    rhop = np.sqrt(psi)
    # plt.plot(rhop_vec_ne, ne_vec * 1.e20)
    # return
    Te = np.zeros(len(rhop))
    ne = np.zeros(len(rhop))
    j = np.zeros(len(rhop))
    r = np.zeros(len(rhop))
    # theta = np.zeros(len(rhop))
    u_drift = np.zeros(len(rhop))
    # rhop_vec_j,j_vec =  read_file(j_loc)
    for i in range(len(rhop)):
        Te[i] = search_interpolate(rhop[i], rhop_vec_Te, Te_vec) * 1000.0
        ne[i] = search_interpolate(rhop[i], rhop_vec_ne, ne_vec) * 1.e20
        j[i] = search_interpolate(rhop[i], rhop_vec_j, j_vec)
        # theta[i] = search_interpolate(rhop[i], rhop_vec_r, theta_vec)
        # r[i] = search_interpolate(rhop[i], rhop_vec_r, r_vec)
    # plt.plot(rhop_vec_j, j_vec)
    # plt.plot(rhop, j)
    # return
    delta_omega = np.zeros(len(rhop))
    u_drift_sh = np.zeros(len(rhop))
    u_drift_Relax = np.zeros(len(rhop))
    u_step = 1.0 / 128.0
    u = np.arange(-1.0, 1.0, u_step)
    # u2 = np.arange(-0.5,0.5,u_step/2.0)
    uxx = np.arange(0, 1.0, u_step)
    ull = np.arange(-1.0, 1.0, u_step)
    sh_obj = sh.SpitzerHaerm(1)
    norm_sh = np.zeros(len(rhop))
    norm_Relax = np.zeros(len(rhop))
    params = [1.0, 1.0, 1.0, y[0]]
    u_drift_fit = np.zeros(len(rhop))
    u_drift_fit_int = np.zeros(len(rhop))
    norm_fit_int = np.zeros(len(rhop))
    Te_fit = np.zeros(len(rhop))
    u_par_step = 1.0 / 128
    u_perp_step = 1.0 / 128
    u_par = np.arange(-1.0, 1.0, u_par_step)
    u_perp = np.arange(0.0, 1.0, u_perp_step)
    rhopfile = open(wpath + "/frhop.dat", "w")
    u_perp_file = open(wpath + "/u_perp.dat", "w")
    u_par_file = open(wpath + "/u_par.dat", "w")
    for i in range(len(u_perp)):
        u_perp_file.write("{0: 1.8f}\n".format(u_perp[i]))
    for i in range(len(u_par)):
        u_par_file.write("{0: 1.8f}\n".format(u_par[i]))
    u_par_file.flush()
    u_perp_file.flush()
    u_par_file.close()
    u_perp_file.close()
    f_cur = np.empty([len(ull), len(uxx)])
    for i in range(len(rhop)):
        cur_f = remap_f(x, y, Fe[i], u_perp, u_par, f_cur)
        u_th = np.sqrt(1 - \
            cnst.physical_constants["electron mass"][0] ** 2 * \
            cnst.physical_constants["speed of light in vacuum"][0] ** 4 / \
            (cnst.physical_constants["electron mass"][0] * \
            cnst.physical_constants["speed of light in vacuum"][0] ** 2 + \
            cnst.physical_constants["elementary charge"][0] * Te[i]) ** 2)
        v_th = u_th / (np.sqrt(1 + u_th ** 2)) * \
            cnst.physical_constants["speed of light in vacuum"][0]
        u_drift[i] = j[i] / (-cnst.physical_constants["elementary charge"][0] * ne[i])
        u_drift[i] = u_drift[i] * np.sqrt(1.0 - (u_drift[i] / cnst.c) ** 2) / cnst.c
        R = j[i] / (cnst.physical_constants["elementary charge"][0] * ne[i]) / \
            v_th
        vd = j[i] / (cnst.physical_constants["elementary charge"][0] * ne[i])
        Z = 1
        params = [vd, Te[i]]
        popt, pcov = curve_fit(SpizerHaermDist1D_static, u, cur_f[:, 0], params)
        u_drift_fit[i] = popt[0] / cnst.c
        Te_fit[i] = popt[1]
        # plt.plot(u,SpizerHaermDist1D_static(u,popt[0], popt[1]))
        # plt.plot(u,cur_f[:,0])
        # return u, SpizerHaermDist1D_static(u,popt[0], popt[1]),cur_f[:,0],SpizerHaermDist1D_static(u,vd,Te[i])
        u_drift_sh[i] = (SpizerHaermDist1D(u[0], Te[i], sh_obj, Z, R, v_th) * u[0] + \
             u[-1] * SpizerHaermDist1D(u[-1], Te[i], sh_obj, Z, R, v_th)) * u_step * 0.5
        norm_sh[i] = (SpizerHaermDist1D(u[0], Te[i], sh_obj, Z, R, v_th) + \
                SpizerHaermDist1D(u[-1], Te[i], sh_obj, Z, R, v_th)) * u_step * 0.5
        u_drift_Relax[i] = cur_f[0, 0] * u_step * 0.5 * u[0] + \
                           cur_f[-1, 0] * u_step * 0.5 * u[-1]
        norm_Relax[i] = cur_f[0, 0] * u_step * 0.5 + \
                           cur_f[-1, 0] * u_step * 0.5
        u_drift_fit_int[i] = (SpizerHaermDist1D_static(u[0], popt[0], popt[1]) * u[0] + \
             u[-1] * SpizerHaermDist1D_static(u[-1], popt[0], popt[1])) * u_step * 0.5
        norm_fit_int[i] = (SpizerHaermDist1D_static(u[0], popt[0], popt[1]) + \
                SpizerHaermDist1D_static(u[-1], popt[0], popt[1])) * u_step * 0.5
        for k in range(1, len(u) - 1):
            u_drift_sh[i] += SpizerHaermDist1D(u[k], Te[i], sh_obj, Z, R, v_th) \
                *u_step * u[k]
            norm_sh[i] += SpizerHaermDist1D(u[k], Te[i], sh_obj, Z, R, v_th) \
                *u_step
            u_drift_Relax[i] += u[k] * cur_f[k, 0] * u_step * 0.5
            norm_Relax[i] += cur_f[k, 0] * u_step * 0.5
        for k in range(1, len(u) - 1):
            u_drift_fit_int[i] += SpizerHaermDist1D_static(u[k], popt[0], popt[1]) \
                *u_step * u[k]
            norm_fit_int[i] += SpizerHaermDist1D_static(u[k], popt[0], popt[1]) \
                *u_step
        rhopfile.write("{0: 1.5f}\n".format(rhop[i]))
        thefile = open(wpath + "/fu{0:0>3}.dat".format(i), "w")
        for k in range(len(u_par)):
            for l in range(len(u_perp)):
                f = SpizerHaermDist2D_static(u_par[k], u_perp[l], popt[0], popt[1])
                if(np.log(f) == -np.inf):
                    thefile.write("{0: 1.8E}".format(-1.e5))
                elif(f < 0):
                    thefile.write("{0: 1.8E}".format(-1.e5))
                else:
                    thefile.write("{0: 1.8E}".format(np.log(f)))
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()
    u_drift_sh = u_drift_sh / norm_sh
    # print u_drift_sh
    # norm_sh = norm_sh[i1:i2]
    # rhop = rhop[i1:i2]
    u_drift_Relax = u_drift_Relax / norm_Relax
    u_drift_fit_int = u_drift_fit_int / norm_fit_int
    # norm_Relax = norm_Relax[i1:i2]
    # print(u_drift/norm)
    print(Te, Te_fit)
    return rhop, np.abs(u_drift), np.abs(u_drift_sh), np.abs(u_drift_Relax), \
        np.abs(u_drift_fit), np.abs(u_drift_fit_int)
    # plt.plot(rhop,u_drift_sh/norm_sh,"-b")
    # plt.plot(rhop,u_drift_Relax/norm_Relax,"-g")
    # plt.plot(rhop_vec_ne,ne_vec,"x")
    # plt.plot(rhop,Te)

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


def Juettner2D_cycl(u, Te):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / (abs(Te) * cnst.e)
    a = 1.0 / (1 + 105 / (128 * mu ** 2) + 15 / (8 * mu))
    gamma = np.sqrt(1 + u ** 2)
    return a * gamma * u * mu * np.exp(mu * (1 - gamma)) / 2.0
""" 
def SpizerHaermDist1D(u, Te, sh_obj, Z, R,v_therm):
    v = u/(np.sqrt(1 + u**2)) * cnst.c
    return ( 1.0 + np.sign(v) * sh_obj.evaluate(R,abs(v/v_therm))) * \
        Juettner1D(u, Te) 
"""
def SpizerHaermDist1D_static(u, vd, Te):
    u_th = np.sqrt(2 * abs(Te) / 510998.910 + (Te / 510998.910) ** 2)
    R = vd / cnst.c / u_th * np.sqrt(1.0 + u_th ** 2)
    v = (u / np.sqrt(1.0 + u ** 2)) / (u_th / np.sqrt(1.0 + u_th ** 2))
    return (1.0 + np.sign(v) * R * SpitzerD_over_R(abs(v)))  # * \
        # Juettner1D(u, Te)

def SpitzerD_over_R(v):
    return 0.7619 * (0.09476232 * v ** 4 - 0.08852586 * v ** 3 + 1.32003051 * v ** 2 - 0.19511956 * v)

def SpizerHaermDist2D_static(u_par, u_perp, vd, Te):
    u_th = np.sqrt(2 * abs(Te) / 510998.910 + (Te / 510998.910) ** 2)
    v_th = u_th / (np.sqrt(1 + u_th ** 2)) * \
            cnst.physical_constants["speed of light in vacuum"][0]
    v = u_par / (np.sqrt(1 + u_par ** 2 + u_perp ** 2)) * cnst.c
    R = vd / \
            v_th
    return (1.0 - np.sign(v) * R * sh.static_evaluate(abs(v / v_th)))  # * \
        # Juettner2D(u_par, u_perp, Te)


def make_Sh_deform(rhop, j, rpath):
    rhop_vec, Te_vec = read_file(rpath + "/te_ida.res")
    rhop_vec_ne, ne_vec = read_file(rpath + "/te_ida.res")
    Te = Te_vec[0] * 1.e3
    ne = ne_vec[0] * 1.e20
    u_par = np.arange(-1.0, 1.0, 1.e-3)
    deform = SpizerHaermDist2D_static(u_par, 0.5, j[0] / (-cnst.e * ne), Te)
    return u_par, deform

def SpizerHaermDist1D_weighted_v4(u, Te, sh_obj, Z, R, v_therm):
    v = u / (np.sqrt(1 + u ** 2))
    return (1.0 - np.sign(v) * sh_obj.evaluate(R, v / v_therm)) * u ** 4 * \
        Juettner1D(u, Te)

def write_SH_data(args):
    rpath = args[0]
    wpath = args[1]
    rhop_vec_j = args[2]
    j_vec = args[3]
    rhop, Te_vec = read_file(rpath + "/te_ida.res")
    rhop_vec_ne, ne_vec = read_file(rpath + "/ne_ida.res")
    Te = Te_vec * 1000
    ne = np.zeros(len(rhop))
    j = np.zeros(len(rhop))
    last_rhop = 0
    for i in range(len(rhop)):
        ne[i] = search_interpolate(rhop[i], rhop_vec_ne, ne_vec) * 1.e20
        j[i] = search_interpolate(rhop[i], rhop_vec_j, j_vec)
        if (rhop[i] >= 1.0):
            last_rhop = i
            break
    rhop = rhop[0: last_rhop]
    Te = Te[0: last_rhop]
    ne = ne[0: last_rhop]
    j = j[0: last_rhop]
    u_par_step = 1.0 / 128
    u_perp_step = 1.0 / 128
    u_par = np.arange(-1.0, 1.0, u_par_step)
    u_perp = np.arange(0.0, 1.0, u_perp_step)
    rhopfile = open(wpath + "/frhop.dat", "w")
    u_perp_file = open(wpath + "/u_perp.dat", "w")
    u_par_file = open(wpath + "/u_par.dat", "w")
    for i in range(len(u_perp)):
        u_perp_file.write("{0: 1.8f}\n".format(u_perp[i]))
    for i in range(len(u_par)):
        u_par_file.write("{0: 1.8f}\n".format(u_par[i]))
    u_par_file.flush()
    u_perp_file.flush()
    u_par_file.close()
    u_perp_file.close()
    sh_obj = sh.SpitzerHaerm(1)
    u_th = 0.0
    for i in range(len(rhop)):
        u_th = np.sqrt(1 - \
            cnst.physical_constants["electron mass"][0] ** 2 * \
            cnst.physical_constants["speed of light in vacuum"][0] ** 4 / \
            (cnst.physical_constants["electron mass"][0] * \
            cnst.physical_constants["speed of light in vacuum"][0] ** 2 + \
            cnst.physical_constants["elementary charge"][0] * Te[i]) ** 2)
        # u_t += (Juettner2D(u_par[0],u_perp[0],Te[i])*(u_par[0] + u_perp[0])**2 +\
        #         (u_par[-1] + u_perp[-1])**2*Juettner2D(u_par[-1],u_perp[-1],Te[i])) * \
        #         u_par_step*u_perp_step * 0.25
        # for k in range(1,len(u_par)-1):
        #    u_th += (Juettner2D(u_par[k],u_perp[0],Te[i])*(u_par[k] + u_perp[0])**2 +\
        #            (u_par[k] + u_perp[-1])**2*Juettner2D(u_par[k],u_perp[-1],Te[i])) * \
        #            u_par_step*u_perp_step * 0.5
        #    for l in range(1,len(u_perp)-1):
        #        u_th += (u_par[k] + u_perp[l])**2 * \
        #            Juettner2D(u_par[k],u_perp[l],Te[i])* u_par_step*u_perp_step
        v_th = u_th / (np.sqrt(1 + u_th ** 2)) * \
            cnst.physical_constants["speed of light in vacuum"][0]
        # print(v_th)
            #
        R = j[i] / (cnst.physical_constants["elementary charge"][0] * ne[i]) / \
            v_th
        if(i % 20 == 0):
            print(Te[i], v_th, R)
        Z = 1
        # break
        rhopfile.write("{0: 1.5f}\n".format(rhop[i]))
        thefile = open(wpath + "/fu{0:0>3}.dat".format(i), "w")
        for k in range(len(u_par)):
            for l in range(len(u_perp)):
                f_cur = SpizerHaermDist(u_par[k], u_perp[l], Te[i], sh_obj, Z, \
                    R, v_th)
                if(np.log(f_cur) == -np.inf):
                    thefile.write("{0: 1.8E}".format(-1.e5))
                elif(f_cur < 0):
                    thefile.write("{0: 1.8E}".format(-1.e5))
                else:
                    thefile.write("{0: 1.8E}".format(np.log(f_cur)))
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()
    print("Spitzer-Haerm data set created. \n")


def SpizerHaermDist(u_par, u_perp, Te, sh_obj, Z, R, v_therm):
    v_par = u_par / (np.sqrt(1 + u_par ** 2 + u_perp ** 2)) * cnst.c
    if(v_par >= 0.0):
        return Juettner2D(u_par, u_perp, Te) * \
            (1.0 + sh_obj.evaluate(R, v_par / v_therm))
    else:
        return Juettner2D(u_par, u_perp, Te) * \
             (1.0 - sh_obj.evaluate(R, -v_par / v_therm))


"""
def Juettner2D(u_par, u_perp, Te):    
    mu = cnst.physical_constants["electron mass"][0] * cnst.c**2 / \
        (abs(Te) * cnst.e)
    a = 1.0/(1 + 105.0/(128.0 * mu**2) + 15.0/(8.0 * mu))
    gamma = np.sqrt(1 + u_par**2 + u_perp**2)
    return a * np.sqrt(mu/(2*np.pi))**3 * \
            np.exp(mu*(1 - gamma))
            
def Juettner2D_drift(u_par, u_perp, Te, u_par_drift,u_perp_drift):    
    mu = cnst.physical_constants["electron mass"][0] * cnst.c**2 / \
        (abs(Te) * cnst.e)    
    a = 1.0 /  (np.exp(-u_par_drift**2 *mu) + np.sqrt(np.pi* u_par_drift* \
        (1 + erf(u_par_drift* np.sqrt(mu)))))
    return a * np.sqrt(mu/(2*np.pi))**3 * \
            np.exp(-mu*((u_perp-u_perp_drift)**2 + (u_par - u_par_drift)**2))            
"""
def Juettner1D_drift(u_perp, Te, u_perp_drift):
    mu = cnst.physical_constants["electron mass"][0] * cnst.c ** 2 / \
        (abs(Te) * cnst.e)
    a = 1.0 / (1 + 105.0 / (128.0 * mu ** 2) + 15.0 / (8.0 * mu))
    gamma = np.sqrt(1 + (u_perp - u_perp_drift) ** 2)
    return a * np.sqrt(mu / (2 * np.pi)) * \
            np.exp(mu * (1 - gamma))


def ratio_B(x1, x2, B1, B2):
    numerator = InterpolatedUnivariateSpline(x1, B1)
    B1_remapped = numerator(x2)
    return B1_remapped / B2
# c
def make_test_f(rpath):
    rhop_prof, Te_vec = np.loadtxt(os.path.join(rpath, "Te_file.dat"), skiprows=1, unpack=True)
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
    return distribution(None, rhop, u, pitch, Fe, None, rhop_prof, Te_vec, ne_vec)

def make_synchroton_f(rpath, B):
    rhop_prof, Te_vec = np.loadtxt(os.path.join(rpath, "Te_file.dat"), skiprows=1, unpack=True)
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
                        g0, g2, f = SynchrotonDistribution(u[j], zeta, Te_spl(rhop[i]), ne_spl(rhop[i]), B, 1.0)
                        Fe[i, j, :] *= (1.0 + f)
    return distribution(None, rhop, u, pitch, Fe, None, rhop_prof, Te_vec, ne_vec)

def apply_synchroton_to_RELAX_f(rpath, B):
    dist_obj = load_f_from_ASCII(os.path.join(rpath, "fRe"))
    Te_spl = InterpolatedUnivariateSpline(dist_obj.rhop_1D_profs, dist_obj.Te_init)
    ne_spl = InterpolatedUnivariateSpline(dist_obj.rhop_1D_profs, dist_obj.ne_init)
    zeta = np.cos(dist_obj.pitch)
    for i in range(len(dist_obj.rhop)):
        for j in range(len(dist_obj.u)):
                    if(dist_obj.u[j] > 0.2):
                        g0, g2, f = SynchrotonDistribution(dist_obj.u[j], zeta, Te_spl(dist_obj.rhop[i]), ne_spl(dist_obj.rhop[i]), B, 1.0)
                        dist_obj.f[i, j] *= (1.0 + f)
    return distribution(None, dist_obj.rhop, dist_obj.u, dist_obj.pitch, dist_obj.f, None, dist_obj.rhop_1D_profs, dist_obj.Te_init, dist_obj.ne_init)

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

def load_and_interpolate_dist(rpath, LUKE=True):
    if(LUKE):
        try:
            try:
                ne_filename = os.path.join(rpath, "..", "ne_ida.res")
                Te_filename = os.path.join(rpath, "..", "Te_ida.res")
                rhop_vec_ne, ne = read_file(ne_filename)
                rhop_vec_Te, Te = read_file(Te_filename)
                ne = ne * 1.e20
                Te = Te * 1.e3
            except IOError:
                ne_filename = os.path.join(rpath, "ne_file.dat")
                ne_data = np.loadtxt(ne_filename, skiprows=1)
                rhop_vec_ne = ne_data.T[0]
                ne = ne_data.T[1]
                Te_filename = os.path.join(rpath, "Te_file.dat")
                Te_data = np.loadtxt(Te_filename, skiprows=1)
                rhop_vec_Te = Te_data.T[0]
                Te = Te_data.T[1]
            rhop, x, y, Fe = read_LUKE_data(path)
            plt.plot(u, Fe[np.argmin(np.abs(LUKE_data['xrhoP'][0] - 0.2)), :, np.argmin(np.abs(y - 100.0 / 180.0 * np.pi))])
            # Make the grid of u and mhu equidistant, finer and limit u_max
            # Is already equidistant, but doesn't cover full range
            # dFe_du = np.array(dFe_du)
            # dFe_dpitch = np.array(dFe_dpitch)
            print("Fe", np.shape(Fe))
            print("pn", np.shape(u))
            print("mu", np.shape(mu))
            print("psi", np.shape(psi))
        except IOError as e:
            print(e)
            print("Could not find LUKE.mat at ", os.path.join(rpath, "LUKE.mat"))
            return
    else:
        try:
            ipsi, psi, u, mu, Fe = read_Fe(rpath + "/")
        except IOError:
            print("Cannot access file " + rpath + "/relax_output.dat")
            print("Generating thermal distributions instead")
            Fe, psi, u, mu = make_test_f(rpath)
            Fe, error = make_thermal_edges(Fe, LUKE_data['xrhoP'][0], rhop_vec_Te, Te, u)
            if(error != 0):
                return
        print("points u: {0:}, points pitch: {1:}\n".format(len(u), len(mu)))
        print("points psi: {0:}\n".format(len(psi)))
        print(np.sqrt(psi[ipsi]))
    # dFe_du = []
    # dFe_dpitch = []
    f_prof = []

    for i in range(len(psi)):
        f_int = RectBivariateSpline(x , y, Fe[i])
        f_prof.append(f_int)
    u_new = np.linspace(u[0], u[-1], 500)
    Fe_new = np.zeros(len(u_new))
    for i in range(len(u_new)):
        Fe_new[i] = f_prof[np.argmin(np.abs(LUKE_data['xrhoP'][0] - 0.2))].eval(u_new[i], 100.0 / 180.0 * np.pi)
#    plt.plot(u_new, Fe_new)
#    plt.show()
    return np.sqrt(psi), x, y, f_prof

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
#            plt.contourf(u, pitch, (Fe_grid * Fe[i]).T)
#            plt.show()
    return good

def read_LUKE_data(path, rhop_max=1.5, no_preprocessing=True, Flip=False):
    try:
        LUKE_f = loadmat(os.path.join(path, "LUKE.mat"))
        try:
            ne_filename = os.path.join(path, "..", "ne_ida.res")
            Te_filename = os.path.join(path, "..", "Te_ida.res")
            rhop_vec_ne, ne = read_file(ne_filename)
            rhop_vec_Te, Te = read_file(Te_filename)
            ne = ne * 1.e20
            Te = Te * 1.e3
        except IOError:
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
        mu = y
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
        dist_obj = distribution(None, rhop, x, y, Fe, None, rhop_vec_ne, Te, ne)
        return dist_obj
    except IOError as e:
        print(e)
        print("Could not find LUKE.mat for distribution at ", os.path.join(path, "LUKE.mat"))
        return []

def read_LUKE_profiles(path):
    try:
        LUKE_mat = loadmat(os.path.join(path, "LUKE.mat"), struct_as_record=False, squeeze_me=True)
        scalar = LUKE_mat["data_proc"].scalar
        radial = LUKE_mat["data_proc"].radial
        waves = LUKE_mat["data_proc"].wave
        quasi_linear_beam = beam(radial.xrhoT, radial.xrhoP, radial.P_tot * 1.e6, radial.J_tot * 1.e6, \
                                 scalar.p_rf_2piRp * 1.e6, scalar.I_tot * 1.e6)
        linear_beam = beam(radial.xrhoT, radial.xrhoP, (waves.wxP_rf_lin[0] + waves.wxP_rf_lin[1]) * 1.e6, np.zeros(len(radial.xrhoP)), \
                                 scalar.p_rf_2piRp_lin * 1.e6, 0.0, \
                                 PW_beam=[waves.wxP_rf[0], waves.wxP_rf[1]], j_beam=[np.zeros(len(radial.xrhoP)), np.zeros(len(radial.xrhoP))])
        return quasi_linear_beam, linear_beam
    except IOError as e:
        print(e)
        print("Could not find LUKE.mat for beams at ", os.path.join(path, "LUKE_DATA.mat"))
        return [], []

def dp_dt_rad(u_perp, u_par, B):
    u = np.sqrt(u_perp ** 2 + u_par ** 2)
    tau_r = 6.0 * np.pi * cnst.epsilon_0 * (cnst.m_e * cnst.c) ** 3 / (cnst.e ** 4 * B ** 2)
    print("tau_r", tau_r)
    return -np.sqrt(1.0 + 1.0 / u ** 2) / tau_r * u_perp ** 2

def collision_time(v, ne, Z_eff=1.5, ln_lambda=21):
    return 1.0 / (4.0 * np.pi * ne * Z_eff * cnst.e ** 4 * ln_lambda / ((4.0 * np.pi * cnst.epsilon_0) ** 2 * cnst.m_e ** 2 * v ** 3))

def read_waves_mat_to_beam(waves_mat, EQSlice):
    rho_prof = waves_mat["rhop_prof"]
    j = waves_mat["j_prof"]
    PW = waves_mat["PW_prof"]
    PW_tot = waves_mat["PW_tot"]
    j_tot = waves_mat["j_tot"]
    rays = []
    B_tot = np.sqrt(EQSlice.Br ** 2 + EQSlice.Bt ** 2 + EQSlice.Bz ** 2)
    B_tot_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, B_tot)
    PW_beam = []
    j_beam = []
    for key in ["s", "R", "phi", "z", "rhop", "PW", "Npar"]:
        waves_mat[key] = np.atleast_3d(waves_mat[key])
        if(waves_mat[key].shape[-1] == 1):
            waves_mat[key] = np.swapaxes(waves_mat[key].T, 1, 2)
    for key in ["PW_beam", "j_beam"]:
        waves_mat[key] = np.atleast_2d(waves_mat[key])
        if(waves_mat[key].shape[-1] == 1):
            waves_mat[key] = waves_mat[key].T
    for ibeam in range(len(waves_mat["R"])):
        print("Processing beam: " + str(ibeam + 1))
        PW_beam.append(waves_mat["PW_beam"][ibeam])
        j_beam.append(waves_mat["j_beam"][ibeam])
        rays.append([])
        for iray in range(len(waves_mat["R"][ibeam])):
            print("Processing ray: " + str(iray + 1))
            rays[-1].append({})
            rays[-1][-1]["s"] = waves_mat["s"][ibeam][iray]
            rays[-1][-1]["R"] = waves_mat["R"][ibeam][iray]
            rays[-1][-1]["phi"] = waves_mat["phi"][ibeam][iray]
            rays[-1][-1]["z"] = waves_mat["z"][ibeam][iray]
            rays[-1][-1]["rhop"] = waves_mat["rhop"][ibeam][iray]
            rays[-1][-1]["PW"] = waves_mat["PW"][ibeam][iray]
            rays[-1][-1]["Npar"] = waves_mat["Npar"][ibeam][iray]
            rays[-1][-1]["omega_c"] = cnst.e * B_tot_spl(rays[-1][-1]["R"], rays[-1][-1]["z"], grid=False) / cnst.m_e
    PW_beam = np.array(PW_beam)
    j_beam = np.array(j_beam)
    return beam(waves_mat["rhot_prof"], rho_prof, PW, j, PW_tot, j_tot, PW_beam, j_beam, rays)

def read_dist_mat_to_beam(dist_mat):
    rho_prof = dist_mat["rhop_prof"]
    j = dist_mat["j_prof"]
    PW = dist_mat["PW_prof"]
    PW_tot = dist_mat["PW_tot"]
    j_tot = dist_mat["j_tot"]
    return beam(dist_mat["rhot_prof"], rho_prof, PW, j, PW_tot, j_tot, None, None, None)

def export_fortran_friendly(args):
    # print(uxx, ull)
    dist_obj = args[0]
    wpath = args[1]
    # shot = args[2]
    # time = args[3]
    # print y
    # LUKE = False
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
#        f_cur = remap_f(x, y, Fe[i], uxx, ull, f_cur)
#        test_integral(Fe[i], x[1] - x[0], y[1] - y[0], 64, 128)
#        test_integral(f_cur, ull[1] - ull[0], uxx[1] - uxx[0], 64, 128)
        rhopfile.write("{0: 1.12e}\n".format(dist_obj.rhop[i]))
        thefile = open(wpath + "/fu{0:0>3}.dat".format(i), "w")
#        thedufile = open(wpath + "/df_du{0:0>3}.dat".format(i), "w")
#        thedpitchfile = open(wpath + "/df_dpitch{0:0>3}.dat".format(i), "w")
        for j in range(len(dist_obj.u)):
            for k in range(len(dist_obj.pitch)):
                thefile.write("{0: 1.8E} ".format(np.log(dist_obj.f[i, j, k])))
#                thedufile.write("{0: 1.8E} ".format(dFe_du[i, j, k]))
#                thedpitchfile.write("{0: 1.8E} ".format(dFe_dpitch[i, j, k]))
            thefile.write("\n")
#            thedufile.write("\n")
#            thedpitchfile.write("\n")
        thefile.flush()
        thefile.close()
#        thedufile.flush()
#        thedufile.close()
#        thedpitchfile.flush()
#        thedpitchfile.close()
    rhopfile.flush()
    rhopfile.close()
    print("Distribution ready")

# export_fortran_friendly([os.path.join("/ptmp1/work/sdenk/nssf/", "33698", "5.00", "OERT", "ECRad_data"),
#                         os.path.join("/ptmp1/work/sdenk/nssf/", "33698", "5.00", "OERT", "ECRad_data", "fLu"), True, 1.5])

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

def make_EField(R, z, Psi, R_ax, z_ax, Psi_target, V_loop):
    R_aus, z_aus = get_R_aus(R, z, Psi, R_ax, z_ax, Psi_target)
    E_field = V_loop / (2.0 * np.pi * R_aus)
    return R_aus, z_aus, E_field

def make_RDiff_old(Psi, scale, psi_prof, Te):
    Te_spl = InterpolatedUnivariateSpline(psi_prof, Te)
    psi_out = np.concatenate([[0.0], Psi, [1.0]])
    Rdiff = scale * (1.e0 - np.abs(Te_spl(psi_out, 1) / np.max(np.abs(Te_spl(psi_out, 1))))) * (1.0 - np.exp(-(1 - np.sqrt(psi_out)) ** 4 * 1000))
    Rdiff[0] = Rdiff[1]
    return psi_out, Rdiff

def make_RDiff(Psi, scale, rho_width):
    psi_out = np.concatenate([[0.0], Psi, [1.0]])
    Rdiff = scale * np.exp(-psi_out / rho_width ** 2)
    Rdiff[0] = Rdiff[1]
    return psi_out, Rdiff

class f_interpolator:
    def __init__(self, working_dir, dist="Re", order=3):
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
        if(dist == "Lu" or dist == "Re"):
            f_folder = os.path.join(working_dir, "ECRad_data", "f" + dist)
            x = np.loadtxt(os.path.join(f_folder, "u.dat"), skiprows=1)
            y = np.loadtxt(os.path.join(f_folder, "pitch.dat"), skiprows=1)
            self.rhop = np.loadtxt(os.path.join(f_folder, "frhop.dat"), skiprows=1)
            Fe = []
            for irhop in range(len(self.rhop)):
                Fe.append(np.loadtxt(os.path.join(f_folder, "fu{0:03d}.dat".format(irhop))))
            Fe = np.array(Fe)
            rhop_B_min, B_min = get_B_min_from_file(os.path.join(working_dir, "ECRad_data"))
            self.B_min_spline = InterpolatedUnivariateSpline(rhop_B_min, B_min)
        elif(dist == "Ge"):
            f_folder = os.path.join(working_dir, "ECRad_data", "f" + dist)
            x = np.loadtxt(os.path.join(f_folder, "vpar.dat"), skiprows=1)
            y = np.loadtxt(os.path.join(f_folder, "mu.dat"), skiprows=1)
            self.rhop = np.loadtxt(os.path.join(f_folder, "grhop.dat"), skiprows=1)
            self.B0 = np.loadtxt(os.path.join(f_folder, "B0.dat"))
            Fe = []
            for irhop in range(len(self.rhop)):
                Fe.append(np.loadtxt(os.path.join(f_folder, "gvpar{0:03d}.dat".format(irhop))))
            Fe = np.array(Fe)
        elif(dist == "Ge0"):
            f_folder = os.path.join(working_dir, "ECRad_data", "fGe")
            x = np.loadtxt(os.path.join(f_folder, "vpar.dat"), skiprows=1)
            y = np.loadtxt(os.path.join(f_folder, "mu.dat"), skiprows=1)
            self.rhop = np.loadtxt(os.path.join(f_folder, "grhop.dat"), skiprows=1)
            Fe = np.loadtxt(os.path.join(f_folder, "f0.dat"))
            self.B0 = np.loadtxt(os.path.join(f_folder, "B0.dat"))
            self.static_dist = True
        else:
            print("Invalid distribution flag", dist)
            raise ValueError
        # dFe_du = []
        # dFe_dpitch = []
        self.x = x
        self.y = y
        self.Fe = Fe
        if(self.static_dist):
            self.spline_mat = None
        else:
            # Spline to interpolate rho
            self.spline_mat = []
            for i in range(len(x)):
                self.spline_mat.append([])
                for j in range(len(y)):
                    self.spline_mat[-1].append(InterpolatedUnivariateSpline(self.rhop, self.Fe.T[j][i], k=1))

    def get_spline(self, rhop, Te):
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

def test_integral(f, dx, dy, x0, y0, lenx, leny, r=False):
    int = 0
    x = x0
    y = y0
    # print len(f[:,0]), len(f[0]), lenx, leny
    if(r):
        int += 0.25 * f[0, 0] * dy * dx * x
    else:
        int += 0.25 * f[0, 0] * dy * dx
    y += dy
    for j in range(1, leny):
        if(r):
            int += 0.5 * f[0, j] * dy * dx * x
        else:
            int += 0.5 * f[0, j] * dy * dx
        y += dy
    if(r):
        int += 0.25 * f[0, leny - 1] * dy * dx * x
    else:
        int += 0.25 * f[0, leny - 1] * dy * dx
    x += dx
    for i in range(1, lenx):
        y = 0.0
        if(r):
            int += 0.5 * f[i, j] * dy * dx * x
        else:
            int += 0.5 * f[i, j] * dy * dx
        y += dy
        for j in range(1, leny):
            if(r):
                int += f[i, j] * dy * dx * x
            else:
                int += f[i, j] * dy * dx
            y += dy
        if(r):
            int += 0.5 * f[i, j] * dy * dx * x
        else:
            int += 0.5 * f[i, j] * dy * dx
        x += dx
    y = 0.0
    if(r):
        int += 0.25 * f[lenx - 1, 0] * dy * dx * x
    else:
        int += 0.25 * f[lenx - 1, 0] * dy * dx
    y += dy
    for j in range(1, leny):
        if(r):
            int += 0.25 * f[lenx - 1, j] * dy * dx * x
        else:
            int += 0.5 * f[lenx - 1, j] * dy * dx
        y += dy
    if(r):
        int += 0.25 * f[lenx - 1, leny - 1] * dy * dx * x
    else:
        int += 0.25 * f[lenx - 1, leny - 1] * dy * dx
    print int
"""
def remap_f(x,y, Fe_rhop, uxx, ull, Fe_remapped):
    spline = RectBivariateSpline(x,y,Fe_rhop, kx = 3, ky =3)
    for i in range(len(ull)):
        for j in range(len(uxx)):
            cur_x = np.sqrt(uxx[j]**2 + ull[i]**2)
            cur_y = np.arctan2(uxx[j],ull[i])
            #print cur_x, cur_y
            Fe_remapped[i,j] = spline.ev(cur_x, cur_y)
            #print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped
 """
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
        # print cur_x, cur_y
        Fe_remapped[i] = spline.ev(cur_x, cur_y)
        # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped

def remap_f1D_uxx(x, y, Fe_rhop, uxx, ull, Fe_remapped):
    spline = RectBivariateSpline(x, y, Fe_rhop, kx=5, ky=5)
    for i in range(len(uxx)):
        cur_x = np.sqrt(uxx[i] ** 2 + ull ** 2)
        cur_y = np.arctan2(uxx[i], ull)
        # print cur_x, cur_y
        Fe_remapped[i] = spline.ev(cur_x, cur_y)
        # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped

def remap_f_res(x, y, Fe_rhop, uxx, ull, Fe_remapped):
    spline = RectBivariateSpline(x, y, Fe_rhop, kx=5, ky=5)
    for i in range(len(ull)):
        cur_x = np.sqrt(uxx[i] ** 2 + ull[i] ** 2)
        cur_y = np.arctan2(uxx[i], ull[i])
            # print cur_x, cur_y
        Fe_remapped[i] = spline.ev(cur_x, cur_y)
            # print(len(Fe_remapped[:,0]), len(Fe_remapped[0,:]))
    return Fe_remapped

# MAIN
# if __name__=='__main__':
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

def create_test_data():
    ull = np.arange(-1., 1., 1.0 / 128.0)
    uxx = np.arange(0.0, 1., 1.0 / 128.0)
    temp_file = open(root + "F90/ECRad_Model/Te_rhop_relax.dat")
    temps = temp_file.readlines()
    temp_file.close()
    Te = []
    rhop = []
    rhopfile = open(\
            root + "F90/ECRad_Model/ECRad_data/frhop.dat", "w")
    uxxfile = open(\
            root + "F90/ECRad_Model/ECRad_data/u_perp.dat", "w")
    ullfile = open(\
            root + "F90/ECRad_Model/ECRad_data/u_par.dat", "w")
    for i in range(len(uxx)):
        uxxfile.write("{0: 1.8E}\n".format(uxx[i]))
    for i in range(len(ull)):
        ullfile.write("{0: 1.8E}\n".format(ull[i]))
    ullfile.flush()
    ullfile.close()
    uxxfile.flush()
    for i in range(len(temps)):
        rhop.append(float(temps[i].split("   ")[0]))
        Te.append(float(temps[i].split("   ")[1]))
    # print Te
    # print rhop
    for i in range(len(rhop)):
        rhopfile.write("{0: 1.5f}\n".format(rhop[i]))
        thefile = open(\
            root + "F90/ECRad_Model/ECRad_data/fu{0:0>2}.dat".format(i), "w")
        for j in range(len(ull)):
            for k in range(len(uxx)):
                f = make_f(Te[i], uxx[k], ull[j])
                thefile.write("{0: 1.8E}".format(np.log(f)))
            thefile.write("\n")
        thefile.flush()
        thefile.close()
    rhopfile.flush()
    rhopfile.close()



def read():
    # test_data()
    # export_fortran_friendly()
    # plt.plot(range(64),y)
    # plt.show()
    #
    # print(len(x),len(y), len(psi))
    ipsi, psi, x, y, Fe = read_Fe(base + "ECRad_data/")
    ull = np.arange(-1., 1., 0.1 / 128.0)
    uxx = np.arange(0.0, 1., 0.1 / 128.0)
    # ull, uxx = momenta_on_equatorial_plane(x, y)
    f = np.zeros([len(ull), len(uxx)])
    f = remap_f(x, y, Fe[ipsi], uxx, ull, f)
    # print(len(ull[0]), len(ull[:,0]), len(uxx[0]), len(uxx[:,0]))
    # plt.contour(range(64), range(128),ull,100)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.contour(ull, uxx, f.T, 50)
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0.0, 0.25)

    ax.set_xlabel(r'$p_\parallel / (m_\mathrm{e} c)$')
    ax.set_ylabel(r'$p_\perp / (m_\mathrm{e} c)$')
    plt.suptitle(r'AUG27764 @ $\sqrt{\psi} =$ ' +
          str(round(np.sqrt(psi[ipsi]), 3)))
    plt.show()


class distribution:
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
        self.u_th = np.zeros(len(rhop))
        self.f_fast_fraction = np.zeros(len(rhop))
        for i in range(len(self.rhop)):
            self.u_th[i], self.f_fast_fraction[i] = get_u_th_and_fast_fraction(self.u, self.pitch, f[i])
        print("Maximum thermal momentum {0:1.2e}".format(np.max(self.u_th)))
        print("Fast electron fraction for this flux surface {0:1.2e}".format(self.f_fast_fraction[np.argmax(self.u_th)]))
        for i in range(len(self.rhop)):
            # Remap for LFS
            remap_f_Maj(self.u, self.pitch, self.f_log, i, self.ull, self.uxx, self.f_cycl_log[i], 1, 1, LUKE=True)
            self.ne[i], self.Te[i] = get_0th_and_2nd_moment(self.ull, self.uxx, np.exp(self.f_cycl_log[i]))
            # print(self.Te[i], self.ne[i])
            print("Finished distribution profile slice {0:d}/{1:d}".format(i + 1, len(self.rhop)))
        self.ne = self.ne * ne_spl(self.rhop)
        self.f_cycl = np.exp(self.f_cycl_log)
        self.f_cycl_log10 = np.log10(self.f_cycl)
        print("distribution shape:", self.f.shape)
        print("Finished remapping.")

class beam:
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

def make_dist_from_Gene_input(path, shot, time, EQObj, debug=False):
    if(not h5py_ready):
        print("h5py not loaded - cannot load h5py GENE data")
        return
    h5_fileID = h5py.File(os.path.join(path, "xvspelectrons_1c.h5"), 'r')
    ne_file = np.loadtxt(os.path.join(path, "ne_file.dat"), skiprows=1)
    ne_spl = InterpolatedUnivariateSpline(ne_file.T[0], ne_file.T[1], k=1)
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
    EQSlice = EQObj.read_EQ_from_shotfile(time)
    rhop_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, EQSlice.rhop)
    rhop = rhop_spl(R, z, grid=False)
    # f /= cnst.c ** 3 * cnst.m_e / (2.0 * np.pi)
    # B0 = EQObj.get_B_on_axis(time)
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
    return rhop[rhopindex], R, z, beta_par, mu_norm, f[rhopindex], f0, g[rhopindex], Te, ne, B_local

class Gene:
    def __init__(self, path, shot, time=None, EQSlice=None, it=0, EQObj=None):
        if(not h5py_ready):
            print("h5py not loaded - cannot load h5py GENE data")
            return
        h5_fileID = h5py.File(os.path.join(path, "xvsp_electrons_1e.h5"), 'r')['xvsp_electrons']
        gene_pos = h5py.File("/ptmp1/work/sdenk/ECRad7/xvspelectrons_1c.h5")
        if(EQSlice is None):
            if(time is None):
                print("The Gene class has to be initialized with either time or an EQSlice object present")
            if(EQData is None):
                print("Either EQSlice or EQData must be present when GENE class is initialized")
            EQSlice = EQObj.read_EQ_from_shotfile(time)
        beta_max = 0.5
        self.R = np.array(gene_pos["axes"]["Rpos_m"]).flatten()
        self.z = np.array(gene_pos["axes"]["Zpos_m"]).flatten()
        dR_down = self.R[1] - self.R[0]
        dR_up = self.R[-1] - self.R[-2]
        dz_down = self.z[1] - self.z[0]
        dz_up = self.z[-1] - self.z[-2]
        self.R = np.concatenate([[self.R[0] - dR_down], self.R, [self.R[-1] + dR_up]])
        self.z = np.concatenate([[self.z[0] - dz_down], self.z, [self.z[-1] + dz_up]])
        self.beta_par = np.array(h5_fileID["axes"]['vpar_m_s']).flatten() / cnst.c
        self.mu_norm = np.array(h5_fileID["axes"]['mu_Am2']).flatten() / (cnst.m_e * cnst.c ** 2)
        self.total_time_cnt = len(h5_fileID['delta_f'].keys())
        self.g = np.array(h5_fileID['delta_f'][u'{0:010d}'.format(it)]).T * cnst.c ** 3
        self.f0 = np.array(h5_fileID['misc']['F0']).T * cnst.c ** 3
        self.Te = h5_fileID['general information'].attrs["Tref,eV"]
        self.ne = h5_fileID['general information'].attrs["nref,m^-3"]
        rhop_spl = RectBivariateSpline(EQSlice.R, EQSlice.z, EQSlice.rhop)
        self.rhop = rhop_spl(self.R, self.z, grid=False)
        self.B0 = h5_fileID["misc"].attrs['Blocal,T']
        self.beta_perp = np.sqrt(self.mu_norm * 2.0 * self.B0)
        self.f0 = self.f0[:, self.beta_perp < beta_max]
        self.f0 = self.f0[np.abs(self.beta_par) < beta_max, :]
        self.g = self.g[:, :, self.beta_perp < beta_max]
        self.g = self.g[:, np.abs(self.beta_par) < beta_max, :]
        self.mu_norm = self.mu_norm[self.beta_perp < beta_max]
        self.beta_par = self.beta_par[np.abs(self.beta_par) < beta_max]
        self.f = np.concatenate([[self.f0], self.g + self.f0, [self.f0]])
        rhopindex = np.argsort(self.rhop)
        self.rhop = self.rhop[rhopindex]

class Gene_BiMax(Gene):
    def __init__(self, path, shot, time=None, EQSlice=None, it=0, eq_exp='AUGD', eq_diag='EQH', eq_ed=0):
        Gene.__init__(self, path, shot, time, EQSlice, it, eq_exp, eq_diag, eq_ed)

    def make_bi_max(self):
        self.Te_perp, self.Te_par = get_dist_moments_non_rel(self.rhop, self.beta_par, self.mu_norm, \
                                                             self.f, self.Te, self.ne, self.B0, \
                                                             slices=1, ne_out=False)

def browse_gene_dists(path, shot, time, it):
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
#    plt.contour(vpar, vperp, g.T)
#    plt.show()

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


def get_u_th_and_fast_fraction(u, pitch, f):
    f_1st_moment = np.zeros(len(u))
    f_0th_moment = np.zeros(len(u))
    for i in range(len(u)):
        f_0th_moment[i] = simps(u[i] ** 2 * f[i, :] * np.pi * 2.e0 * np.sin(pitch), pitch)
        f_1st_moment[i] = simps(u[i] ** 2 * f[i, :] * np.pi * 2.e0 * u[i] * np.sin(pitch), pitch)
    zeros_mom = simps(f_0th_moment, u)
    first_mom = simps(f_1st_moment, u)
    u_supra = np.sqrt(3.e0)  # 2.0 * first_mom / zeros_mom
    f_0th_moment_low_energy = np.zeros(len(u[u > u_supra]))
    if(np.all(u < u_supra)):
        print('Cannot resolve fast fraction because u_max ({0:1.2f}) > u_supra({1:1.2f})'.format(np.max(u), u_supra))
        return first_mom / zeros_mom, 0.e0
    for i in range(len(u[u > u_supra])):
        f_0th_moment_low_energy[i] = simps(u[u > u_supra][i] ** 2 * f[u > u_supra][i, :] * np.pi * 2.e0 * np.sin(pitch), pitch)
    fast_e_fraction = simps(f_0th_moment_low_energy, u[u > u_supra])
    return first_mom / zeros_mom, fast_e_fraction

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
    if(not h5py_ready):
        print("h5py not loaded - cannot load h5py GENE data")
        return
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
    if(not h5py_ready):
        print("h5py not loaded - cannot load h5py GENE data")
        return
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
            Te_perp[i], Te_par[i], ne_prof[i] = get_bimaxwellian_moments_non_rel(np.min(beta_par), np.max(beta_perp), \
                                                         np.min(beta_perp), np.max(beta_perp), f_spl, ne_out=True)
        else:
            Te_perp[i], Te_par[i] = get_bimaxwellian_moments_non_rel(np.min(beta_par), np.max(beta_perp), \
                                                         np.min(beta_perp), np.max(beta_perp), f_spl)
        print("Te, Te_perp, Te_par, 1 - Te_perp/Te, 1 - Te_par/Te", Te, \
              Te_perp[i], Te_par[i], (1.0 - Te_perp[i] / Te), (1.0 - Te_par[i] / Te))
    if(ne_out):
        return Te_perp, Te_par, ne_prof
    else:
        return Te_perp, Te_par

def make_bimax_from_GENE(path, shot, time, wpath_parent, subdir_list, wrong=False, write=False):
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

def plot_dist_moments(path, shot, time, eq_exp='AUGD', eq_diag='EQH', eq_ed=0):
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
    leg = ax.legend(lns, labs)
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

def export_gene_fortran_friendly(wpath, rhop, beta_par, mu_norm, ne, f, f0, B0):
    f = f / ne
    f0 = f0 / ne
    f[f < 1.e-20] = 1.e-20
    f0[f0 < 1.e-20] = 1.e-20
    rhopfile = open(wpath + "/grhop.dat", "w")
    mufile = open(wpath + "/mu.dat", "w")
    vparfile = open(wpath + "/vpar.dat", "w")
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

if(__name__ == "__main__"):
    pass
#    rhop, R, z, beta_par, mu_norm, f, f0, g, Te, ne, B0 = make_dist_from_Gene_input("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/ECRad_data", 33585, 3.0)
#    Te_perp, Te_par, ne_prof = get_dist_moments_non_rel(rhop, beta_par, mu_norm, f, Te, ne, B0, slices=1, ne_out=True)
#    plt.plot(rhop, ne_prof)
#    BiMax = Gene_BiMax("/ptmp1/work/sdenk/ECRad7/", 33585, time=3.0)
#    plt.plot(BiMax.rhop, BiMax.Te_par)
#    plt.plot(BiMax.rhop, BiMax.Te_perp)
#    plt.show()
#    browse_gene_dists("/ptmp1/work/sdenk/ECRad5/", 33585, time=3.0, it=5)
#    relax_time(50.e3, 8.e3, 3.e19)
#    print(dp_dt_rad(1.0, 0.0, 2.5))
#    print(collision_time(1.0 / np.sqrt(2.0) * cnst.c, 4.5e19))
    # Dreicer_field_and_E_crit(1.5e0, 3.e0, 0.4)
#    t = collision_time(rel_thermal_beta(511.0 / 10.0) * cnst.c, 2.e19)
#    t_wave = 1.0 / 140.e9
#    print(t, t_wave, t / t_wave)
#    print(get_thermal_av_cyc_freq(8.e3, 131.6))
#    dir = "/ptmp1/work/sdenk/nssf/32082/4.25/OERT/ed_1/ECRad_data/"
# #    dir = "/ptmp1/work/sdenk/nssf/33705/4.90/OERT/ed_17/ECRad_data/"
#    # "/ptmp1/work/sdenk/nssf/31539/2.81/OERT/ed_12/ECRad_data/"
#    dist_obj = make_synchroton_f(dir, 2.5)  # make_test_f
# #    dist_obj = apply_synchroton_to_RELAX_f(dir, 2.35)
#    if not os.path.isdir(os.path.join(dir, "fRe")):
#        os.mkdir(os.path.join(dir, "fRe"))
#    export_fortran_friendly([dist_obj, os.path.join(dir, "fRe")])
    u = np.linspace(0.2, 3.0, 100)
#    g0a, g2a, f = SynchrotonDistribution_approx(u, 0.0, 9.e3, 5.e19, 2.5)
    g0b, g2b, f1 = SynchrotonDistribution(u, 0.0, 9.e3, 5.e19, 2.5, 1.5)
    g0c, g2c, f2 = SynchrotonDistribution(u, 0.0, 25.e3, 9.e19, 5.6, 1.5)
#    plt.plot(u, (1.0 + f), "-", label="approx, $u=u_\perp, Z=1$")
    plt.plot(u, (1.0 + f1), "--", label="exact, $u=u_\perp, Z=1.0$")
    plt.plot(u, (1.0 + f2), ":", label="exact, $u=u_\perp, Z=1.5$")
#    plt.plot(u, (1.0 + f), "-", label="2.5 T, $u=u_\perp$")
#    g0, g2, f = SynchrotonDistribution_approx(u, 0.0, 8.e3, 5.e19, 2.0)
#    plt.plot(u, (1.0 + f), "--", label="2.0 T, $u=u_\perp$")
#    g0, g2, f = SynchrotonDistribution_approx(u, 0.0, 8.e3, 5.e19, 1.8)
#    plt.plot(u, (1.0 + f), ":", label="1.8 T, $u=u_\perp$")
    plt.gca().set_xlabel(r"$u$")
    plt.gca().set_ylabel(r"$f/f_0$")
    plt.legend()
    plt.show()
#    plt.figure()
#    plt.plot(u, g0a, "-")
#    plt.plot(u, -g2a, "--")
#    plt.plot(u, g2b, "--")
#    plt.plot(u, g2c, ":")
#    plt.show()
#    make_dist_from_Gene_input("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ECRad_data", 33585, 3.0, debug=True)
#    make_bimax_from_GENE("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/ECRad_data", 33585, 3.0, \
#                         "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                         ["ed_5", "ed_6", "ed_7", "ed_8", "ed_9", "ed_10"], True, False)
#    make_bimaxjuett_from_GENE("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ed_4/ECRad_data", 33585, 3.0, \
#                         "/ptmp1/work/sdenk/nssf/33585/3.00/OERT/", \
#                         ["ed_11", "ed_12", "ed_13", "ed_14", "ed_15", "ed_16"])
# #    plot_dist_moments("/ptmp1/work/sdenk/nssf/33585/3.00/OERT/ECRad_data", 33585, 3.0)
# test_make_EField()
# make_iso_flux("/ptmp1/work/sdenk/nssf/30907/0.73/",30907, 0.73)
# make_f_grid('/ptmp1/work/sdenk/nssf/30907/1.45/ECRad_data',30907, 1.45,16, "Th")
# test_fit()
