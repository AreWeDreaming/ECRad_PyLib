'''
Created on Oct 14, 2015

@author: sdenk
'''
import sys
import ctypes as ct
import numpy as np
sys.path.append('/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib')
import dd
import os
from scipy.interpolate import InterpolatedUnivariateSpline

if ct.sizeof(ct.c_long) == 8:
    libso = 'lib64'
else:
    libso = 'lib'
libkk = ct.cdll.LoadLibrary('/afs/ipp/aug/ads/' + libso + '/@sys/libkk.so')
tb_path = ""
output_path = "."
profpnts = 150
btf_corr_fact = 1.005  # Empirical correction factor for Bt given by MBI (shot >= 30160)
btf_field_cor_old = 1.0 / 0.99  # Empirical correction factor for Bt given by EQH (shot < 30160)

def make_topfile(shot, time):
    # Note this routine uses MBI-BTFABB for a correction of the toroidal magentic field
    # Furthermore, an empirical BTF correction factor of 1.005 is applied
    # This creates a topfile that is consistent with the magnetic field used in OERT
    print("Creating topfile for #{0:n} t = {1:1.2f}".format(shot, time))
    try:
        EQH = EQU(shot)
    except:
        print("Error loading EQH/EQI shotfile for discharge #{0:n}".format(shot))
        print("Cannot create topfile")
        return -1
    EQH.Load(shot, Diagnostic='EQH')
    R = EQH.getR(time)
    z = EQH.getz(time)
    Psi = EQH.getPsi(time)
    KKobj = KK()
    ed = EQH.ed
    print(ed)
    special_points = KKobj.kkeqpfx(shot, time, diag="EQH", ed=ed)
    columns = 8
    columns -= 1  # just for counting
    R0 = 1.65  # Point for which BTFABB is defined
    print("Magnetic axis position: ", "{0:1.3f}".format(special_points.Raxis))
    topfile = open(os.path.join(tb_path, "topfile"), "w")
    topfile.write('Number of radial and vertical grid points in AUGD:EQH:{0:5n}: {1:1.4f}\n'.format(shot, time))
    topfile.write('   {0: 8n} {1: 8n}\n'.format(len(R), len(z)))
    topfile.write('Inside and Outside radius and psi_sep\n')
    topfile.write('   {0: 1.7E}  {1: 1.7E}  {2: 1.7E}'.format(R[0], R[-1], \
        special_points.psispx))
    topfile.write('\n Radial grid coordinates\n')
    cnt = 0
    for i in range(len(R)):
        topfile.write("  {0: 1.7E}".format(R[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    topfile.write("\n")
    topfile.write('Vertical grid coordinates\n')
    cnt = 0
    for i in range(len(z)):
        topfile.write("  {0: 1.7E}".format(z[i]))
        if(cnt == columns):
            topfile.write("\n")
            cnt = 0
        else:
            cnt += 1
    topfile.write("\n")
    B_r = np.zeros((len(R), len(z)))
    B_t = np.zeros((len(R), len(z)))
    B_z = np.zeros((len(R), len(z)))
    # Adapted from mod_eqi.f90 by R. Fischer
    # adjust only 1/R part of toroidal field with btf_corr_factor
    # the ripple is not included, because this is just a 2D equilibrium
    rv = 2.40
    vz = 0.e0
    magnetic_field_outside = KKobj.kkrzBrzt(shot, time, [rv], [vz], diag="EQH", ed=ed)
    Btf0_eq = magnetic_field_outside.bt
    Btf0_eq = Btf0_eq * rv / R0
    R_temp = np.zeros(len(z))
    for i in range(len(R)):
        R_temp[:] = R[i]
        magn_field = KKobj.kkrzBrzt(shot, time, R_temp, z, diag="EQH", ed=ed)
        B_r[i] = magn_field.br
        B_t[i] = magn_field.bt
        B_z[i] = magn_field.bz
    if(shot >= 30160):
        try:
            print("Opening MBI")
            MBI = dd.shotfile('MBI', int(shot))
            signal = MBI.getSignal("BTFABB", \
                          tBegin=time - 1.e-5, tEnd=time + 1.e-5)
        except:
            print("Could not find MBI data")
            return -1

        # ivR = np.argmin(np.abs(pfm_dict["Ri"] - rv))
        # jvz = np.argmin(np.abs(pfm_dict["zj"] - vz))
        Btf0 = np.mean(signal)
        btf_field_cor = 1.005
        Btok = Btf0 * R0 / R
        for i in range(len(z)):
            Btok_eq = Btf0_eq * R0 / R  # vacuum toroidal field from EQH
            Bdia = B_t.T[i] - Btok_eq  # subtract vacuum toroidal field from equilibrium to obtain diamagnetic field
            B_t.T[i] = (Btok * btf_field_cor) + Bdia  # add corrected vacuum toroidal field to be used
    else:
        Btok_eq = Btf0_eq * R0 / R  # vacuum toroidal field from EQH
        Bdia = B_t.T[i] - Btok_eq  # subtract vacuum toroidal field from equilibrium to obtain diamagnetic field
        B_t.T[i] = (Btok_eq * btf_field_cor_old) + Bdia  # add corrected vacuum toroidal field to be used
    B_r = B_r.T  # in topfile z comes first regardless of the arrangement
    B_t = B_t.T  # in topfile z comes first regardless of the arrangement
    B_z = B_z.T  # in topfile z comes first regardless of the arrangement
    Psi = Psi.T  # in topfile z comes first regardless of the arrangement
    topfile.write('B_r on grid\n')
    cnt = 0
    for i in range(len(B_r)):
        for j in range(len(B_r[i])):
            topfile.write("  {0: 1.7E}".format(B_r[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.write("\n")
    topfile.write('B_t on grid\n')
    cnt = 0
    for i in range(len(B_t)):
        for j in range(len(B_t[i])):
            topfile.write("  {0: 1.7E}".format(B_t[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.write("\n")
    cnt = 0
    topfile.write('B_z on grid\n')
    for i in range(len(B_z)):
        for j in range(len(B_z[i])):
            topfile.write("  {0: 1.7E}".format(B_z[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
    topfile.write("\n")
    cnt = 0
    topfile.write('Normalised psi on grid\n')
    for i in range(len(Psi)):
        for j in range(len(Psi[i])):
            topfile.write("  {0: 1.7E}".format(Psi[i][j]))
            if(cnt == columns):
                topfile.write("\n")
                cnt = 0
            else:
                cnt += 1
        # topfile.write("\n")
    topfile.flush()
    topfile.close()
    print("topfile successfully written to " + os.path.join(output_path, "topfile"))
    print("Writting Te and ne profiles")
    try:
        IDA = dd.shotfile('IDA', int(shot))
    except:
        print("Error loading IDA shotfile for discharge #{0:n}".format(shot))
        print("No temperature or density file created")
        return -1
    i_time = IDA.getTimeBaseIndices('time', time, time)[0]
    ne_vec = np.array(IDA.getSignalGroup('ne'))[i_time]
    Te_vec = np.array(IDA.getSignalGroup('Te'))[i_time]
    rhop_vec = np.array(IDA.getAreaBase('rhop').data)[i_time]
    IDA.close()
    Te_spline = InterpolatedUnivariateSpline(rhop_vec, np.log(Te_vec))
    ne_spline = InterpolatedUnivariateSpline(rhop_vec, np.log(ne_vec))
    rhop_short = np.linspace(0.0, np.max(rhop_vec), profpnts)
    ne_short = np.exp(ne_spline(rhop_short)) / 1.e19
    Te_short = np.exp(Te_spline(rhop_short)) / 1.e3
    ne_file = open(os.path.join(output_path, 'ne.dat'), 'w')
    ne_file.write("    {0: n}\n".format(profpnts))
    for i in range(profpnts):
        ne_file.write("{0: 1.12e} {1: 1.12e}\n".format(rhop_short[i], ne_short[i]))
    ne_file.flush()
    ne_file.close()
    Te_file = open(os.path.join(output_path, 'Te.dat'), 'w')
    Te_file.write("    {0: n}\n".format(profpnts))
    for i in range(profpnts):
        Te_file.write("{0: 1.12e} {1: 1.12e}\n".format(rhop_short[i], Te_short[i]))
    Te_file.flush()
    Te_file.close()
    print("Te data successfully written to " + os.path.join(output_path, "Te.dat"))
    print("ne data successfully written to " + os.path.join(output_path, "ne.dat"))
    return 0




# import IPython


class EQU:

    def __init__(self , Shotnumber=None):
        self.Status = False
        if Shotnumber is not None :
            self.Load(Shotnumber, Diagnostic="EQH")

    def __del__(self):
        self.Unload()
        del self.Status

    def Load(self , Shotnumber, Experiment='AUGD', Diagnostic='EQI', Edition=0L):
        self.Unload()
        if Diagnostic == 'EQI' or Diagnostic == 'EQH' or Diagnostic == 'EQB':
            try:
                sf = dd.shotfile(Diagnostic, Shotnumber, Experiment, Edition)
                self.Shotnumber = Shotnumber
            except:
                print "Error reading shotfile"
                return False

            self.Nz = sf.getParameter('PARMV', 'N').data + 1
            self.NR = sf.getParameter('PARMV', 'M').data + 1
            self.Ntime = sf.getParameter('PARMV', 'NTIME').data + 1
            self.R = (sf.getSignalGroup("Ri"))[0:self.Ntime, 0:self.NR]
            self.z = (sf.getSignalGroup("Zj"))[0:self.Ntime, 0:self.Nz]
            self.time = (sf.getSignal("time"))[0:self.Ntime]
            self.PsiOrigin = sf.getSignalGroup("PFM")[0:self.Ntime, 0:self.Nz, 0:self.NR]
            # #time, R, z
            self.Psi = np.swapaxes(self.PsiOrigin, 1, 2)
            self.ed = sf.edition
            sf.close()

            self.Status = True



    def Unload(self):
        if self.Status:
            self.Status = False
            del self.Nz
            del self.NR
            del self.Ntime
            del self.R
            del self.z
            del self.time
            del self.Psi


    def getPsi(self, timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.Psi[idx]

    def getR(self, timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.R[idx]

    def getz(self, timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.z[idx]

    def __call__(self , timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.R[idx], self.z[idx], self.Psi[idx]


exp_eq = 'AUGD'
dia_eq = 'EQH'

class kkhelp:
    status = False

class KK:
    def kkeqpfx(self, nshot, tshot, exp=exp_eq, diag=dia_eq, ed=0):

        nr = 4
        nr1 = nr + 1
# Input
        c_ed = ct.c_long(ed)
        _ed = ct.byref(c_ed)
        c_err = ct.c_long(0)
        _err = ct.byref(c_err)
        c_shot = ct.c_long(nshot)
        _shot = ct.byref(c_shot)
        c_exp = ct.c_char_p(exp)
        c_dia = ct.c_char_p(diag)
        c_tshot = ct.c_float(tshot)
        _tshot = ct.byref(c_tshot)
# Special input
        c_lpfx = ct.c_long(nr)
        _lpfx = ct.byref(c_lpfx)
# Output
        c_pfx = (ct.c_float * nr1)()
        c_rpfx = (ct.c_float * nr1)()
        c_zpfx = (ct.c_float * nr1)()
        _pfx = ct.byref(c_pfx)
        _rpfx = ct.byref(c_rpfx)
        _zpfx = ct.byref(c_zpfx)

        status = libkk.kkeqpfx(_err, c_exp, c_dia, c_shot, _ed, _tshot, \
                               _lpfx, _pfx, _rpfx, _zpfx)

        output = kkhelp()
        output.ed = c_ed.value
        output.err = c_err.value
# Float output
        output.Raxis = c_rpfx[0]
        output.zaxis = c_zpfx[0]
        output.Rspx = c_rpfx[1]
        output.zspx = c_zpfx[1]
        output.Rlim = c_rpfx[2]
        output.zlim = c_zpfx[2]
        output.psiaxis = c_pfx[0]
        output.psispx = c_pfx[1]
        return output

    def kkrzBrzt(self, nshot, tshot, Rin, zin, exp=exp_eq, diag=dia_eq, ed=0):

# Input
        nr = len(Rin)
        c_nr = ct.c_long(nr)
        _nr = ct.byref(c_nr)
        c_ed = ct.c_long(ed)
        _ed = ct.byref(c_ed)
        c_err = ct.c_long(0)
        _err = ct.byref(c_err)
        c_shot = ct.c_long(nshot)
        _shot = ct.byref(c_shot)
        c_exp = ct.c_char_p(exp)
        c_dia = ct.c_char_p(diag)
        c_tshot = ct.c_float(tshot)
        _tshot = ct.byref(c_tshot)

# Special input
        c_Rin = (ct.c_float * nr)()
        c_Rin[:] = Rin[:]
        _Rin = ct.byref(c_Rin)
        c_zin = (ct.c_float * nr)()
        c_zin[:] = zin[:]
        _zin = ct.byref(c_zin)

# Output    Br, Bz, Bt,      fPF, fJp
        fpf = (ct.c_float * nr)()
        fjp = (ct.c_float * nr)()
        br = (ct.c_float * nr)()
        bz = (ct.c_float * nr)()
        bt = (ct.c_float * nr)()
        _fpf = ct.byref(fpf)
        _fjp = ct.byref(fjp)
        _br = ct.byref(br)
        _bz = ct.byref(bz)
        _bt = ct.byref(bt)

        status = libkk.kkrzbrzt(_err, c_exp, c_dia, c_shot, _ed, _tshot, _Rin, _zin, \
                                c_nr, _br, _bz, _bt, _fpf, _fjp)

        output = kkhelp()
        output.ed = c_ed.value
        output.err = c_err.value

# Numpy output
        output.br = np.array(br[0:nr])
        output.bz = np.array(bz[0:nr])
        output.bt = np.array(bt[0:nr])
        output.fpf = np.array(fpf[0:nr])
        output.fjp = np.array(fjp[0:nr])

        return output

def __init__():
    make_topfile(int(sys.argv[1]), float(sys.argv[2]))
__init__()
