import ctypes as ct
# import scipy as sp
import numpy as np
import os
verbose = False
libkk = ct.cdll.LoadLibrary('/afs/ipp-garching.mpg.de/aug/ads/lib64/amd64_sles11/libkk.so')
__libkk__ = ct.cdll.LoadLibrary('/afs/ipp-garching.mpg.de/aug/ads/lib64/%s/libkk.so' % os.environ['SYS'])

__libc__ = ct.CDLL('libc.so.6')
__savedstdout__ = __libc__.dup(1)
__stdout__ = __libc__.fdopen(1, "w")

def __blockstdout__():
    __libc__.freopen("/dev/null", "w", __stdout__)

def __releasestdout__():
    __libc__.freopen("/dev/fd/%s" % str(__savedstdout__), "w", __stdout__)


class point:
    R = 0.0
    z = 0.0
    rho = 0.0
    def __init__(self , z=0.0 , R=0.0 , rho=0.0):
        self.z = z
        self.R = R
        self.rho = rho

    def __lt__(self , right):
        return self.z < right.z

    def __le__(self , right):
        return self.z <= right.z

    def __gt__(self , right):
        return self.z > right.z

    def __ge__(self , right):
        return self.z >= right.z


class magneticField(object):
    def __init__(self, time, R, z, Bt, Bz, Br, fpf, jpol):
        object.__init__(self)
        self.time = time
        if np.size(Bt) != 1:
            self.R = R
            self.z = z
            self.Bt = Bt
            self.Bz = Bz
            self.Br = Br
            self.fpf = fpf
            self.jpol = jpol
        else:
            self.R = R[0]
            self.z = z[0]
            self.Bt = Bt[0]
            self.Bz = Bz[0]
            self.Br = Br[0]
            self.fpf = fpf[0]
            self.jpol = jpol[0]
        self.Bpol = np.hypot(self.Bz, self.Br)

    __getitem__ = object.__getattribute__

    def keys(self):
        return self.__dict__.keys()

class equilibrium(object):
    def __init__(self , shotnumber=None, experiment='AUGD', diagnostic='EQH', edition=0, verbose=False):
        self.__status = False
        self.__shotnumber = ct.c_int(0)
        self.shotnumber = ct.c_int32(0)
        self.__edition = ct.c_int(0)
        self.edition = ct.c_int32(0)
        self.experiment = b''
        self.diagnostic = b''
        self.verbose = verbose
        if shotnumber is not None:
            self.Open(shotnumber , experiment , diagnostic , edition)


    def __del__(self):
        self.Close()

    def status():
        def fget(self):
            return self.shotnumber.value != 0
        return locals()
    status = property(**status())

    def Open(self , shotnumber , exper='AUGD' , diag='EQH' , edition=0):
        self.Close()
        shotnumber = int(shotnumber)
        if shotnumber > 0:
            self.__status = True
            self.__shotnumber = ct.c_int(shotnumber)
            self.shotnumber.value = shotnumber
            self.__edition = ct.c_int(edition)
            self.edition.value = edition
            self.__diag = ct.c_char_p(diag)
            try:
                self.experiment = exper.encode()
            except Exception:
                self.experiment = exper
            try:
                self.diagnostic = diag.encode()
            except Exception:
                self.diagnostic = diag
            self.__exper = ct.c_char_p(exper)
        return self.status

    def Close(self):
        if self.__status:
            self.__status = False
            self.__shotnumber = ct.c_int(0)
            self.__edition = ct.c_int(0)
            self.shotnumber.value = 0
            self.edition.value = 0
            self.experiment = b''
            self.diagnostic = b''



    def getMagneticField(self, time, R, z):
        if not self.status:
            raise Exception('No shotnumber specified.')
        error = ct.c_int32(0)
        t = ct.c_float(time)
        N = np.size(R)
        rin = np.empty(N, dtype=np.float32)
        rin[:] = R
        zin = np.empty(N, dtype=np.float32)
        zin[:] = z
        lin = ct.c_int32(rin.size)
        br = np.zeros(rin.size, dtype=np.float32)
        bz = np.zeros(rin.size, dtype=np.float32)
        bt = np.zeros(rin.size, dtype=np.float32)
        fpf = np.zeros(rin.size, dtype=np.float32)
        jpol = np.zeros(rin.size, dtype=np.float32)
        lexper = ct.c_uint64(len(self.experiment))
        ldiag = ct.c_uint64(len(self.diagnostic))
        if not verbose:
            __blockstdout__()
        __libkk__.kkrzbrzt_(ct.byref(error), ct.c_char_p(self.experiment), ct.c_char_p(self.diagnostic),
                            ct.byref(self.shotnumber), ct.byref(self.edition), ct.byref(t), rin.ctypes.data_as(ct.c_void_p),
                            zin.ctypes.data_as(ct.c_void_p), ct.byref(lin) , br.ctypes.data_as(ct.c_void_p),
                            bz.ctypes.data_as(ct.c_void_p) , bt.ctypes.data_as(ct.c_void_p), fpf.ctypes.data_as(ct.c_void_p),
                            jpol.ctypes.data_as(ct.c_void_p), lexper , ldiag)
        if not verbose:
            __releasestdout__()
        return magneticField(t.value, rin, zin, bt, br, bz, fpf, jpol)

    B = getMagneticField

    def Rz_to_rhopol(self, time, R, z):
        if not self.status:
            raise Exception('No shotnumber specified.')
        if np.size(R) != np.size(z):
            raise Exception('size R != size z')

        N = np.size(R)
        time = np.atleast_1d(time)
        nt = len(time)

        rhoPol = np.zeros((nt, N), dtype=np.float32)
        fpf = np.zeros((nt, N), dtype=np.float32)
        error = ct.c_int32(0)

        rin = np.zeros(N, dtype=np.float32)
        zin = np.zeros(N, dtype=np.float32)
        rin[:] = R
        zin[:] = z
        lin = ct.c_int32(N)
        lexper = ct.c_uint64(len(self.experiment))
        ldiag = ct.c_uint64(3)
        ed = ct.byref(self.edition)


        if not verbose: __blockstdout__()

        for i, t in enumerate(np.ravel(time)):
            t = ct.c_float(t)
            __libkk__.kkrzpfn_(ct.byref(error), ct.c_char_p(self.experiment), ct.c_char_p(self.diagnostic),
                            ct.byref(self.shotnumber), ed, ct.byref(t),
                            rin.ct.data_as(ct.c_void_p), zin.ct.data_as(ct.c_void_p), ct.byref(lin),
                            fpf[i, :].ct.data_as(ct.c_void_p), rhoPol[i, :].ct.data_as(ct.c_void_p), lexper, ldiag)
        __releasestdout__()

        return np.squeeze(rhoPol)

    def s_to_Rz(self , s):

        N = np.size(s)
        if not self.__status:
            raise Exception('No shotnumber specified.')

        error = ct.c_int(0)
        _error = ct.byref(error)
        _shotnumber = ct.byref(self.__shotnumber)
        _edition = ct.byref(self.__edition)
        if N == 1:
            sin = ct.c_float(s)
        else:
            sin = (ct.c_float * N)()
            for i in range(N):
                sin[i] = s[i]
        _sin = ct.byref(sin)
        Raus = (ct.c_float * N)()
        zaus = (ct.c_float * N)()
        aaus = (ct.c_float * N)()
        _Raus = ct.byref(Raus)
        _zaus = ct.byref(zaus)
        _aaus = ct.byref(aaus)
        length = ct.c_int(N)
        _length = ct.byref(length)
        lexper = ct.c_long(len(self.experiment))
        ldiag = ct.c_long(len(self.diagnostic))
        libkk.kkgcsrza_(_error , self.__exper, self.__diag, _shotnumber, _edition, _length, _sin, _Raus, _zaus, _aaus, lexper, ldiag)
        output = []
        for i in range(N):
            output.append(point(zaus[i] , Raus[i] , 0.0))
        return output
        # return np.nan


    def rhopol_to_Rz(self, time, rhopol, angle, degrees=False):  # angle in degrees...
        N = np.size(rhopol)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)

            if N == 1:
                rhopf = ct.c_float(rhopol) if isinstance(rhopol, float) else ct.c_float(rhopol[0])
                Rn = ct.c_float(0)
                zn = ct.c_float(0)
            else:
                rhopf = (ct.c_float * N)()
                Rn = (ct.c_float * N)()
                zn = (ct.c_float * N)()
                for i in range(N):
                    rhopf[i] = rhopol[i]
                    Rn[i] = 0
                    zn[i] = 0
            _rhopf = ct.byref(rhopf)
            _Rn = ct.byref(Rn)
            _zn = ct.byref(zn)
            lrho = ct.c_int(N)
            _lrho = ct.byref(lrho)
            iorg = ct.c_int(0)
            _iorg = ct.byref(iorg)
            ang = ct.c_float(angle) if degrees else ct.c_float(angle / np.pi * 180.)
            _angle = ct.byref(ang)

            lexper = ct.c_long(len(self.__exper))
            ldiag = ct.c_long(3)

            libkk.kkrhorz_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t, \
                            _rhopf, _lrho, _iorg, _angle, \
                            _Rn, _zn, \
                            lexper, ldiag)
            if N == 1:
                return {'error' : np.int_(error), \
                        'time'  : np.float32(t), \
                        'R'     : np.float32(Rn), \
                        'z'     : np.float32(zn)}
            else:
                return {'error' : np.int_(error), \
                        'time'  : np.float32(t), \
                        'R'     : np.array(Rn), \
                        'z'     : np.array(zn)}


    def rhopol_to_q(self, time, rhopol):
        N = np.size(rhopol)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)

            if N == 1:
                rhopf = ct.c_float(rhopol) if isinstance(rhopol, float) else ct.c_float(rhopol[0])
                q = ct.c_float(0)
                pf = ct.c_float(0)
            else:
                rhopf = (ct.c_float * N)()
                q = (ct.c_float * N)()
                pf = (ct.c_float * N)()
                for i in range(N):
                    rhopf[i] = rhopol[i]
                    q[i] = 0
                    pf[i] = 0
            _rhopf = ct.byref(rhopf)
            _q = ct.byref(q)
            _pf = ct.byref(pf)
            lrho = ct.c_int(N)
            _lrho = ct.byref(lrho)

            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            libkk.kkrhopfq_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t, \
            _rhopf, _lrho, \
            _q, _pf, \
            lexper, ldiag)

            if N == 1:
                return {'error' : np.int_(error), \
                'time'  : np.float32(t), \
                'q'     : np.float32(q), \
                'pf'     : np.float32(pf)}
            else:
                return {'error' : np.int_(error), \
                'time'  : np.float32(t), \
                'q'     : np.frombuffer(q, np.float32), \
                'pf'     : np.frombuffer(pf, np.float32)}

    def rhopol_to_rhotor(self, time, rhopol):
        N = np.size(rhopol)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)

            if N == 1:
                rhopf = ct.c_float(rhopol) if isinstance(rhopol, float) else ct.c_float(rhopol[0])
                rhot = ct.c_float(0)
                pf = ct.c_float(0)
                fpf = ct.c_float(0)
                ftf = ct.c_float(0)
            else:
                rhopf = (ct.c_float * N)()
                rhot = (ct.c_float * N)()
                pf = (ct.c_float * N)()
                fpf = (ct.c_float * N)()
                ftf = (ct.c_float * N)()
                for i in range(N):
                    rhopf[i] = rhopol[i]
                    rhot[i] = 0
                    pf[i] = 0
                    fpf[i] = 0
                    ftf[i] = 0
            _rhopf = ct.byref(rhopf)
            _rhot = ct.byref(rhot)
            _pf = ct.byref(pf)
            lrho = ct.c_int(N)
            _lrho = ct.byref(lrho)
            _fpf = ct.byref(fpf)
            _ftf = ct.byref(ftf)

            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            libkk.kkrhopto_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t, \
            _rhopf, _lrho, \
            _rhot, _fpf, _ftf, \
            lexper, ldiag)

            # kkrhopto (&error,exp,diag,&shot,&edition, &time,
            # rhopf, &lrho,
            # rhopf, fpf, ftf , strlen(exp), strlen(diag) );

            if N == 1:
                return {'error' : np.int_(error), \
                'time'  : np.float32(t), \
                'rhotor'     : np.float32(rhot), \
                'fpf'     : np.float32(fpf), \
                'ftf': np.float32(ftf)}
            else:
                return {'error' : np.int_(error), \
                'time'  : np.float32(t), \
                'rhotor'     : np.array(rhot, dtype=float), \
                'fpf'     : np.array(fpf, dtype=float), \
                'ftf': np.array(ftf, dtype=float)}

    def theta_to_sfla(self, time, q, theta, degrees=False):
        N = np.size(theta)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)
            cq = ct.c_float(q)
            _q = ct.byref(cq)
            swrad = ct.c_int(0) if degrees else ct.c_int(1)
            _swrad = ct.byref(swrad)
            langle = ct.c_int(N)
            _langle = ct.byref(langle)
            _Rmag = ct.byref(ct.c_float(0))
            _zmag = ct.byref(ct.c_float(0))
            _tSHf = ct.byref(ct.c_float(0))  #  no idea what that is, docs aren't clear

            # todo: N*_angle, N*_Rn/_zn, N*
            if N == 1:
                angle = ct.c_float(theta)
                Rn = ct.c_float(0)
                zn = ct.c_float(0)
                thetsn = ct.c_float(0)
                Brn = ct.c_float(0)
                Bzn = ct.c_float(0)
                Btn = ct.c_float(0)
            else:
                angle = (ct.c_float * N)()
                Rn = (ct.c_float * N)()
                zn = (ct.c_float * N)()
                thetsn = (ct.c_float * N)()
                Brn = (ct.c_float * N)()
                Bzn = (ct.c_float * N)()
                Btn = (ct.c_float * N)()
                for i in range(N):
                    angle  [i] = theta[i]
                    Rn     [i] = 0
                    zn     [i] = 0
                    thetsn [i] = 0
                    Brn    [i] = 0
                    Bzn    [i] = 0
                    Btn    [i] = 0

            _angle = ct.byref(angle)
            _Rn = ct.byref(Rn)
            _zn = ct.byref(zn)
            _thetsn = ct.byref(thetsn)
            _Brn = ct.byref(Brn)
            _Bzn = ct.byref(Bzn)
            _Btn = ct.byref(Btn)

            libkk.kkeqqfl_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t, \
            _q, _langle, _angle, _swrad, \
            _Rmag, _zmag, _Rn, _zn, _tSHf, \
            _thetsn, _Brn, _Bzn, _Btn, \
            lexper, ldiag)

            if np.int_(error) != 0:
                print 'kkeqqfl_ error ', np.int_(error)
                if np.int_(error) == 14:
                    print 'libkk note: 2*np\'s pi is a little too big for libkk.so; lower it a little bit'
                return {'error': np.int_(error)}

            if N == 1:
                return {'error' : np.int_(error), \
                'time' : np.float32(t), \
                'sfla' : np.float32(thetsn)}
            else:
                return {'error' : np.int_(error), \
                'time'  : np.float32(t), \
                'sfla'  : np.array(thetsn)}


    def psi_to_rhopol(self, time, psi):
        N = np.size(psi)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)
            lrho = ct.c_int(N)
            _lrho = ct.byref(lrho)
            if N == 1:
                PFi = ct.c_float(psi)
                rhoPF = ct.c_float(0)
            else:
                PFi = (ct.c_float * N)()
                rhoPF = (ct.c_float * N)()
                for i in range(N):
                    PFi[i] = psi[i]
                    rhoPF[i] = 0

            _PFi = ct.byref(PFi)
            _rhoPF = ct.byref(rhoPF)

            # kkPFrhoP (iERR ,expnam,dianam,nSHOT,nEDIT,tSHOT,
            #             > PFi,Lrho,
            #             < rhoPF)
            libkk.kkpfrhop_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                _PFi, _lrho,
                _rhoPF,
                lexper, ldiag)
            if N == 1:
                return np.float32(rhoPF)
            else:
                return np.frombuffer(rhoPF, np.float32)

    def get_jpol(self, time, N, NSOL=0):
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            PFL = (ct.c_float * (N + NSOL + 1))()
            Jpol = (ct.c_float * (N + NSOL + 1))()
            Jpolp = (ct.c_float * (N + NSOL + 1))()
            _PFL = ct.byref(PFL)
            _Jpol = ct.byref(Jpol)
            _Jpolp = ct.byref(Jpolp)


            N = ct.c_int(N)
            NSOL = ct.c_int(NSOL)
            _LPFp = ct.byref(N)
            _LPFe = ct.byref(NSOL)

        # kkEQJpolsol (iERR ,expnam,dianam,nSHOT,nEDIT,tSHOT,
        #            < LPFp,LPFe,PFL,Jpol,Jpolp)
            if NSOL.value != 0:
                libkk.kkeqjpolsol_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                    _LPFp, _LPFe, _PFL, _Jpol, _Jpolp,
                    lexper, ldiag)
            else:
                libkk.kkeqjpol_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                    _LPFp, _PFL, _Jpol, _Jpolp,
                    lexper, ldiag)
                # kkeqjpol returns values from the separatrix to the magnetic axis,
                # but kkeqjpolsol from the axis via the separatrix to the scrape off layer.
            return {'pfl':np.frombuffer(PFL, np.float32)[:N.value + NSOL.value + 1],
                    'Jpol':np.frombuffer(Jpol, np.float32)[:N.value + NSOL.value + 1],
                    'Jpolp':np.frombuffer(Jpolp, np.float32)[:N.value + NSOL.value + 1],
                    'N':N.value,
                    'NSOL':NSOL.value}
        pass

    def get_p(self, time, psi):
        N = np.size(psi)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            PFL = (ct.c_float * (N + 1))()
            pres = (ct.c_float * (N + 1))()
            presp = (ct.c_float * (N + 1))()
            _PFL = ct.byref(PFL)
            _pres = ct.byref(pres)
            _presp = ct.byref(presp)
            N = ct.c_int(N)
            _LPF = ct.byref(N)
            # kkeqpres (&error, exp,diag,&shot,&edition, &time,
            #          &lpf, pfl, pres, presp, strlen(exp), strlen(diag) );)
            libkk.kkeqpres_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                            _LPF, _PFL, _pres, _presp,
                            lexper, ldiag)
            return {'pfl':np.frombuffer(PFL, np.float32)[:N.value + 1],
                    'pres':np.frombuffer(pres, np.float32)[:N.value + 1],
                    'presp':np.frombuffer(presp, np.float32)[:N.value + 1],
                    'N':N.value}

    def get_special_points(self, time):
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)
            # kkeqpfx (&error, ,exp,diag,&shot,&edition, &time,
            #         &lpfx, pfxx, rpfx, zpfx,
            #         strlen(exp), strlen(diag) );
            N = 5
            _lpfx = ct.byref(ct.c_int(N - 1))
            pfxx = (ct.c_float * N)()
            rpfx = (ct.c_float * N)()
            zpfx = (ct.c_float * N)()
            _pfxx = ct.byref(pfxx)
            _rpfx = ct.byref(rpfx)
            _zpfx = ct.byref(zpfx)

            # print self.__shotnumber.value

            libkk.kkeqpfx_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                          _lpfx, _pfxx, _rpfx, _zpfx,
                          lexper, ldiag)

            # 0. Magnetic axis
            # 1. Primary X point
            # 2. Primary limiter point
            # 3. Secondary X point
            # 4. Secondary limiter point
            return {'pfxx':np.frombuffer(pfxx, np.float32),
                    'rpfx':np.frombuffer(rpfx, np.float32),
                    'zpfx':np.frombuffer(zpfx, np.float32)}


    def get_ffprime(self, time, psi, typ=11):
        # typ:
        # flag if quantities should be read from shotfile or calculated:
        # 11: read from shotfile if available, otherwise calculate
        #  1: read from shotfile, return error if not available
        #  2: calculate, regardless if available in the shotfile


        N = np.size(psi)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            PFL = (ct.c_float * (N + 1))()
            ffp = (ct.c_float * (N + 1))()
            _PFL = ct.byref(PFL)
            _ffp = ct.byref(ffp)
            N = ct.c_int(N)
            _LPF = ct.byref(N)

            _typ = ct.byref(ct.c_int(typ))

            # kkeqffp_(&error,exp,diag,&shot,&edition, &time,
            #         &typ, &lpf, pfl, ffp, strlen(exp), strlen(diag) );
            libkk.kkeqffp_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                            _typ, _LPF, _PFL, _ffp,
                            lexper, ldiag)
            return {'pfl':np.frombuffer(PFL, np.float32)[:N.value + 1],
                'ffp':np.frombuffer(ffp, np.float32)[:N.value + 1], 'N':N.value}

    def get_Rinv(self, time, psi, typ=11):
        # typ:
        # flag if quantities should be read from shotfile or calculated:
        # 11: read from shotfile if available, otherwise calculate
        #  1: read from shotfile, return error if not available
        #  2: calculate, regardless if available in the shotfile


        N = np.size(psi)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            PFL = (ct.c_float * (N + 1))()
            rinv = (ct.c_float * (N + 1))()
            r2inv = (ct.c_float * (N + 1))()
            _PFL = ct.byref(PFL)
            _rinv = ct.byref(rinv)
            _r2inv = ct.byref(r2inv)
            N = ct.c_int(N)
            _LPF = ct.byref(N)

            _typ = ct.byref(ct.c_int(typ))

            # kkeqffp_(&error,exp,diag,&shot,&edition, &time,
            #         &typ, &lpf, pfl, ffp, strlen(exp), strlen(diag) );
            libkk.kkeqrinv_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                            _typ, _LPF, _PFL, _rinv, _r2inv,
                            lexper, ldiag)
            return {'pfl':np.frombuffer(PFL, np.float32)[:N.value + 1],
                'rinv':np.frombuffer(rinv, np.float32)[:N.value + 1],
                'r2inv':np.frombuffer(r2inv, np.float32)[:N.value + 1], 'N':N.value}


    def get_jpar(self, time, N, typ=11):
            # void kkeqjpar_( int*, char*, char*, int*, int*, float*,
            #                int*, int*,
            #                float*, float*, float*,
            #                long, long);
            if self.__status:
                error = ct.c_int(0)
                _error = ct.byref(error)
                _shotnumber = ct.byref(self.__shotnumber)
                _edition = ct.byref(self.__edition)
                t = ct.c_float(time)
                _t = ct.byref(t)
                lexper = ct.c_long(len(self.experiment))
                ldiag = ct.c_long(3)

                PFL = (ct.c_float * (N + 1))()
                Jpar = (ct.c_float * (N + 1))()
                Jparp = (ct.c_float * (N + 1))()
                _PFL = ct.byref(PFL)
                _Jpar = ct.byref(Jpar)
                _Jparp = ct.byref(Jparp)


                N = ct.c_int(N)
                _LPFp = ct.byref(N)
                _typ = ct.byref(ct.c_int(typ))

                libkk.kkeqjpar_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                    _typ, _LPFp,
                    _PFL, _Jpar, _Jparp,
                    lexper, ldiag)

                return {'pfl':np.frombuffer(PFL, np.float32)[:N.value + 1],
                            'Jpar':np.frombuffer(Jpar, np.float32)[:N.value + 1],
                            'Jparp':np.frombuffer(Jparp, np.float32)[:N.value + 1],
                            'N':N.value}

    def get_pfm(self, time, m=65, n=129):
        o_m = m
        o_n = n
        print("initial dimensions:", o_m, o_n)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            mdim = ct.c_int(m)
            m = ct.c_int(m)
            n = ct.c_int(n)
            Ri = (ct.c_float * (m.value))()
            zj = (ct.c_float * (n.value))()
            pfm = ((ct.c_float * (m.value)) * n.value)()

            _mdim = ct.byref(mdim)
            _m = ct.byref(m)
            _n = ct.byref(n)
            _Ri = ct.byref(Ri)
            _zj = ct.byref(zj)
            _pfm = ct.byref(pfm)
            print("Final dimensions:", m.value, n.value)
            # kkeqpfm_( int*, char*, char*, int*, int*, float*,
            #   &mdim, &m, &n, Ri, zj, pfm,
            #   long, long);
            libkk.kkeqpfm_(_error, self.__exper, self.__diag, _shotnumber, _edition, _t,
                            _mdim, _m, _n, _Ri, _zj, _pfm,
                            lexper, ldiag)

            return {
                'pfm':np.frombuffer(pfm, np.float32).reshape(o_n, o_m)[:n.value + 1, :m.value + 1],
                'Ri':np.frombuffer(Ri, np.float32)[:m.value + 1],
                'zj':np.frombuffer(zj, np.float32)[:n.value + 1]
                }

    def psi_to_v(self, time, psis):
        N = np.size(psis)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            fpf = (ct.c_float * N)()
            for i in range(N):
                fpf[i] = psis[i]
            _fpf = ct.byref(fpf)
            lin = ct.c_int(N)

            v = (ct.c_float * N)()
            work = (ct.c_float * N)()
            _v = ct.byref(v)
            _work = ct.byref(work)

            libkk.kkpfv(_error, self.__exper, self.__diag, self.__shotnumber, _edition, _t,
                            fpf, lin,
                            _v, _work)  # ,

            return {'volume': np.frombuffer(v, np.float32), 'workingarea': np.frombuffer(work, np.float32)}


            # libkk.kkpfv(byref(error), byref(exp), byref(diag), shot, byref(edition),
            #        byref(tshot),
            #        fpf, lin,
            #        byref(v), byref(work) )

    def get_volume(self, time, psi):
        N = np.size(psi)
        if self.__status:
            error = ct.c_int(0)
            _error = ct.byref(error)
            _shotnumber = ct.byref(self.__shotnumber)
            _edition = ct.byref(self.__edition)
            t = ct.c_float(time)
            _t = ct.byref(t)
            lexper = ct.c_long(len(self.experiment))
            ldiag = ct.c_long(3)

            PFL = (ct.c_float * (N + 1))()
            vol = (ct.c_float * (N + 1))()
            volp = (ct.c_float * (N + 1))()
            _PFL = ct.byref(PFL)
            _vol = ct.byref(vol)
            _volp = ct.byref(volp)
            N = ct.c_int(N)
            _LPF = ct.byref(N)

            libkk.kkeqvol(_error, self.__exper, self.__diag, self.__shotnumber, _edition, _t,
                          _LPF, _PFL, _vol, _volp);

            return {'pfl':np.frombuffer(PFL, np.float32)[:N.value + 1],
                'vol':np.frombuffer(vol, np.float32)[:N.value + 1],
                'volp':np.frombuffer(volp, np.float32)[:N.value + 1], 'N':N.value}


        # libkk.kkeqvol(byref(error), exp,diag, shot, byref(edition), byref(time),
        #            byref(lpf), byref(pfl), byref(vol), byref(volp));

            # kkeqffp_(&error,exp,diag,&shot,&edition, &time,
            #         &typ, &lpf, pfl, ffp, strlen(exp), strlen(diag) );
            # libkk.kkeqffp_(_error, self.__exper, self.__diag, _shotnumber,_edition,_t,
            #                _typ, _LPF, _PFL, _ffp,
            #                lexper, ldiag)
            # return {'pfl':np.frombuffer(PFL, np.float32)[:N.value+1],
            #    'ffp':np.frombuffer(ffp, np.float32)[:N.value+1], 'N':N.value}



# void kkeqqfld_( int*, char*, char*, int*, int*, float*,
#                float*, int*, float*, int*,
#                float*, float*, float*, float*, float*,
#                float*, float*, float*, float*, float*,
#                long, long);

# C                        +---------------------------------+
# C              (3.4)      calc.  straight field line angle
# C                        +---------------------------------+
# C
# C     ___________________ {EQU,FPP} calc. straight f.l.angle for  qval:
# C                kkEQqFL  (iERR  ,expnam,dianam,nSHOT,nEDIT, tSHOT,
# C               >                 qval,Nangl,angle,swrad,
# C               <                 Rmag,zmag,Rn,zn,           tSHf ,
# C               <                 thetsn,     Brn,Bzn,Btn)
# C                kkEQqFLd (iERR  ,expnam,dianam,nSHOT,nEDIT, tSHOT,
# C               >                 qval,Nangl,angle,swrad,
# C               <                 Rmag,zmag,Rn,zn,           tSHf ,
# C               <                 thetsn,dtht,Brn,Bzn,Btn)
#C                -------------------------------------------------------


# C     (3.4)      calc.  straight field line angle ..kkEQqFL,d ( )
# C        qval   ...q_value                         (real* 4)      -> <--
# C        Nangl  ...# of elements in angle          (integer)      ->
# C        angle  ...1D array of angle [dgr] | [rad] (real *4)      ->
# C        swrad  ...switch for units of angles:                    ->
# C               := 0 ...angles in [degrees]
# C               := 1 ...angles in [radians]
# C        R,zmag ...position of magn.axis  [m]      (real *4)         <--
# C        Rn,zn  ...1D arrays of surface points     (real *4)         <--
# C        thetsn ...straight_field_line_angle       (real *4)         <--
# C               ...1D array: [dgr] | [rad]
# C        dtht   ...d/dtheta(thetsn)                (real *4)         <--
# C        Br,z,tn...1D arrays of B_components [T]   (real *4)         <--
#




