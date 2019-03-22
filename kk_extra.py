'''
Created on Aug 11, 2016

@author: sdenk
'''
import ctypes
import scipy
import numpy
from IPython import embed
import sys

libkk = ctypes.cdll.LoadLibrary('/afs/ipp-garching.mpg.de/aug/ads/lib64/amd64_sles11/libkk8x.so')


class point:
    R = 0.0
    z = 0.0
    rho = 0.0
    def __init__(self , z=0.0 , R=0.0 , rho=0.0):
        self.z = z
        self.R = R
        self.rho = rho

    def __lt__(self , right):
        if self.z < right.z:
            return True
        return False

    def __le__(self , right):
        if self.z <= right.z:
            return True
        return False

    def __gt__(self , right):
        if self.z > right.z:
            return True
        return False

    def __ge__(self , right):
        if self.z >= right.z:
            return True
        return False



class kk_extra:
    __status = False
    __shotnumber = ctypes.c_int(0)
    __edition = ctypes.c_int(0)

    def __init__(self , shotnumber=None, experiment='AUGD', diagnostic='FPP', edition=0):
        self.__status = False
        self.__shotnumber = ctypes.c_int(0)
        self.__edition = ctypes.c_int(0)
        if shotnumber is not None:
            self.Open(shotnumber , experiment , diagnostic , edition)


    def __del__(self):
        self.Close()


    def Open(self , shotnumber , exper='AUGD' , diag='FPP' , edition=0):
        self.Close()
        if shotnumber > 0:
            self.__status = True
            self.__shotnumber = ctypes.c_int(shotnumber)
            self.__edition = ctypes.c_int(edition)
            self.__diag = ctypes.c_char_p(diag)
            vars(self)['experiment'] = exper
            vars(self)['diagnostic'] = diag
            self.__exper = ctypes.c_char_p(exper)
        return True



    def Close(self):
        if self.__status:
            self.__status = False
            self.__shotnumber = ctypes.c_int(0)
            self.__edition = ctypes.c_int(0)
            del self.experiment
            del self.diagnostic

    def get_f(self, time, psi, typ=11):
        # typ:
        # flag if quantities should be read from shotfile or calculated:
        # 11: read from shotfile if available, otherwise calculate
        #  1: read from shotfile, return error if not available
        #  2: calculate, regardless if available in the shotfile

        N = numpy.size(psi)
        if self.__status:
            error = ctypes.c_int(0)
            _error = ctypes.byref(error)
            _shotnumber = ctypes.byref(self.__shotnumber)
            _edition = ctypes.byref(self.__edition)
            t = ctypes.c_float(time)
            _t = ctypes.byref(t)
            lexper = ctypes.c_long(len(self.experiment))
            ldiag = ctypes.c_long(3)

            PFL = (ctypes.c_float * (N + 1))()
            f = (ctypes.c_float * (N + 1))()
            fp = (ctypes.c_float * (N + 1))()
            _PFL = ctypes.byref(PFL)
            _f = ctypes.byref(f)
            _fp = ctypes.byref(fp)
            N = ctypes.c_int(N)
            _LPF = ctypes.byref(N)

            c_typ = ctypes.c_int(typ)

            # kkeqffp_(&error,exp,diag,&shot,&edition, &time,
            #         &typ, &lpf, pfl, ffp, strlen(exp), strlen(diag) );
            libkk.kkeqffs(_error, self.__exper, self.__diag, self.__shotnumber, _edition, _t,
                            c_typ, _LPF, _PFL, _f, _fp,
                            lexper, ldiag)
            return {'pfl':numpy.frombuffer(PFL, numpy.float32)[:N.value + 1],
                'f':numpy.frombuffer(f, numpy.float32)[:N.value + 1], \
                'fp':numpy.frombuffer(fp, numpy.float32)[:N.value + 1], 'N':N.value}

