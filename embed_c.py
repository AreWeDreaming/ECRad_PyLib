'''
Created on Dec 19, 2016

@author: sdenk
'''

import ctypes as ct
libjuettner = ct.cdll.LoadLibrary("libeJp.so")
f = ct.c_double(0.0)
libjuettner.Juettner_BiNorm(ct.c_double(Te_par), ct.c_double(Te_perp), \
    ct.byref(f))
return f.value
