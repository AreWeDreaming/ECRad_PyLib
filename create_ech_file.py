#!/usr/bin/env python

import numpy as np
import os, sys
import ctypes as ct
import dd_20140409 as dd
import datetime
from IPython import embed

shot = input('shotnumber?\n') #31113
t = input('time in seconds?\n') #1.93
#embed()

# don't modify from here unless you know what you are doing

ECS = dd.shotfile()
ECS.Open('ECS', shot)


lib_path = '/afs/ipp/u/ecrh/lib/libaug_ecrh_setmirrors__git.so'

#lib = ct.cdll.LoadLibrary('/afs/ipp-garching.mpg.de/home/e/ecrh/@sys/lib/libaug_ecrh_setmirrors.so')
lib = ct.cdll.LoadLibrary(lib_path)

def ammbeta2poltor(ecrhN, gyroN, shot, edition, amm, beta):
	ecspath = '/afs/ipp/u/augd/shots/%i/L1/ECS/%i.%i' % (shot/10, shot, edition)
	dateasdouble = float(datetime.datetime.fromtimestamp(os.path.getctime(ecspath)).strftime("%Y%m%d.%H%M"))
	sysunt = ct.c_int32(ecrhN*100 + gyroN)
	datum = ct.c_double(dateasdouble)
	amm = ct.c_double(amm)
	beta = ct.c_double(beta)
	alpha = ct.c_double(0)
	theta = ct.c_double(0)
	phi = ct.c_double(0)
	error = ct.c_int32(0)
	lib.ab2getall_ (ct.byref(sysunt), ct.byref(datum), ct.byref(amm), ct.byref(beta), ct.byref(alpha), ct.byref(theta), ct.byref(phi), ct.byref(error))
	return {'pol':theta.value, 'tor':phi.value, 'error':error.value}

paramsets = [
'P_sy1_g1', # gyrotron 1 of ECRH1
'P_sy1_g2', # gyrotron 2 of ECRH1
'P_sy1_g3', # gyrotron 3 of ECRH1
'P_sy1_g4', # gyrotron 4 of ECRH1
'P_sy2_g1', # gyrotron 1 of ECRH2
'P_sy2_g2', # gyrotron 2 of ECRH2
'P_sy2_g3', # gyrotron 3 of ECRH2
'P_sy2_g4'  # gyrotron 4 of ECRH2
]
items = ['GPolPos', 'GTorPos']   # ECRH1: theta, phi; ECHR2: beta, a
# pol <-> a ; tor <-> beta
mapping = [1]*4 + [2]*4 # see above, transfer as appropriate using library

# gyro specifications from Emil:
R_R = [238,  238,  231.1,  231.1,  236.1,   236.1,   236.1,   236.1 ]
Z_Z = [  0,    0,      0,      0,     32,      32,     -32,     -32 ]

# gyr_freq <--- freq item

outf = open('astra_ecrh_%i.txt'%shot, 'w')

print >>outf, '!ECH file'

print "Template for Gyrotron settings, these settings for central heating"
print ""
print "0 Gyrotron  f(GHz)  Power(MW)  Mode  ModeNr theta(deg)  phi(deg)"

for i in xrange(8):
	pol = ECS.GetParameter(paramsets[i], 'GPolPos')
	tor = ECS.GetParameter(paramsets[i], 'GTorPos')
	f = ECS.GetParameter(paramsets[i], 'gyr_freq')
	if mapping[i] == 2: # ECRH 2 => conversion necessary
		res = ammbeta2poltor((i / 4 + 1), i % 4 + 1, shot, ECS.edition, pol*1e3, tor)
		pol = res['pol']
		tor = res['tor']
	power = ECS.GetSignal('PG%i'%(i+1) if i < 4 else 'PG%iN'%(i-3))[np.abs(ECS.GetTimebase('T-B')-t).argmin()]
	#print i+1, pol, tor
	#print "0 Gyrotron  f(GHz)  Power(MW)  Mode  ModeNr theta(deg)  phi(deg)"
	print "%i             %3i     %3.2f      0       0       %4.2f    %4.2f"%(i+1, round(f/1e9), round(power/1e6,2), round(pol,2), round(tor,2))
	print >>outf, '''  %i
%s       +0.0000e+00 %+5.4e %+5.4e %+5.4e
%+5.4e %+5.4e +0.0000e+00 +0.0000e+00 +0.0000e+00
+0.0000e+00''' % (i+1, 'ZRD%i'%(31+i), f/1e9, R_R[i], Z_Z[i], pol, tor)

outf.close()
