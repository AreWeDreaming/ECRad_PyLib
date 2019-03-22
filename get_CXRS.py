#!/usr/bin/env python
# line above sets the Python interpreter
#
# Created by Teobaldo Luda - @IPP - February 2018
# --- credits to: Nils Leuthold, Giovanni Tardini
#                 Philip Schneider, Alexander Bock
#	          Clemente Angioni, Aaron Ho
#
#
# ---> ICRH is not taken into account yet !!!!!!!!!!
#
#
import itertools
import threading
import time
import sys

done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\r...loading modules ' + c)
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write('\rDone!     ')

t = threading.Thread(target=animate)
t.start()

#long process here
sys.path.append('/afs/ipp/home/g/git/python/repository/')
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib/')
sys.path.append('/afs/ipp/home/g/git/python/trgui')
sys.path.append('/afs/ipp/home/a/aho/common/')
#sys.path.append('/afs/ipp/home/t/tluda/GPR1D/')
import os
import matplotlib.pylab as plt 			# load plotting library
#import dd, map_equ_20161123, dd_20140407       	# load latest (!) dd library
import map_equ_20180130
import dd_20180130 as dd
import numpy as np
import math
import mom2rz
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from GPR1D import GPR1D
import IPython
from pylab import savefig
from os.path import expanduser

cwd = expanduser("~")
#os.chdir('~')
#cwd = os.getcwd()



#eqm = map_equ_20161123.equ_map()
eqm = map_equ_20180130.equ_map()
sf = dd.shotfile()

time.sleep(1)
done = True

print ('\n\nWorking in the folder: %s' %(cwd))

# use "import dd_YYYYMMDD as dd" to safely load a specific
# version from /afs/ipp/aug/ads-diags/common/python/lib

#print ("\n~~~~~~~~~~~~~")
#shot = (raw_input("Shot: "))
#time = (raw_input("Time: "))
#print ("You chose SHOT=%s at TIME=%s" %(shot,time))

if len(sys.argv) > 1:
	try:
	    	shot = int(sys.argv[1])
		tBeg = float(sys.argv[2])
		tEnd = float(sys.argv[3])
	except:
		shot = 27666#34244
		#shot = 25521
		shot = 16217
		shot = 32201
		shot = 30693

		time = 2.1

		tBeg = time-0.1
		tEnd = time+0.1
		#tBeg = 1.55
		#tEnd = 1.85



print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print (' Creating ASTRA exp file for Shot #%d ' %shot)
print ('     between %.2f and %.2f seconds...    ' %(tBeg, tEnd))
print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


time = np.mean([tBeg,tEnd])

#print (time)
#print (tBeg)
#print (tEnd)

elm_filter = False   # Filter out ELMs
GPfit = True	    # Gaussian fit or Spline
vts_coredge = True  # Thomson scattering:  True  -> Core+Edge
		    #                      False -> only Core
ASTRA_flag = False    # Write input file for ASTRA, otherwise write profiles in text file



try: 
    eqm.Open(shot, diag='EQH')
    eqm._read_profiles()
except:
    print('\n...EQH not available, reading EQI...\n')
    eqm.Open(shot, diag='EQI')
    eqm._read_profiles()

#----------------------------------------------------------------> CXRS
#plt.figure()
diags = ['CEZ', 'CMZ', 'CUZ', 'COZ', 'CPZ', 'CHZ'] 
marker = ['<', '+', '^', '.', 'o', '>']

def readCZ(diag):
    sf.Open(diag, shot)
    try:
        CZ_Ti = sf.GetSignal('Ti_c')
        CZ_err_Ti = sf.GetSignal('err_Ti_c')
    except:
	print('Could not read Ti_c from %s' %diag)
    if CZ_Ti==None:
        CZ_Ti = sf.GetSignal('Ti')
        CZ_err_Ti = sf.GetSignal('err_Ti')
    CZ_vrot = sf.GetSignal('vrot')
    CZ_err_vrot = sf.GetSignal('err_vrot')
    try:
        CZ_time = sf.GetTimebase('time')
    except:
        CZ_time = sf.GetTimebase('T-'+diag)  
    #print (CZ_Ti)
    
    
    CZ_mask = np.argmin(np.abs(CZ_time - time))
    CZ_time = CZ_time[CZ_mask]
    try:
        CZ_R = sf.GetAreabase('R_time')
        CZ_z = sf.GetAreabase('z_time')
        CZ_R = CZ_R[CZ_mask]
        CZ_z = CZ_z[CZ_mask]
        #CZ_R = np.mean(CZ_R, axis=0)
        #CZ_z = np.mean(CZ_z, axis=0)
    except:
        CZ_R = sf.GetAreabase('R')
        CZ_z = sf.GetAreabase('z')  

    CZ_Ti = CZ_Ti[CZ_mask]
    CZ_err_Ti = CZ_err_Ti[CZ_mask]
    CZ_vrot = CZ_vrot[CZ_mask]
    CZ_err_vrot = CZ_err_vrot[CZ_mask]
    '''
    try:
        CZ_stat = CZ('fit_stat').data
        CZ_stat = CZ_stat[CZ_mask]
        ind = (CZ_stat > 1)
        CZ_err_Ti[ind] = CZ_err_Ti[ind]*2
        CZ_err_vrot[ind] = CZ_err_vrot[ind]*2
    except:
        print('\nfit_stat not available in shotfile\n')
    '''
    sf.Close()

    
    ind = (CZ_R > 0)
    CZ_R = CZ_R[ind]
    CZ_z = CZ_z[ind]
    CZ_Ti = CZ_Ti[ind]
    CZ_err_Ti = CZ_err_Ti[ind]
    CZ_vrot = CZ_vrot[ind]
    CZ_err_vrot = CZ_err_vrot[ind]
    
    ind = (CZ_Ti == 0)
  #  CZ_R = CZ_R[ind[0]]
  #  CZ_z = CZ_z[ind[0]]
    CZ_Ti[ind] = np.nan
    CZ_err_Ti[ind] = np.nan
    CZ_vrot[ind] = np.nan
    CZ_err_vrot[ind] = np.nan
    
    
    Ti  =  CZ_Ti
    err_Ti=CZ_err_Ti
    vrot  =CZ_vrot
    err_vrot=CZ_err_vrot

    CZ_Rhot = np.squeeze(eqm.rz2rho(CZ_R, CZ_z, t_in=time, coord_out='rho_tor'))

#    plt.subplot(121)
#    plt.plot(CZ_Rhot, Ti)
     #   plt.errorbar(IDA_Rhot, Te, yerr=Te_unc, fmt='+')
#    plt.subplot(122)
#    plt.plot(CZ_Rhot, vrot)
     #   plt.errorbar(IDA_Rhot, ne, yerr=ne_unc, fmt='+')

    if elm_filter == True:
        try:
            ELM_mask = [True]*(len(CZ_time))
            for elm,elm_time in enumerate(t_begELM):
             #   print (t_begELM[elm])
             #   print (t_endELM[elm])
                tmp_mask = np.invert((CZ_time > t_begELM[elm]) & (CZ_time < t_endELM[elm]))
                ELM_mask = ELM_mask * tmp_mask

            CZ_time = CZ_time[ELM_mask]
            Ti  =  np.nanmean(CZ_Ti[ELM_mask], axis=0)
            err_Ti=np.nanmean(CZ_err_Ti[ELM_mask], axis=0)
            vrot = np.nanmean(CZ_vrot[ELM_mask], axis=0)
            err_vrot=np.nanmean(CZ_err_vrot[ELM_mask], axis=0)

            Idx = np.argmin(np.abs(CZ_time - time))
            print (CZ_time[Idx])
            CZ_Rhot = np.squeeze(eqm.rz2rho(CZ_R, CZ_z, t_in=CZ_time[Idx], coord_out='rho_tor'))

 #           plt.subplot(121)
 #           plt.plot(CZ_Rhot, Ti, label='without ELMs')
 #           plt.legend(loc='best')
            #   plt.errorbar(IDA_Rhot, Te, yerr=Te_unc, fmt='+')
 #           plt.subplot(122)
 #           plt.plot(CZ_Rhot, vrot)
            #   plt.errorbar(IDA_Rhot, ne, yerr=ne_unc, fmt='+')

        except:
            print('ELM shotfile not available ...')
            '''
            ELM_mask = np.invert(ELM_mask)
            Ti  =  np.mean(CZ_Ti[ELM_mask], axis=0)
            err_Ti=np.mean(CZ_err_Ti[ELM_mask], axis=0)
            vrot = np.mean(CZ_vrot[ELM_mask], axis=0)
            err_vrot=np.mean(CZ_err_vrot[ELM_mask], axis=0)

            plt.subplot(121)
            plt.plot(CZ_Rhot, Ti)
            #   plt.errorbar(IDA_Rhot, Te, yerr=Te_unc, fmt='+')
            plt.subplot(122)
            plt.plot(CZ_Rhot, vrot)
            #   plt.errorbar(IDA_Rhot, ne, yerr=ne_unc, fmt='+')
            '''
    return (CZ_Rhot, Ti, err_Ti, vrot, err_vrot)



CXRS_Rhot=[]
CXRS_Ti = []
CXRS_err_Ti=[]
CXRS_vrot = []
CXRS_err_vrot=[]
for i,diag in enumerate(diags):
    try:
        CZ_Rhot, CZ_Ti, CZ_err_Ti, CZ_vrot, CZ_err_vrot = readCZ(diag)


        if diag in 'CEZ':
            CEZ_vrot_mean = np.nanmean(CZ_vrot)
            print (CEZ_vrot_mean)

        if diag=='COZ':

            COZ_vrot_mean = np.nanmean(CZ_vrot[12:])
            ind = np.invert(np.isnan(CZ_vrot))
            CZ_vrot[ind] = CZ_vrot[ind]+(CEZ_vrot_mean-COZ_vrot_mean)
            CZ_err_vrot[ind] = CZ_err_vrot[ind]+(CEZ_vrot_mean-COZ_vrot_mean)

            CXRS_Rhot= np.append(CXRS_Rhot, CZ_Rhot[12:])
            CXRS_Ti= np.append(CXRS_Ti, CZ_Ti[12:])
            CXRS_err_Ti= np.append(CXRS_err_Ti, CZ_err_Ti[12:])
            CXRS_vrot= np.append(CXRS_vrot, CZ_vrot[12:])
            CXRS_err_vrot= np.append(CXRS_err_vrot, CZ_err_vrot[12:])
            
            plt.subplot(121)
            plt.errorbar(CZ_Rhot[12:], CZ_Ti[12:], yerr=CZ_err_Ti[12:], fmt=marker[i])
            plt.subplot(122)
            plt.errorbar(CZ_Rhot[12:], CZ_vrot[12:], yerr=CZ_err_vrot[12:], fmt=marker[i])
            
        elif diag=='CUZ':

            CUZ_vrot_mean = np.nanmean(CZ_vrot[13:])
            ind = np.invert(np.isnan(CZ_vrot))
            CZ_vrot[ind] = CZ_vrot[ind]+(CEZ_vrot_mean-CUZ_vrot_mean)
            CZ_err_vrot[ind] = CZ_err_vrot[ind]+(CEZ_vrot_mean-CUZ_vrot_mean)

            CXRS_Rhot= np.append(CXRS_Rhot, CZ_Rhot[13:])
            CXRS_Ti= np.append(CXRS_Ti, CZ_Ti[13:])
            CXRS_err_Ti= np.append(CXRS_err_Ti, CZ_err_Ti[13:])
            CXRS_vrot= np.append(CXRS_vrot, CZ_vrot[13:])
            CXRS_err_vrot= np.append(CXRS_err_vrot, CZ_err_vrot[13:])
            
            plt.subplot(121)
            plt.errorbar(CZ_Rhot[13:], CZ_Ti[13:], yerr=CZ_err_Ti[13:], fmt=marker[i])
            plt.subplot(122)
            plt.errorbar(CZ_Rhot[13:], CZ_vrot[13:], yerr=CZ_err_vrot[13:], fmt=marker[i])
            
        elif diag=='CPZ':
            CXRS_Rhot= np.append(CXRS_Rhot, CZ_Rhot)
            CXRS_Ti= np.append(CXRS_Ti, CZ_Ti)
            CXRS_err_Ti= np.append(CXRS_err_Ti, CZ_err_Ti)
            CXRS_vrot= np.append(CXRS_vrot, np.nan*len(CZ_vrot))
            CXRS_err_vrot= np.append(CXRS_err_vrot, np.nan*len(CZ_err_vrot))
            
            plt.subplot(121)
            plt.errorbar(CZ_Rhot, CZ_Ti, yerr=CZ_err_Ti, fmt=marker[i])
            plt.subplot(122)
            plt.errorbar(CZ_Rhot, CZ_vrot, yerr=CZ_err_vrot, fmt=marker[i])

        else:
            CXRS_Rhot= np.append(CXRS_Rhot, CZ_Rhot)
            CXRS_Ti= np.append(CXRS_Ti, CZ_Ti)
            CXRS_err_Ti= np.append(CXRS_err_Ti, CZ_err_Ti)
            CXRS_vrot= np.append(CXRS_vrot, CZ_vrot)
            CXRS_err_vrot= np.append(CXRS_err_vrot, CZ_err_vrot)
            
            plt.subplot(121)
            plt.errorbar(CZ_Rhot, CZ_Ti, yerr=CZ_err_Ti, fmt=marker[i])
            plt.subplot(122)
            plt.errorbar(CZ_Rhot, CZ_vrot, yerr=CZ_err_vrot, fmt=marker[i])
           
    except:
        print('####\nSHOTFILE not available ---> %s\n####\n\n\n' %diag)


print (CXRS_Ti.shape)
print (CXRS_vrot.shape)

"""
#plt.show() 					# show plot
try:
    print (CXRS_Ti.shape)
    print (CXRS_vrot.shape)
except: 
    print ('\nThe CXRS is broken!!\n')
    #time=5.02
    if sf.Open('COZ', shot):
        try:
            CZ_Ti = sf.GetSignal('Ti_c')
            CZ_err_Ti = sf.GetSignal('err_Ti_c')
        except:
            CZ_Ti = sf.GetSignal('Ti')
            CZ_err_Ti = sf.GetSignal('err_Ti')
        CZ_vrot = sf.GetSignal('vrot')
        CZ_err_vrot = sf.GetSignal('err_vrot')
        try:
            CZ_time = sf.GetTimebase('time')
        except:
            CZ_time = sf.GetTimebase('T-'+diag)  

        CZ_mask = np.argmin(np.abs(CZ_time - time))

        CZ_R = sf.GetAreabase('R')
        CZ_z = sf.GetAreabase('z')  

        CZ_Ti = CZ_Ti[CZ_mask]
        CZ_err_Ti = CZ_err_Ti[CZ_mask]
        CZ_vrot = CZ_vrot[CZ_mask]
        CZ_err_vrot = CZ_err_vrot[CZ_mask]
        sf.Close()
        
        ind = (CZ_R > 0)
        CZ_R = CZ_R[ind]
        CZ_z = CZ_z[ind]
        CZ_Ti = CZ_Ti[ind]
        CZ_err_Ti = CZ_err_Ti[ind]
        CZ_vrot = CZ_vrot[ind]
        CZ_err_vrot = CZ_err_vrot[ind]

        ind = (CZ_Ti == 0)
      #  CZ_R = CZ_R[ind[0]]
      #  CZ_z = CZ_z[ind[0]]
        CZ_Ti[ind] = np.nan
        CZ_err_Ti[ind] = np.nan
        CZ_vrot[ind] = np.nan
        CZ_err_vrot[ind] = np.nan

        CZ_Rhot = np.squeeze(eqm.rz2rho(CZ_R, CZ_z, t_in=time, coord_out='rho_tor'))

        plt.subplot(223)
        plt.errorbar(CZ_Rhot[12:], CZ_Ti[12:], yerr=CZ_err_Ti[12:], fmt='b.')
        plt.subplot(224)
        plt.errorbar(CZ_Rhot[12:], CZ_vrot[12:], yerr=CZ_err_vrot[12:], fmt='b.')

        CXRS_Rhot= np.append(CXRS_Rhot, CZ_Rhot[12:])
        CXRS_Ti= np.append(CXRS_Ti, CZ_Ti[12:])
        CXRS_err_Ti= np.append(CXRS_err_Ti, CZ_err_Ti[12:])
        CXRS_vrot= np.append(CXRS_vrot, CZ_vrot[12:])
        CXRS_err_vrot= np.append(CXRS_err_vrot, CZ_err_vrot[12:])

       # plt.figure()
       # plt.plot(Ti)
       # plt.show()

    if sf.Open('CUZ', shot):
        try:
            CZ_Ti = sf.GetSignal('Ti_c')
            CZ_err_Ti = sf.GetSignal('err_Ti_c')
        except:
            CZ_Ti = sf.GetSignal('Ti')
            CZ_err_Ti = sf.GetSignal('err_Ti')
        CZ_vrot = sf.GetSignal('vrot')
        CZ_err_vrot = sf.GetSignal('err_vrot')
        try:
            CZ_time = sf.GetTimebase('time')
        except:
            CZ_time = sf.GetTimebase('T-'+diag)  

        CZ_mask = np.argmin(np.abs(CZ_time - time))

        CZ_R = sf.GetAreabase('R')
        CZ_z = sf.GetAreabase('z')  

        CZ_Ti = CZ_Ti[CZ_mask]
        CZ_err_Ti = CZ_err_Ti[CZ_mask]
        CZ_vrot = CZ_vrot[CZ_mask]
        CZ_err_vrot = CZ_err_vrot[CZ_mask]
        sf.Close()
        
        ind = (CZ_R > 0)
        CZ_R = CZ_R[ind]
        CZ_z = CZ_z[ind]
        CZ_Ti = CZ_Ti[ind]
        CZ_err_Ti = CZ_err_Ti[ind]
        CZ_vrot = CZ_vrot[ind]
        CZ_err_vrot = CZ_err_vrot[ind]

        ind = (CZ_Ti == 0)
      #  CZ_R = CZ_R[ind[0]]
      #  CZ_z = CZ_z[ind[0]]
        CZ_Ti[ind] = np.nan
        CZ_err_Ti[ind] = np.nan
        CZ_vrot[ind] = np.nan
        CZ_err_vrot[ind] = np.nan

        Ti  =  CZ_Ti
        err_Ti=CZ_err_Ti
        vrot  =  CZ_vrot
        err_vrot=CZ_err_vrot

        CZ_Rhot = np.squeeze(eqm.rz2rho(CZ_R, CZ_z, t_in=time, coord_out='rho_tor'))

        plt.subplot(223)
        plt.errorbar(CZ_Rhot[12:], CZ_Ti[12:], yerr=CZ_err_Ti[12:], fmt='g+')
        plt.subplot(224)
        plt.errorbar(CZ_Rhot[12:], CZ_vrot[12:], yerr=CZ_err_vrot[12:], fmt='g+')

        CXRS_Rhot= np.append(CXRS_Rhot, CZ_Rhot[12:])
        CXRS_Ti= np.append(CXRS_Ti, CZ_Ti[12:])
        CXRS_err_Ti= np.append(CXRS_err_Ti, CZ_err_Ti[12:])
        CXRS_vrot= np.append(CXRS_vrot, CZ_vrot[12:])
        CXRS_err_vrot= np.append(CXRS_err_vrot, CZ_err_vrot[12:])
    print ('\nThe CXRS is broken!!\n')
"""


CXRS_err_Ti = CXRS_err_Ti[np.isfinite(CXRS_Ti)]
Ti_Rhot     = CXRS_Rhot[np.isfinite(CXRS_Ti)]
CXRS_Ti     = CXRS_Ti[np.isfinite(CXRS_Ti)]
Ti_Rhot     = CXRS_Rhot[np.isfinite(CXRS_err_Ti)]
CXRS_Ti     = CXRS_Ti[np.isfinite(CXRS_err_Ti)]
CXRS_err_Ti = CXRS_err_Ti[np.isfinite(CXRS_err_Ti)]

CXRS_Ti = np.array([x for _,x in sorted(zip(Ti_Rhot,CXRS_Ti))])
CXRS_err_Ti = np.array([x for _,x in sorted(zip(Ti_Rhot,CXRS_err_Ti))])
Ti_Rhot = np.array(sorted(Ti_Rhot))


CXRS_err_vrot = CXRS_err_vrot[np.isfinite(CXRS_vrot)]
vrot_Rhot     = CXRS_Rhot[np.isfinite(CXRS_vrot)]
CXRS_vrot     = CXRS_vrot[np.isfinite(CXRS_vrot)]
vrot_Rhot     = CXRS_Rhot[np.isfinite(CXRS_err_vrot)]
CXRS_vrot     = CXRS_vrot[np.isfinite(CXRS_err_vrot)]
CXRS_err_vrot = CXRS_err_vrot[np.isfinite(CXRS_err_vrot)]

CXRS_vrot = np.array([x for _,x in sorted(zip(vrot_Rhot,CXRS_vrot))])
CXRS_err_vrot = np.array([x for _,x in sorted(zip(vrot_Rhot,CXRS_err_vrot))])
vrot_Rhot = np.array(sorted(vrot_Rhot))

print (CXRS_Ti.shape)
print (CXRS_vrot.shape)


############################################### Fitting
#    Gaussian Processes Fit by Aaron Ho @DIFFER

if GPfit==True:
    print ('\nGaussian Processes Fitting\n')

    # Define kernels
    xn = np.linspace(0.0,1.0,101)
    IG = GPR1D.IG_WarpingFunction(3.0e-1, 3.0e-1, 1.0e-1, 0.97e0, 0.7)
    kernel = GPR1D.Gibbs_Kernel(2.0e0,wfunc=IG)
    kbounds = np.atleast_2d([[2.0e-1,7.0e-2,7.0e-2],[1.0e0,3.0e-1,2.0e-1]])

    nkernel = GPR1D.Sum_Kernel(GPR1D.RQ_Kernel(1.0e0,1.0e-1,1.0e0),GPR1D.Noise_Kernel(1.0e-2))
   # nkernel = GPR1D.RQ_Kernel(1.0e0,1.0e-1,1.0e0)#,GPR1D.Noise_Kernel(1.0e-2))
    nkbounds = np.atleast_2d([[1.0e-1,1.0e-2,1.0e0],[1.0e1,1.0e0,1.0e1]])

    # GPR fit accounting only for y-errors
    # Define fit settings
    GPR = GPR1D.GPR1D()
    GPR.set_kernel(kernel=kernel,kbounds=kbounds,regpar=1)#<<====== MODIFY THIS PARAMETER TO CHANGE SMOOTHNESS
    #GPR.set_error_kernel(kernel=nkernel,kbounds=nkbounds,regpar=10.0)
    GPR.set_error_kernel(kernel=nkernel,kbounds=nkbounds,regpar=3.0)
    GPR.set_raw_data(xdata=Ti_Rhot, ydata=CXRS_Ti, yerr=CXRS_err_Ti,dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
    # xerr=xe, 
    GPR.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-2,0.4,0.8])
    #GPR.set_search_parameters(epsilon=None)
    #GPR.set_error_search_parameters(epsilon=1.0e-1)
    GPR.set_error_search_parameters(epsilon=None)
    # Uncomment these to test recommended optimizers
    #GPR.set_search_parameters(epsilon=1.0e-3,method='adadelta',spars=[1.0e-2,0.5])
    #GPR.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])

    # Do fit
    GPR.GPRFit(xn,nrestarts=5)
    # Grab outputs
    #(gpname,gppars) = GPR.get_gp_kernel_details()
    #(egpname,egppars) = GPR.get_gp_error_kernel_details()
    (vgp,egp,vdgp,edgp) = GPR.get_gp_results()
    Ti_new = vgp

    plt.subplot(121)
    psig = 1.0
    plt.plot(xn,vgp,color='r')
    yl = vgp - psig * egp
    yu = vgp + psig * egp
    plt.fill_between(xn,yl,yu,facecolor='r',edgecolor='None',alpha=0.1)


    GPR.set_raw_data(xdata=vrot_Rhot, ydata=CXRS_vrot, yerr=CXRS_err_vrot,dxdata=None,dydata=None,dyerr=None)
    # Do fit
    GPR.GPRFit(xn,nrestarts=5)
    # Grab outputs
    #(gpname,gppars) = GPR.get_gp_kernel_details()
    #(egpname,egppars) = GPR.get_gp_error_kernel_details()
    (vgp,egp,vdgp,edgp) = GPR.get_gp_results()
    vrot_new = vgp

    plt.subplot(122)
    psig = 1.0
    plt.plot(xn,vgp,color='r')
    yl = vgp - psig * egp
    yu = vgp + psig * egp
    plt.fill_between(xn,yl,yu,facecolor='r',edgecolor='None',alpha=0.1)

    xnew = xn

    #plt.subplot(121)
    #plt.plot(Ti_Rhot, CXRS_Ti, 'k', alpha=0.2)
    #plt.fill_between(CXRS_Rhot, CXRS_Ti-CXRS_err_Ti, CXRS_Ti+CXRS_err_Ti, alpha=0.2)

    #plt.subplot(122)
    #plt.plot(vrot_Rhot, CXRS_vrot, 'k', alpha=0.2)
    #plt.fill_between(CXRS_Rhot, CXRS_vrot-CXRS_err_vrot, CXRS_vrot+CXRS_err_vrot, alpha=0.2)

    #plt.show() 					# show plot



else:

    ###------------Flip
    rhot_n = np.transpose(Ti_Rhot[::-1])
    rhot_n = -1.*rhot_n
    Ti_n = CXRS_Ti[::-1]
    rhot_f = np.append([rhot_n], [Ti_Rhot])
    Ti_f = np.append([Ti_n], [CXRS_Ti])

    ###------------Interpolate
    order=3                                                   # 2 for quadratic, 3 for cubic spline
    #weights=np.ones(len(Ti.data[jt, ind]))
    xnew = np.linspace(0., 1.0, num=101, endpoint=True)
    #s = UnivariateSpline(rhot, Ti.data[idx, ind], k=order, s=1000000)
    #Ti_f = Ti_f[~np.isnan(Ti_f)]
    #rhot_f = rhot_f[~np.isnan(Ti_f)] ##Avoid NaN
    s = UnivariateSpline(rhot_f, Ti_f, k=order, s=100000)     # Decrease 's' to obtain a less smoothed fit
    ynew = s(xnew)
    #s2 = InterpolatedUnivariateSpline(rhop, Ti.data[jt, ind], k=order)
    #ynew2 = s2(xnew)
    plt.subplot(223)
    plt.plot(xnew,ynew,'r')
    Ti_new = ynew
    ###

    ###------------Flip
    rhot_n = np.transpose(vrot_Rhot[::-1])
    rhot_n = -1.*rhot_n
    vrot_n = CXRS_vrot[::-1]
    rhot_f = np.append([rhot_n], [vrot_Rhot])
    vrot_f = np.append([vrot_n], [CXRS_vrot])

    plt.subplot(224)
    #plt.title('%s @ t=%3.2fs'%(vrot.description, time))
    #plt.ylabel(vrot.unit)
    plt.xlabel('rho_tor')
    #plt.grid('on')
    ###------------Interpolate
    #order=3
    #s = UnivariateSpline(rhot, vrot.data[idx, ind], k=order, s=1000000000)
    #index = np.isnan(vrot_t)
    #vrot_f = vrot_f[~np.isnan(vrot_f)]
    #rhot_f = rhot_f[~np.isnan(vrot_f)] ##Avoid NaN
    s = UnivariateSpline(rhot_f, vrot_f, k=order, s=50000000)
    ynew = s(xnew)
    plt.subplot(224)
    plt.plot(xnew,ynew,'r')
    vrot_new = ynew
    ###


plt.subplot(121)
plt.title('Ti')
plt.ylabel('eV')
plt.xlabel('rho_tor')
#plt.xlim(0,1.1)
#plt.ylim(0,np.nanmax(Ti_new)*1.1)
#plt.grid('on')

plt.subplot(122)
plt.title('rotation')
plt.ylabel('m/s')
plt.xlabel('rho_tor')
#plt.xlim(0,1.1)
#plt.ylim(0,np.nanmax(vrot_new)*1.1)
#plt.grid('on')

plt.suptitle('SHOT %s @ %s s'%(shot, time))
plt.show() 					# show plot


#--------------------------------------------------------------> write ASTRA file
#if ida_flag!=True:
print ('\n~~~~~~~~~~~~~')
yrn = (raw_input('%s@%s: DO YOU WANT TO STORE PROFILES? (y/n): ' %(str(shot), str(time))))    #python 2
#yrn = (input('DO YOU WANT TO STORE PROFILES? (y/n): '))        #python 3
#else:
#	yrn='y'

if yrn=='y' and ASTRA_flag==True:
    exp_dir = cwd+"/exp_ASTRA"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print ('\nCreated directory: %s\n' %exp_dir)

    fig.savefig(exp_dir+"/overview_#"+str(shot)+"_"+str(time)+".png", dpi=100)
	
    file=open(exp_dir+"/"+str(shot)+"_"+str(time),"w")
    file.write("aug -aug- #%s t=%s\n" % (shot, time))
    file.write(" Param |Time   |  Value  |  Error\n|------|-------|---------|-------\n")
    file.write("NA1             101\n")
    file.write("AB              0.622\n")
    file.write("AWALL           0.648\n")
    file.write("RTOR            1.650\n")
    file.write("ELONM           1.579\n")
    file.write("TRICH           0.199\n")
    file.write("ZMJ             1.000\n")
    file.write("AMJ             2.000\n")
#    file.write("RTOR    %.2f    %.5s\n" %(time, RTOR))
    file.write("ABC     %.2f    %.5s\n" %(time, ABC))
    file.write("ELONG   %.2f    %.5s\n" %(time, ELONG))
    file.write("TRIAN   %.2f    %.5s\n" %(time, TRIAN))
    file.write("SHIFT   %.2f    %.5s\n" %(time, SHIFT))
    file.write("UPDWN   %.2f    %.5s\n" %(time, UPDWN))
    file.write("BTOR    %.2f    %.5s\n" %(time, BTOR))
    file.write("IPL     %.2f    %.5s\n" %(time, IPL))
    file.write("ZRD1    %.2f    %.5s\n" %(time, PNI1))
    file.write("ZRD2    %.2f    %.5s\n" %(time, PNI2))
    file.write("ZRD3    %.2f    %.5s\n" %(time, PNI3))
    file.write("ZRD4    %.2f    %.5s\n" %(time, PNI4))
    file.write("ZRD5    %.2f    %.5s\n" %(time, PNI5))
    file.write("ZRD6    %.2f    %.5s\n" %(time, PNI6))
    file.write("ZRD7    %.2f    %.5s\n" %(time, PNI7))
    file.write("ZRD8    %.2f    %.5s\n" %(time, PNI8))
    file.write("ZRD11   %.2f    %.5s\n" %(time, PG1))
    file.write("ZRD12   %.2f    %.5s\n" %(time, PG2))
    file.write("ZRD13   %.2f    %.5s\n" %(time, PG3))
    file.write("ZRD14   %.2f    %.5s\n" %(time, PG4))
    file.write("ZRD15   %.2f    %.5s\n" %(time, PG1N))
    file.write("ZRD16   %.2f    %.5s\n" %(time, PG2N))
    file.write("ZRD17   %.2f    %.5s\n" %(time, PG3N))
    file.write("ZRD18   %.2f    %.5s\n" %(time, PG4N))
    file.write("ZRD20   %.2f    %.5s\n" %(time, PICRH))
    '''
    file.write("ZRD15   %.2f    %.5s\n" %(time, PECRH))
    file.write("ZRD16   %.2f    0.100\n" %(time))
    file.write("ZRD17   %.2f    0.020\n" %(time))
    file.write("ZRD25   %.2f    0.000\n" %(time))
    file.write("ZRD26   %.2f    0.100\n" %(time))
    file.write("ZRD27   %.2f    0.050\n" %(time))
    '''
    try:
        file.write("ZRD50   %.2f    %.5s\n" %(time_e, c_W))
	if np.isnan(c_W_l):
        	file.write("ZRD51   %.2f    %.5s\n" %(time_e, '0.00'))
	else:
        	file.write("ZRD51   %.2f    %.5s\n" %(time_e, c_W_l))

    except:
        print('\n No Tungsten Concentration \n')

    info=list(Te_new/1.e03)            #--------> Te <--------#
    radius=list(xnew)
    gridtipe=12
    points=len(Te_new)
    file.write("\n!AUG #%s\n" % shot)
    file.write("POINTS %s     GRIDTYPE %s     NAMEXP  TEX     NTIMES 1   \n" % (points, gridtipe) )

    file.write("%.4E\n" % time)       # time
    ii=0                              # radius
    for item in radius:
      file.write("    %.4E" % item)   
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n")
    ii=0                              # Te
    for item in info:
      file.write("    %.4E" % item)  
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n\n")

    info=list(Ti_new/1.e03)            #--------> Ti <--------#
    radius=list(xnew)
    #gridtipe=12
    points=len(Ti_new)
    file.write("!AUG #%s\n" % shot)
    file.write("POINTS %s     GRIDTYPE %s     NAMEXP  TIX     NTIMES 1   \n" % (points, gridtipe) )

    file.write("%.4E\n" % time)       # time
    ii=0                              # radius
    for item in radius:
      file.write("    %.4E" % item)   
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n")
    ii=0                              # Ti
    for item in info:
      file.write("    %.4E" % item)  
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n\n")

    info=list(vrot_new)                #-------> vrot <-------#
    radius=list(xnew)
    #gridtipe=12
    points=len(vrot_new)
    file.write("!AUG #%s\n" % shot)
    file.write("POINTS %s     GRIDTYPE %s     NAMEXP  VTORX     NTIMES 1   \n" % (points, gridtipe) )

    file.write("%.4E\n" % time)       # time
    ii=0                              # radius
    for item in radius:
      file.write("    %.4E" % item)   
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n")
    ii=0                              # vtor
    for item in info:
      file.write("    %.4E" % item)  
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n\n")

    info=list(ne_new/1.e19)            #--------> ne <--------#
    radius=list(xnew)
    #gridtipe=12
    points=len(ne_new)
    file.write("!AUG #%s\n" % shot)
    file.write("POINTS %s     GRIDTYPE %s     NAMEXP  NEX     NTIMES 1   \n" % (points, gridtipe) )

    file.write("%.4E\n" % time)       # time
    ii=0                              # radius
    for item in radius:
      file.write("    %.4E" % item)   
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n")
    ii=0                              # ne
    for item in info:
      file.write("    %.4E" % item)  
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n\n")

   #Zeff = 1.2680                      #-------> Zeff <-------#
    file.write("!AUG #%s\n" % shot)
    file.write("POINTS 2     GRIDTYPE %s     NAMEXP  ZEFX     NTIMES 1   \n" %gridtipe )
    file.write("%.5E\n" % time_e)       # time
    file.write("    0.0000e+00    1.0000e+00\n")
    file.write("    %.4E    %.4E\n\n" %(Zeff, Zeff))

    info=mu                            #--------> Mu <--------#
    radius=rho_u
    time_q=tarr_q[indq]
    #gridtipe=12
    points=len(mu)
    file.write("!AUG #%s\n" % shot)
    file.write("POINTS %s     GRIDTYPE %s     NAMEXP  MUX     NTIMES 1   \n" % (points, gridtipe) )

    file.write("%.4E\n" % time_q)       # time
    ii=0                              # radius
    for item in radius:
      file.write("    %.4E" % item)   
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n")
    ii=0                              # Mu
    for item in info:
      file.write("    %.4E" % item)  
      ii+=1
      if ii==6:
          file.write("\n")
          ii=0
    file.write("\n\n")

    points = len(rscat)                #------> BOUNDARY <------#
    #points = np.array(rscat[indb])
    #points =  points.shape
    time_b = tarr[indb]
    zeta = zscat
    radius = rscat
    file.write("!AUG #%s\n" % shot)
    file.write("POINTS %s     NAMEXP  BND     NTIMES 1   \n" %points )
    file.write("%.5E\n" % time_b)       # time
    for r,z in zip(np.nditer(radius),np.nditer(zeta)):
        file.write("  %.5E  %.5E\n" %(r,z))
    file.write("\n\n")


    file.close()
    print ('ASTRA File written:   '+str(shot)+'_'+str(time))
    
    os.system("cp %s/%d_%s \
	%s/astra7/exp/%d_%s" %(exp_dir, shot, time, cwd, shot, time))


elif yrn=='y' and ASTRA_flag==False:
	# Here you can customize how to store the profiles you want
	'''
	file=open('AUG_#'+str(shot)+'_@'+str(time)+'_ne.txt','w')
	file.write('AUG  #%s t=%s --> rhot ne\n\n' % (shot, time))
	for i,rhooo in enumerate(xnew):
	    file.write('  %.5E  %.5E\n' %(xnew[i], ne_new[i]))
	file.close()
	print ('File written:   AUG_#'+str(shot)+'_@'+str(time)+'_ne.txt')

	file=open('AUG_#'+str(shot)+'_@'+str(time)+'_te.txt','w')
	file.write('AUG  #%s t=%s --> Rhot te err_te\n\n' % (shot, time))
	for i,rhooo in enumerate(xnew):
	    file.write('  %.5E  %.5E\n' %(xnew[i], Te_new[i]))
	file.close()
	print ('File written:   AUG_#'+str(shot)+'_@'+str(time)+'_te.txt')

	file=open('AUG_#'+str(shot)+'_@'+str(time)+'_vrot.txt','w')
	file.write('AUG  #%s t=%s --> rhot ne\n\n' % (shot, time))
	for i,rhooo in enumerate(xnew):
	    file.write('  %.5E  %.5E\n' %(xnew[i], vrot_new[i]))
	file.close()
	print ('File written:   AUG_#'+str(shot)+'_@'+str(time)+'_vrot.txt')
	'''

	prof_dir = cwd+"/PROFILES_dir"
	if not os.path.exists(prof_dir):
		os.makedirs(prof_dir)
		print ('\nCreated directory: %s\n' %prof_dir)


	fig.savefig(prof_dir+"/Overview_AUG_#"+str(shot)+"_@"+str(time)+".png", dpi=100)

	file=open(prof_dir+'/Profiles_AUG_#'+str(shot)+'_@'+str(time)+'.txt','w')
	file.write('AUG  #%s t=%s --> rhot Ti vrot\n\n' % (shot, time))
	for i,rhooo in enumerate(xnew):
	    file.write('  %.5E  %.5E  %.5E  %.5E  %.5E\n' %(xnew[i], Ti_new[i], vrot_new[i]))
	file.close()
	print ('File written:   '+prof_dir+'/Profiles_AUG_#'+str(shot)+'_@'+str(time)+'.txt')


else:
    print ('...profiles NOT written\n')
