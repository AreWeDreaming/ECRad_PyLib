'''
Created on Nov 15, 2019

@author: sdenk
'''
import numpy as np

def read_xi_astra(filename):
    data = np.loadtxt(filename, skiprows=2, usecols=(0,1,5,6,7,8,9,11,12,13,15,16,17,18,32,33,40,41, \
                                                     42,43,44,46,47,48,49,50,51,52,53,56,57,61,62,63,64,65,66,67,68,69,70,71))
#     rho= data[:,0]
#     aa = data[:,1]
#     ne = data[:,2]
#     nex= data[:,3]
#     Te = data[:,4]
#     Tex= data[:,5]
#     ni = data[:,6]
#     Ti = data[:,7]
#     Tix= data[:,8]
#     vtor=data[:,9]
#     Xe = data[:,10]
#     Xi = data[:,11]
#     Xix= data[:,12]
    Xex= data[:,13]
#     SN= data[:,14]
#     SNN= data[:,15]
#     Mu = data[:,15]
#     Cbs= data[:,16]
#     BPF= data[:,17]
    FP = data[:,18]
#     BP = data[:,19]
#     CU = data[:,20]
#     Ptot=data[:,21]
#     Cn = data[:,22]
#     Ce = data[:,23]
#     Ci = data[:,24]
#     Car17=data[:,25]
#     Car18=data[:,26]
#     Samul=data[:,27]
#     NNCL= data[:,28]
#     GASv= data[:,29]
#     Xped= data[:,30]
#     dP  = data[:,31]
#     Dn  = data[:,32]
#     Chi = data[:,33]
#     CF3 = data[:,34]
#     Er = data[:,35]
#     Dneo = data[:,36]
#     Xtgi = data[:,37]
#     Xtge = data[:,38]
#     Xneoe = data[:,39]
#     Xneoi = data[:,40]
#     alfs = data[:,41]
    rhop=(((FP-FP[0])/(FP[-1]-FP[0]))**0.5)
    return rhop, Xex
        
        
        
        
        