'''
Created on Nov 13, 2019

@author: sdenk
'''
import sys
import numpy as np
sys.path.append('/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib')
import dd

def crash(shot=35662, exp="AUGD", ed=1):
    IDA = dd.shotfile("IDA", pulseNumber=int(shot), experiment=exp, edition=ed)
    IDA_Te_mat = IDA.getSignalGroup("Te", dtype=np.double)
    print("Loading Te worked but the following line will not!")
    IDA_ECE_dat = IDA.getSignalGroup("ece_dat", dtype=np.double)

if(__name__ == "__main__"):
    crash()