'''
Created on Jan 29, 2017

@author: sdenk
'''
import numpy as np
from equilibrium_utils import EQDataExt

def get_current(nshot, tshot, npts):
    raise(ValueError('Routine get_current not available for TCV'))
# Input



def make_rhop_signed_axis(shot, time, R, rhop, f, f2=None, eq_exp='AUGD', eq_diag='EQH', eq_ed=0, external_folder=''):
    eq_obj = EQData(shot, external_folder=external_folder, eq_exp=eq_exp, eq_diag=eq_diag, eq_ed=eq_ed)
    R_ax, z_ax = eq_obj.get_axis(time)
    HFS = R < R_ax
    last_HFS = HFS[0]
    profile_cnt = 0
    rhop_out = []
    rhop_out.append([])
    f_out = []
    f_out.append([])
    if(f2 is not None):
        f2_out = []
        f2_out.append([])
    for i in range(len(HFS)):
        if(last_HFS != HFS[i]):
            last_HFS = HFS[i]
            rhop_out.append([])
            f_out.append([])
            if(f2 is not None):
                f2_out.append([])
            profile_cnt += 1
        if(HFS[i]):
            rhop_out[profile_cnt].append(-rhop[i])
        else:
            rhop_out[profile_cnt].append(rhop[i])
        f_out[profile_cnt].append(f[i])
        if(f2 is not None):
            f2_out[profile_cnt].append(f2[i])
    for profile_cnt in range(len(rhop_out)):
        rhop_out[profile_cnt] = np.array(rhop_out[profile_cnt])
        f_out[profile_cnt] = np.array(f_out[profile_cnt])
        if(f2 is not None):
            f2_out[profile_cnt] = np.array(f2_out[profile_cnt])
    if(f2 is not None):
        return rhop_out, f_out, f2_out, R_ax
    else:
        return rhop_out, f_out, R_ax

def make_B_min(shot, time, rhop_vec, exp="AUGD", diag="EQH", ed=0):
    raise(ValueError('Routine get_current not available for TCV'))

class EQData(EQDataExt):
    def __init__(self, shot, external_folder='', EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005, Ext_data=False):
        EQDataExt.__init__(self, shot, external_folder='', EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction=1.005, Ext_data=False)

    def init_read_from_shotfile(self, EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0):
        raise(ValueError('Loading equilibrium data from shot file not possible for TCV'))
        self.shotfile_ready = True

    def GetSlice(self, time):
        raise(ValueError('Loading equilibrium data from shot file not possible for TCV'))
        if(not self.shotfile_ready):
            self.init_read_from_shotfile(self, self.EQ_exp, self.EQ_diag, self.EQ_ed)
