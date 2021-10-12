
from Equilibrium_Utils_AUG import EQData
from Shotfile_Handling_AUG import load_IDA_data
from ece_optics_2019 import get_ECE_launch_v2
from ECRH_Launcher import ECRHLauncher
import numpy as np

def run_ECE_TORBEAM_AUG(shot, time, frequencies, launch_override=None, EQ_exp="AUGD", EQ_diag="EQH", EQ_ed=0, bt_vac_correction = 1.005, IDA_exp="AUGD", IDA_ed=0):
    IDA_dict = load_IDA_data(shot, [time], IDA_exp, IDA_ed)
    EQ_obj = EQData(shot, EQ_exp=EQ_exp, EQ_diag=EQ_diag, EQ_ed=EQ_ed)
    eq_slice = EQ_obj.GetSlice(time, bt_vac_correction=bt_vac_correction)
    ece_launch = get_ECE_launch_v2(8, "CECE", 0.055, frequencies, np.zeros(len(frequencies)))
    gy = ECRHLauncher()