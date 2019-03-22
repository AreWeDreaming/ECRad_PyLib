'''
Created on Apr 11, 2016

@author: sdenk
'''
from shotfile_handling import get_diag_data_no_calib
import numpy as np
from Diags import Diag

def look_for_shots():
    shots_with_data = []
    CTA = Diag('CTA', 'AUGD', 'CTA', 0)
    for i in range(32812, 32975):
        try:
            time, signal = get_diag_data_no_calib(CTA, i, single_channel=40)
            if(test_signal(time, signal)):
                print('Shot ', i, " useful")
                shots_with_data.append(i)
        except:
            pass
    np.savetxt('/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/CTA_shots.txt', np.array(shots_with_data, dtype=np.int), fmt='%d')

def refine_list():
    shots_with_data = np.loadtxt('/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/CTA_shots.txt', dtype=np.int)
    CTA = Diag('CTA', 'AUGD', 'CTA', 0)
    shots_with_relevant_data = []
    for i in shots_with_data:
        try:
            time, signal = get_diag_data_no_calib(CTA, i, single_channel=22)
            if(test_signal(time, signal)):
                print('Shot ', i, " useful")
                shots_with_relevant_data.append(i)
        except:
            pass
    np.savetxt('/afs/ipp-garching.mpg.de/home/s/sdenk/Documentation/Data/CTA_shots.txt', np.array(shots_with_relevant_data, dtype=np.int))

def test():
    CTA = Diag('CTA', 'AUGD', 'CTA', 0)
    time, signal = get_diag_data_no_calib(CTA, 32964 , single_channel=40)
    print(test_signal(time, signal))

def test_signal(time, signal):
    max_index = np.argmin(np.abs(time - 0.03))
    # print(max_index, len(signal[:max_index]), time[max_index], (np.max(signal) - np.min(signal)), (np.max(signal[0:max_index]) - np.min(signal[0:max_index])) * 10.0)
    return (np.max(signal) - np.min(signal)) > (np.max(signal[0:max_index]) - np.min(signal[0:max_index])) * 10.0

look_for_shots()
# test()
# refine_list()
