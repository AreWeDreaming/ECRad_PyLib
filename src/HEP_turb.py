'''
Created on Apr 26, 2020

@author: sdenk
'''
from plotting_configuration import plt
from shotfile_handling_AUG import get_HEP_ne
from scipy import fft, ifft
import numpy as np

def plot_ne_trace(shot, rhop):
    time, rhop_base, ne = get_HEP_ne(shot)
    time = np.array(time)
    ich = int(np.mean(np.argmin(np.abs(rhop_base-rhop),axis=1)))
    t_split = np.argmax(time[1:] - time[:-1])
    time_1 = time[:t_split+1]
    dt1 = (time_1[-1] - time_1[0]) / len(time_1)
    time_2 = time[t_split+1:]
    dt2 = (time_2[-1] - time_2[0]) / len(time_2)
    print(dt1, dt2)
    f1 = np.linspace(0.0, 1.0 / (2.0*dt1), len(time_1))
    f2 = np.linspace(0.0, 1.0 / (2.0*dt2), len(time_2))
    plt.plot(f1 / 1.e3, np.log(np.abs(fft(np.nan_to_num(ne.T[ich].T[:t_split+1]/1.e19)))), "-")
    plt.plot(f2 / 1.e3, np.log(np.abs(fft(np.nan_to_num(ne.T[ich].T[t_split+1:]/1.e19)))), "--")
    plt.show()
    
if(__name__ == "__main__"):
    plot_ne_trace(35662, rhop=0.96)