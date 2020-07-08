'''
Created on Jun 19, 2019

@author: sdenk
'''
import numpy as np
from scipy.signal import medfilt
from scipy import fftpack
from Basic_Methods.Data_Fitting import make_fit
# Can be used to filter modes from data
# Useful to reduce systematic uncertainties in calibration of discharges with modes
def remove_mode(t, s, harmonics=None, mode_width=100.0, low_freq=100.0):
    # Fourier filters the strongest mode and its harmonics
    s_fft = fftpack.rfft(s)
    power = np.abs(s_fft)
    sample_freq = fftpack.fftfreq(s.size, d=t[1] - t[0])
    mask = sample_freq > low_freq
#    plt.figure(figsize=(6, 5))
#    plt.plot(sample_freq[mask], np.real(power)[mask])
#    plt.xlabel('Frequency [Hz]')
#    plt.ylabel('plower')
    p_est = np.zeros(3)
    i_max = np.argmax(power[mask])
    p_est[0] = power[mask][i_max]
    p_est[1] = sample_freq[mask][i_max]
    p_est[2] = mode_width
    try:
        p = make_fit("gauss", sample_freq[mask], power[mask], p_est=p_est)[0]
    except RuntimeError:
        print("Warning mode not filtered!")
        return s, 0.0, 0.0
    mode_height = np.abs(s_fft[mask][i_max]) / len(s_fft) * 4.0
    mode_phase = np.angle(s_fft[mask][i_max])
    f_center = p[1]
    f_width = p[2] * 2.0
    n_max = 1
    if(harmonics is not None):
        n_max = harmonics
    for n in range(1, n_max + 1):
        mode_filter = np.logical_and(np.abs(sample_freq) > n * (f_center - f_width), np.abs(sample_freq) < n * (f_center + f_width))
        s_fft[mode_filter] = 0.0
    filtered_sig = fftpack.irfft(s_fft)
#    plt.plot(sample_freq[mask], eval_func("gauss", sample_freq[mask], p), "--g")
#    plt.plot(t, s)
#    plt.plot(t, filtered_sig, "--")
#    plt.plot(t, np.mean(filtered_sig) - mode_height * np.cos(2.0 * np.pi * f_center * t + mode_phase))
#    plt.show()
    return filtered_sig, mode_height, mode_phase

def smooth(y_arr, median=False):
    if(median):
#        kernel_size = int(len(y_arr) / 10.e0)
#        if(kernel_size % 2 == 0):
#            kernel_size -= 1
        y_median = medfilt(y_arr)
        y_smooth = np.mean(y_median)
        std_dev = np.std(y_median, ddof=1)
    else:
        y_smooth = np.mean(y_arr)
        std_dev = np.std(y_arr, ddof=1)
    return y_smooth, std_dev