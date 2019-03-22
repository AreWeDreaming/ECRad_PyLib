'''
Created on Jun 4, 2018

@author: sdenk
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from plotting_configuration import *

def get_min_distance(x, y, px, py):
    f = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    s = np.linspace(0.0, 1.0, len(x))
    f_spl = InterpolatedUnivariateSpline(s, f, k=3)
    s_high = np.linspace(0.0, 1.0, 500)
    df = f_spl(s_high, 1)
    df_spl = InterpolatedUnivariateSpline(s_high, df)
#    df_spl = f_spl.derivative(1)
    roots = df_spl.roots()
    print(np.min(f_spl(roots)))
    print(np.min(np.sqrt((x - px) ** 2 + (y - py) ** 2)))
    plt.plot(s, f, "^")
    plt.plot(s_high, f_spl(s_high), "-")
    plt.plot(roots, f_spl(roots), "+")
    plt.show()

if(__name__ == "__main__"):
    vessel_data = np.loadtxt("ASDEX_Upgrade_vessel.txt", skiprows=1)
    get_min_distance(vessel_data.T[0], vessel_data.T[1], 3.25, 0.05)
