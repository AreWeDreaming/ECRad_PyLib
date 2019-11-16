'''
Created on Nov 4, 2019

@author: sdenk
'''
from shotfile_handling_AUG import load_IDI_data
from equilibrium_utils_AUG import EQData
from plotting_configuration import plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def make_Ti_u_file( filename, shot, times=None, diag="IDI", exp="AUGD", ed=0, \
                    eq_exp="AUGD", eq_diag="EQH", eq_ed=0, review =False):
    comments = [";-SHOT",";-SHOT DATE-  UFILES ASCII FILE SYSTEM", ";-NUMBER OF ASSOCIATED SCALAR QUANTITIES-", \
                ";-INDEPENDENT VARIABLE LABEL: X-", ";-INDEPENDENT VARIABLE LABEL: Y-", \
                ";-DEPENDENT VARIABLE LABEL-", ";-PROC CODE- 0:RAW 1:AVG 2:SM. 3:AVG+SM", \
                ";-# OF X PTS-", ";-# OF Y PTS-",  ";----END-OF-DATA-----------------COMMENTS:-----------"]
    IDI_dict = load_IDI_data(shot, timepoints=times, exp=exp, ed=ed)
    EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
    IDI_dict["rhot_prof"] = [np.linspace(0.0, 1.0, 100)]
    IDI_dict["Ti_rhot"] = []
    reviewed_time = []
#     fig = plt.figure(figsize=(12.5, 8.5))
#     ax = fig.add_subplot(111)
    plt.ion()
    for itime, time in enumerate(IDI_dict["time"]):
        rhop = EQ_obj.rhot_to_rhop(time, IDI_dict["rhot_prof"][0])
        Ti_cur = IDI_dict["Ti"][itime]
        Ti_cur[Ti_cur < 0.0] = 20.e-3 # room temperature
        Ti_cur = np.log(Ti_cur)
        Ti_spline = InterpolatedUnivariateSpline(IDI_dict["rhop_prof"][itime], Ti_cur)
        Ti_mapped = np.exp(Ti_spline(rhop))
        if(review):
            plt.gca().clear()
            plt.plot(IDI_dict["rhot_prof"][0], Ti_mapped / 1.e3)
            plt.draw()
            plt.pause(0.5)
            print("Time is {0:2.3f} s".format(time))
            a = raw_input("Press enter to use this time point!\n Enter any key before pressing enter to ignore it.")
            if(len(a) == 0):
                IDI_dict["Ti_rhot"].append(Ti_mapped)
                reviewed_time.append(time)
        else:
            IDI_dict["Ti_rhot"].append(Ti_mapped)
            reviewed_time.append(time)
    IDI_dict["time"] = np.array(reviewed_time)
    n_time = len(IDI_dict["time"])
    n_prof = len(IDI_dict["rhot_prof"][0])
    IDI_dict["Ti_rhot"] = np.array(IDI_dict["Ti_rhot"])
#        
    ufile = open(filename, "w")
    ufile.write("  {0:30s}".format(str(shot) + exp + " 2 0 6") + comments[0] + "\n")
    ufile.write("  {0:30s}".format("") + comments[1] + "\n")
    ufile.write("  {0:30s}".format("0") + comments[2] + "\n")
    ufile.write(" {0:30s}".format("Time                Seconds ") + comments[3] + "\n")
    ufile.write(" {0:30s}".format("rho_tor") + comments[4] + "\n")
    ufile.write(" {0:30s}".format("TI                  eV") + comments[5] + "\n")
    ufile.write(" {0:30s}".format("0") + comments[6] + "\n")
    ufile.write(" {0:30s}".format("{0: 10d}".format(n_time)) + comments[7] + "\n")
    ufile.write(" {0:30s}".format("{0: 10d}".format(n_prof)) + comments[8] + "\n")
    ncol = 6
    format_str = "{0: 1.6e}"
    IDI_dict["Ti_rhot"] = IDI_dict["Ti_rhot"].T.flatten()
    IDI_dict["rhot_prof"] = IDI_dict["rhot_prof"][0]
    for key in ["time", "rhot_prof", "Ti_rhot"]:
        i = 0
        N = len(IDI_dict[key])
        while i < N:
            ufile.write(format_str.format(IDI_dict[key][i]))
            i += 1
            if(i %  ncol == 0):
                ufile.write("\n")
        if(i % ncol != 0):
            ufile.write("\n")
    ufile.write(comments[9] + "\n")
    ufile.close()
    
if(__name__=="__main__"):
    make_Ti_u_file("/afs/ipp/u/sdenk/Documentation/Data/u35662Ti.dat", 35662, review=True)
    
    