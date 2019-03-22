'''
Created on Apr 3, 2018

@author: sdenk
'''
import os
from shutil import move, copyfile
exchange_folder = "/afs/eufus.eu/g2itmdev/user/g2sdenk/nssf_exchange/"
target_folder = "/tokp/work/sdenk/nssf"

def nssf_exchange(shot, time, ida_log_ed):
    shot_in_folder = os.path.join(exchange_folder, str(shot), "{0:1.2f}".format(time))
    ed_out = 1
    shot_out_folder = os.path.join(target_folder, str(shot), "{0:1.2f}".format(time), "OERT")
    if(not os.path.isdir(shot_out_folder)):
        print("Target " + shot_out_folder + " must exist!")
        return
    ed_out_path = os.path.join(shot_out_folder, "ed_{0:d}".format(ed_out))
    while os.path.isdir(ed_out_path):
        ed_out += 1
        ed_out_path = os.path.join(shot_out_folder, "ed_{0:d}".format(ed_out))
    ed_in = 0
    ed_in_path = os.path.join(shot_in_folder, "ed_{0:d}".format(ed_in))
    ed_ida_log_path = os.path.join(target_folder, str(shot), "{0:1.2f}".format(time), "OERT", "ed_{0:d}".format(ida_log_ed))
    while os.path.isdir(ed_in_path):
        print("Processing: ", ed_in_path, ed_out_path)
        move(ed_in_path, ed_out_path)
        copyfile(os.path.join(ed_ida_log_path, "ida.log"), os.path.join(ed_out_path, "ida.log"))
        ed_out += 1
        ed_out_path = os.path.join(shot_out_folder, "ed_{0:d}".format(ed_out))
        ed_in += 1
        ed_in_path = os.path.join(shot_in_folder, "ed_{0:d}".format(ed_in))

if(__name__ == "__main__"):
#    nssf_exchange(34663, 3.60, 96)
#    nssf_exchange(33705, 4.90, 78)
    nssf_exchange(33697, 4.80, 135)

