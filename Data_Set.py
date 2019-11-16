'''
Created on Nov 6, 2019

@author: sdenk
'''

class DataSet():
    '''
    classdocs
    '''


    def __init__(self, name, type, time_window, measurements, uncertainties, positions=None):
        '''
        Constructor
        '''
        import numpy as np
        # time window of the measurements, (time_beg, time_end)
        self.time_window = time_window 
        # measured vales as (1D array)
        self.measurements = measurements # measured values as 1D array
        # corresponding absolute uncertainty as 1D array
        # for now, no distinction between statistical and systematic uncertainty
        if(np.count_nonzero(uncertainties) != len(uncertainties)):
            raise ValueError("Zero values in data set uncertainty encountered")
        if(len(measurements) != len(measurements)):
            raise AttributeError("Length of measurement {0:d} not equal to length of uncertainty {1:d}".format(len(measurements), len(uncertainties)))
        self.uncertainties = uncertainties 
        # Optional, since it does not make sense for some diagnostics, e.g. Interferometry
        # 1 D array
        self.positions = positions
        self.name = name
        self.type = type
    
    def is_ready(self):
        # At the moment no further steps needed
        return True
    
    
    
    
    