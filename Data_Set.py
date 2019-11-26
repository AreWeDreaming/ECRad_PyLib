'''
Created on Nov 6, 2019

@author: sdenk
'''
import numpy as np
class DataSet():
    '''
    classdocs
    '''


    def __init__(self, name, type, time_window, measurements, uncertainties, positions=None):
        '''
        Constructor
        '''
        # If data is two dimensional then first dimension has to be time!
        # time window of the measurements, (time_beg, time_end)
        self.time_window = time_window
        self.measurements = np.array(measurements) # measured values as 1D array
        self.uncertainties = np.array(uncertainties)
        # corresponding absolute uncertainty as 1D array
        # for now, no distinction between statistical and systematic uncertainty
        if(np.count_nonzero(self.uncertainties.flatten()) != len(self.uncertainties.flatten())):
            raise ValueError("Zero values in data set uncertainty encountered")
        if(self.measurements.shape != self.uncertainties.shape):
            raise AttributeError("Length of measurement {0:d} not equal to length of uncertainty {1:d}".format(len(measurements), len(uncertainties)))
        # Optional, since it does not make sense for some diagnostics, e.g. Interferometry
        self.positions = positions
        self.name = name
        self.type = type
    
    def is_ready(self):
        # At the moment no further steps needed
        return True
    
    
    
    
    