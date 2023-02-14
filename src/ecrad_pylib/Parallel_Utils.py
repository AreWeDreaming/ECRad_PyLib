# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:31:50 2014

@author: Severin Denk
"""
from threading import Thread
class WorkerThread(Thread):
    """Worker Thread Class."""
    def __init__(self, function, args, kwargs=None):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self._want_abort = 0
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start()

    def run(self):
        """Run Worker Thread."""
        if(self.kwargs is None):
            self.function(self.args)
        else:
            self.function(self.args, self.kwargs)
            


