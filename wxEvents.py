'''
Created on Aug 29, 2016

@author: sdenk
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:51:28 2014

@author: sdenk
"""
import wx
Unbound_EVT_NEW_STATUS = wx.NewEventType()  # Call This
EVT_NEW_STATUS = wx.PyEventBinder(Unbound_EVT_NEW_STATUS, 1)  # Bind that

Unbound_EVT_UPDATE = wx.NewEventType()  # Call This
EVT_UPDATE = wx.PyEventBinder(Unbound_EVT_UPDATE, 1)  # Bind that

Unbound_EVT_NEXT_TIME_STEP = wx.NewEventType()  # Call This
EVT_NEXT_TIME_STEP = wx.PyEventBinder(Unbound_EVT_NEXT_TIME_STEP, 1)  # Bind that

Unbound_EVT_FIT_FINISHED = wx.NewEventType()  # Call This
EVT_FIT_FINISHED = wx.PyEventBinder(Unbound_EVT_FIT_FINISHED, 1)  # Bind that

Unbound_EVT_UPDATE_DATA = wx.NewEventType()  # Call This
EVT_UPDATE_DATA = wx.PyEventBinder(Unbound_EVT_UPDATE_DATA , 1)

Unbound_EVT_IDA_DATA_READY = wx.NewEventType()  # Call This
EVT_IDA_DATA_READY = wx.PyEventBinder(Unbound_EVT_IDA_DATA_READY , 1)

Unbound_EVT_LOAD_CONFIG = wx.NewEventType()  # Call This
EVT_LOAD_CONFIG = wx.PyEventBinder(Unbound_EVT_LOAD_CONFIG, 1)  # Bind that

Unbound_EVT_UPDATE_CONFIG = wx.NewEventType()  # Call This
EVT_UPDATE_CONFIG = wx.PyEventBinder(Unbound_EVT_UPDATE_CONFIG , 1)

Unbound_EVT_LOCK_EXPORT = wx.NewEventType()  # Call This
EVT_LOCK_EXPORT = wx.PyEventBinder(Unbound_EVT_LOCK_EXPORT , 1)

Unbound_EVT_REPLOT = wx.NewEventType()  # Call This
EVT_REPLOT = wx.PyEventBinder(Unbound_EVT_REPLOT , 1)

Unbound_EVT_LOCK = wx.NewEventType()  # Call This
EVT_LOCK = wx.PyEventBinder(Unbound_EVT_LOCK , 1)

Unbound_EVT_LOAD_MAT = wx.NewEventType()  # Call This
EVT_LOAD_MAT = wx.PyEventBinder(Unbound_EVT_LOAD_MAT, 1)  # Bind that

Unbound_EVT_MAKE_DPLOT = wx.NewEventType()  # Call This
EVT_MAKE_DPLOT = wx.PyEventBinder(Unbound_EVT_MAKE_DPLOT, 1)  # Bind that

Unbound_EVT_DONE_PLOTTING = wx.NewEventType()  # Call This
EVT_DONE_PLOTTING = wx.PyEventBinder(Unbound_EVT_DONE_PLOTTING, 1)  # Bind that

Unbound_EVT_RESIZE = wx.NewEventType()  # Call This
EVT_RESIZE = wx.PyEventBinder(Unbound_EVT_RESIZE, 1)  # Bind that

Unbound_EVT_THREAD_FINISHED = wx.NewEventType()  # Call This
EVT_THREAD_FINISHED = wx.PyEventBinder(Unbound_EVT_THREAD_FINISHED, 1)  # Bind that

Unbound_EVT_AUG_DATA_READ = wx.NewEventType()  # Call This
EVT_AUG_DATA_READ = wx.PyEventBinder(Unbound_EVT_AUG_DATA_READ, 1)

Unbound_EVT_CPO_DATA_WRITTEN = wx.NewEventType()  # Call This
EVT_CPO_DATA_WRITTEN = wx.PyEventBinder(Unbound_EVT_CPO_DATA_WRITTEN, 1)

Unbound_EVT_UNLOCK = wx.NewEventType()  # Call This
EVT_UNLOCK = wx.PyEventBinder(Unbound_EVT_UNLOCK, 1)

class UpdateConfigEvt(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

class UpdateDataEvt(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.data = None
        self.Config = None
        self.Results = None

    def SetData(self, data):
        self.data = data

    def SetResults(self, Results):
        self.Results = Results

    def SetConfig(self, Config):
        self.Config = Config

class NewStatusEvt(wx.PyCommandEvent):  # For the statusbar
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.Status = None

    def SetStatus(self, Status):
        self.Status = Status

class DonePlottingEvt(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

    def SetFig(self, fig):
        self.fig = fig

class ThreadFinishedEvt(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)
    def SetSuccess(self, success):
        self.success = success

class LockExportEvt(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

class UnlockEvt(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

class LoadMatEvt(wx.PyCommandEvent):
    def __init(self, evtType, id):
        self.filename = None
        wx.PyCommandEvent.__init__(self, evtType, id)
    def SetFilename(self, filename):
        self.filename = filename

class LoadedConfigSEvt(wx.PyCommandEvent):
    def __init(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

class LoadConfigEvt(wx.PyCommandEvent):
    def __init(self, evtType, id):
        self.Filename = None
        wx.PyCommandEvent.__init__(self, evtType, id)
    def SetFilename(self, filename):
        self.Filename = filename

