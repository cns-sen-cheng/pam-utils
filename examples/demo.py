# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:32:35 2014

@author: martin
"""


import matplotlib.pyplot as mp
import matplotlib

from nest import *
import nest.voltage_trace
import nest.raster_plot
import numpy as np

import pamutils.pam2nest as pam2nest
import pamutils.nest_vis as nest_vis


EXPORT_PATH = './'
DELAY_FACTOR = 4.36

m = []

def analyseNetwork():
    nest.ResetKernel()
    np.random.seed(1)
    global m
    m = pam2nest.import_connections(EXPORT_PATH + 'demo.zip')
    
    nest_vis.printNeuronGroups(m)
    nest_vis.printConnections(m)
    
    w_means = [10.]
    w_sds = [1.]
    d_means = [DELAY_FACTOR]
    d_sds = [1.]
    ngs = pam2nest.CreateNetwork(m, 'iaf_psc_delta',
                                 w_means, w_sds,
                                 d_means, d_sds)
    
    len(ngs)
    
    noise         = Create("poisson_generator", 50)
    dc_1            = nest.Create('dc_generator')
    
    voltmeter       = Create("voltmeter", 2)
    espikes         = Create("spike_detector")
    
    SetStatus(noise, [{'start': 0., 'stop': 10., 'rate': 100.0}])
    SetStatus(dc_1, {'start': 10., 'stop': 10.5, 'amplitude': 100.})

    Connect(noise, ngs[1][:50], conn_spec='one_to_one' , syn_spec = {'model': 'static_synapse', 'weight': 2000., 'delay': 1.})

    ConvergentConnect(ngs[0] + ngs[1],espikes)
    
    Simulate(1000.0)
        
    nest.raster_plot.from_device(espikes, hist=False)
    nest.raster_plot.show()    

   
if __name__ == "__main__":
    analyseNetwork()

    
