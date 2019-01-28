"""
Created on Wed Aug  20 17:06:00 2014

This is just a helping unit to collect all functions that I need to analyse
NEST-networks

@author: Martin Pyka
"""


import nest

import io
import csv
from matplotlib import pyplot
import numpy as np


def getEventsFromSpikeDetector(spike_detector, t_range=[0, float("inf")]):
    """ Returns senders and times data from a spike-detector. This is for 
    example helpful if you want to write your own plotter for the spikes
    or analyse those data further """
    status = nest.GetStatus(spike_detector)
    times =  status[0]['events']['times']
    sender = status[0]['events']['senders'][(times >= t_range[0]) & (times < t_range[1])]
    times = times[(times >= t_range[0]) & (times < t_range[1])]
    return sender, times


def scatter(spike_detector, area=3):
    sender, times = getEventsFromSpikeDetector(spike_detector)
    pyplot.scatter(times, sender, s=area )
    
    
def getPOA(ng, sd, interval=[0, float("inf")]):
    """ returns the percentage of neurons for a given neuron group that were
    active for a given time frame
    ng        neurongroup which was observed by the spike detector
    sd        spike detector
    interval  start and end of the interval for which the percentage should be
              determined
    """
    # get sender and time points
    sender, times = getEventsFromSpikeDetector(sd)
    
    # get number of active unique neurons for this particular time interval
    n_a = len(np.unique(sender[(times >= interval[0]) & (times < interval[1])]))
    
    return float(n_a) / float(len(ng))
    
def getPrePostNgs(m, ngs, c):
    ''' Returns pre- and post-neurongroup for a given connection
    index '''
    pre_ngs = ngs[m['connections'][0][c][1]]
    post_ngs = ngs[m['connections'][0][c][2]]
    return pre_ngs, post_ngs

def getConnInfo(m, ngs, c, info):
    ''' returns the weights as they are returned by GetConnections 
    for a given connection-index c '''
    pre_ngs, post_ngs = getPrePostNgs(m, ngs, c)
    
    connections = nest.GetConnections(pre_ngs, post_ngs)
    result = nest.GetStatus(connections, info)
    return result
    
def exportSpikeDetectorData(filename, ngs, sd_list):
    """ Exports data from a list of spike detector ids into a given file
    in CSV-format """
    sender = np.array((), dtype=int)
    times = np.array(())
    indices = np.array((), dtype = int)
    
    for index, sd in enumerate(sd_list):
        s, t = getEventsFromSpikeDetector(sd)
        i = np.ones(len(s), dtype=int) * index
        
        sender = np.concatenate((sender, s))
        times = np.concatenate((times, t))
        indices = np.concatenate((indices, i))
    
    f = open(filename + '.csv', 'w')
    writer = csv.writer(
        f,
        delimiter=";",
        quoting=csv.QUOTE_NONNUMERIC
    )
    
    # compute permutation for sorting of times
    perm = sorted(range(len(times)), key=lambda k: times[k])

    for i in perm:
        writer.writerow([sender[i] - ngs[indices[i]][0], indices[i], times[i]])    
        
    f.close()        
        
        