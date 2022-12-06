#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def sorted_ratio_idx(data, r):
    """
    Extract the inddx of data which is len(data)*r th largest in data
    The value is same as median when r = 0.5
        r: ratio of idx
    """
    l = len(data)
    sort_idxs = np.argsort(data)
    return sort_idxs[int(np.round(l*r))]

def sorted_partial_idxs(data, r_s, r_e):
    """
        r_s: ratio of data extracted from
        r_e: ratio of data extracted to 
    """
    l = len(data)
    sort_idxs = np.argsort(data)
    idx_s = int(np.round(l*r_s))
    idx_e = int(np.round(l*r_e))
    
    idx_range = np.arange(idx_s, idx_e, 1)
    return sort_idxs[idx_range]

def sorted_partial_average(data, r_s, r_e):
    l = len(data)
    sorted_data = np.sort(data)
    idx_s = int(np.round(l*r_s))
    idx_e = int(np.round(l*r_e))
    return sorted_data[idx_s:idx_e].mean()

def est_el_delay(freq, cplx):
    # calc group delay 
    del_f = freq[1] - freq[0]
    delay = ((cplx[1:]-cplx[:-1])/(0.5*(cplx[1:]+cplx[:-1]))).imag/del_f
    
    # estimate electric delay by using center half of data
    el_delay = sorted_partial_average(delay, 0.25, 0.75)

    # calibrate more with phase unwrapping
    cplx_n = cplx*np.exp(-1j*el_delay*freq)
    phase = np.unwrap(np.angle(cplx_n))
    el_delay_cor = (phase[-1]-phase[0])/(freq[-1]-freq[0])
    
    return el_delay+el_delay_cor

def correct_el_delay(freq, cplx, el_delay=None):
    if el_delay is None:
        el_delay = est_el_delay(freq, cplx)
    return cplx*np.exp(-1j*el_delay*freq)
