#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from labrad.units import Unit
from .base import fitter_base, cplx_loader

def lorentzian_func(freq, freq_c, gamma, alpha, offset):
    return alpha/((freq-freq_c)**2 + (gamma/2.0)**2) + offset


def est_linewidth_peak_tip(freq, data, n=2):
    """
        n: FWHM estimated from width of (1-1/n) values
    """
    ptp = np.ptp(data)
    
    # for dip
    if np.median(data) > (np.min(data) + ptp/2):
        idx_c = np.argmin(data)
        min_val = data[idx_c]
        cond = data < (min_val + ptp/n)
    else: # for peak
        idx_c = np.argmax(data)
        max_val = data[idx_c]
        cond = data > (max_val - ptp/n)
        
    l = len(data)
    i = 0
    while(idx_c+i+1<l and cond[idx_c+i+1]):
        i += 1
    j = 0
    while(idx_c-(j+1)>=0 and cond[idx_c-(j+1)]):
        j += 1
    idx_l = idx_c - j
    idx_r = idx_c + i
    
    if i==0 and j==0:
        r = (data[idx_c]-0.5*(data[idx_c+1]+data[idx_c-1]))/(ptp)
        return np.sqrt((1-r)/r)*2*(freq[1]-freq[0])
    else:
        return np.sqrt(n-1)*(freq[idx_r] - freq[idx_l])

class lorentzian_fitter(fitter_base, cplx_loader):
    def __init__(self):
        super().__init__()
        ## init param name
        self.p_dict = [('freq_center', Unit('xx')), ('decay_rate', Unit('xx')), ('amplitude', Unit('yy*xx^2')), ('offset', Unit('yy'))]

    def convert_dip_to_peak(self):
        if np.median(self.y) > (np.min(self.y) + np.ptp(self.y)/2):
            self.y = -self.y
        
    def p_initializer(self):
        freq_c_init = self.x[np.argmax(self.y)]
        gamma_init = est_linewidth_peak_tip(self.x, self.y, 2)
        alpha_init = np.ptp(self.y)*(gamma_init/2.)**2
        offset_init = np.min(self.y)
        
        p_init = [freq_c_init, gamma_init, alpha_init, offset_init]
        return p_init
        
    def fit_func(self, p, x):
        return lorentzian_func(x, *p)