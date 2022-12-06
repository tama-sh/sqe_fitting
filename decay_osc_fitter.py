#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from labrad.units import Unit
from .base import fitter_base, cplx_loader

def decay_osc_func(p, x):
    return p[0]*np.exp(-x / p[1])*np.cos(2*np.pi*p[2]*x + p[3]) + p[4]

class decay_osc_fitter(fitter_base, cplx_loader):
    def __init__(self, auto_flip=True):
        super().__init__()
        self.auto_flip = auto_flip
        self.flipped = False
        ## init param name
        self.p_dict = [('amp', Unit('yy')), ('decay_time', Unit('xx')), ('freq', Unit('xx^-1')), ('theta_offset', Unit('rad')), ('amp_offset', Unit('yy'))]

    def p_initializer(self):
        N = len(self.x)
        offset_init = np.mean(self.y)
        y_n = self.y - offset_init
        fft_data = np.fft.fft(y_n)
        freq_idx = np.argmax(np.abs(fft_data[:int(N/2)]))
        fft_peak = fft_data[freq_idx]
        T = self.x[-1] - self.x[0]
        del_freq = 1/T

        theta_init = np.angle(fft_peak)
        freq_init = del_freq*freq_idx

        vmax1 = np.max(np.abs(y_n[:int(N/2)]))
        vmax2 = np.max(np.abs(y_n[int(N/2):]))
        amp_init = np.max(abs(y_n))

        if vmax1 < vmax2:
            tau_init = 10*T
        else:
            tau_init = 0.5*T/np.log(vmax1/vmax2)   ## another possible way is fit fft signal with lorentian
        p_init = [amp_init, tau_init, freq_init, theta_init, offset_init]
        return p_init

    def fit_func(self, p, x):
        return decay_osc_func(p, x)
    
    def post_process(self):
        self.flipped = False
        amp_opt = self.p_opt[0]
        if amp_opt < 0:
            self.p_opt[0] = -amp_opt
            self.p_opt[3] = self.p_opt[3] - np.pi
        freq_opt = self.p_opt[2]
        if freq_opt < 0:
            self.p_opt[2] = -freq_opt
            self.p_opt[3] = -self.p_opt[3]
        
        theta_opt = self.p_opt[3]%(2*np.pi)  # wrap in [0, 2*pi)
        offset_opt = self.p_opt[4]
        if self.auto_flip and theta_opt > 0.5*np.pi and theta_opt < 1.5*np.pi:
            self.flipped = True
            if hasattr(self, 'cplx_n'):
                self.cplx_n = -self.cplx_n
            if hasattr(self, 'proj_angle'):
                self.proj_angle = (self.proj_angle+np.pi)%(2*np.pi)
            self.y = -self.y    
            theta_opt = theta_opt - np.pi
            offset_opt = -offset_opt
        else:
            theta_opt = (theta_opt + np.pi)%(2*np.pi) - np.pi  # wrap in [-pi, pi)
        self.p_opt[3] = theta_opt
        self.p_opt[4] = offset_opt