#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from labrad.units import Unit
from .base import fitter_base, cplx_loader

def decay_func(p, x):
    return p[0]*np.exp(-x / p[1]) + p[2]

class decay_fitter(fitter_base, cplx_loader):
    def __init__(self):
        super().__init__()
        ## init param name
        self.p_dict = [('amp', Unit('yy')), ('decay_time', Unit('xx')), ('amp_offset', Unit('yy'))]

    def p_initializer(self):
        N = len(self.x)
        offset_init = self.y[-1]
        amp_init = self.y[0] - self.y[-1]
        y_n = self.y - offset_init
        
        if amp_init > 0:
            tau_init = self.x[np.argmax(y_n < 0.5*y_n[0])]/np.log(2)
        else:
            tau_init = self.x[np.argmax(y_n > 0.5*y_n[0])]/np.log(2)

        p_init = [amp_init, tau_init, offset_init]
        return p_init

    def fit_func(self, p, x):
        return decay_func(p, x)