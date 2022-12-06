#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from labrad.units import Unit
from .base import fitter_base

class mod_linear_fitter(fitter_base):
    def __init__(self, period=2*np.pi):
        super().__init__()
        ## init param name
        self.p_dict = [('electrica_delay', Unit('yy/xx')), ('phase_offset', Unit('yy'))]
        
        if np.__version__ < '1.21.0' and (not period == 2*np.pi):
            raise ValueError("Arbitrary period option is only supported in numpy >= 1.21.0")
        else:
            self.period = period
        
    def p_initializer(self):
        if np.__version__ < '1.21.0':
            if not self.period == 2*np.pi:
                raise ValueError("Arbitrary period option is only supported in numpy >= 1.21.0")
            else:
                y_unwrap = np.unwrap(self.y)
        else:
            y_unwrap = np.unwrap(self.y, period=self.period)
        p_init = np.polyfit(self.x, y_unwrap, 1)
        return p_init

    def fit_func(self, p, x):
        return p[0]*x + p[1]
    
    def err_func(self, p, x, y):
        return (y - self.fit_func(p, x) + 0.5*self.period)%(self.period) - 0.5*self.period