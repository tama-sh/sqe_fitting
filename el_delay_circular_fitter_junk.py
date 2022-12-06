#!/usr/bin/env python
# -*- coding: utf-8 -*-

## ref
# https://sites.astro.caltech.edu/~jonas/Theses/Jiansong_Gao_08.pdf
# https://people.cas.uab.edu/~mosya/cl/cl1.pdf

import numpy as np
from labrad.units import WithUnit, Unit
from .base import fitter_base
from scipy.optimize import newton, minimize
from .util import est_el_delay, sorted_partial_average

class el_delay_circular_fitter(fitter_base):
    def __init__(self, period=2*np.pi):
        super().__init__()
        ## init param name
        self.p_dict = [('electrica_delay', Unit('yy/xx'))]
        
    def p_initializer(self):
        x = self.x
        y = self.y
        
        el_delay = est_el_delay(x, y)
        p_init = [el_delay]
        return p_init 
    
    def fit(self, p_init=None, ftol=1.49012e-08, maxfev=0):
        if p_init is not None:
            if not len(p_init) == len(self.p_dict):
                raise Exception('initial paramter length is not matached')
        else:
            p_init = self.p_initializer()
                      
        p_dict = self.get_p_dict_subs()
        self.p_init = []
        for idx in range(len(p_init)):
            if isinstance(p_init[idx], WithUnit):
                if p_init[idx].isCompatible(p_dict[idx][1]):
                    self.p_init.append(p_init[idx][p_dict[idx][1]])
                else:
                    raise Exception('parameter {0} should have compatible unit with {1}'.format(p_dict[idx][0], p_dict[idx][1]))
            else:
                self.p_init.append(p_init[idx])
        self.p_init = np.array(self.p_init)

        self.fit_res = minimize(self.err_func, self.p_init, args=(self.x, self.y))
        self.p_opt = self.fit_res['x']
        
        return self.p_opt
        
        
    def err_func(self, p, x, y):
        yn = y*np.exp(-1j*p[0]*x)
        yr = yn.real
        yi = yn.imag
        w = np.sqrt(yr**2 + yi**2)
        Mww = np.sum(w**2)
        Mxw = np.sum(yr*w)
        Myw = np.sum(yi*w)

        Mxx = np.sum(yr**2)
        Myy = np.sum(yi**2)
        Mxy = np.sum(yr*yi)

        Mw = np.sum(w)
        Mx = np.sum(yr)
        My = np.sum(yi)
        
        n = len(y)
        
        M_mat = np.matrix([
            [Mww, Mxw, Myw, Mw],
            [Mxw, Mxx, Mxy, Mx],
            [Myw, Mxy, Myy, My],
            [Mw, Mx, My, n]
        ])
        
        B_mat = np.matrix([
            [0, 0, 0, -2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-2, 0, 0, 0]
        ])
        
        def Q(eta):
            return np.linalg.det(M_mat-eta*B_mat)
        
        eta = newton(Q, 0)
       
        eigval, eigvec = np.linalg.eig(M_mat - eta*B_mat)
        
        idx = np.argmin(np.abs(eigval))
        opt_vec = np.array(eigvec[idx]).flatten()
        A = opt_vec[0]
        B = opt_vec[1]
        C = opt_vec[2]
        D = opt_vec[3]
        
        A_mat = np.matrix([[A, B, C, D]]).T
        
        self.eta = eta
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
        return (A_mat.T*M_mat*A_mat - eta*(A_mat.T*B_mat*A_mat-1))[0,0]