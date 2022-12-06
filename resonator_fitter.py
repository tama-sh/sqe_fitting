#!/usr/bin/env python
# -*- coding: utf-8 -*-

# obsoleted on 2022/01/21
# replaced with resonator_fitter_derivative_based.py

import numpy as np
from labrad.units import Unit
from .base import fitter_base, cplx_loader
from scipy import signal


def s11_func_w_tilt(freq, freq_c, k_ex, k_in, alpha, theta, tau, phi):
    return alpha*np.exp(1j*(theta-freq*tau))*(1-(2*k_ex*np.exp(1j*phi))/(k_ex+k_in-2j*(freq-freq_c)))

def s11_func(freq, freq_c, k_ex, k_in, alpha, theta, tau):
    return alpha*np.exp(1j*(theta-freq*tau))*(1-(2*k_ex)/(k_ex+k_in-2j*(freq-freq_c)))

def s11_err_func(p, x, y):
    y_fit = s11_func(x, *p)
    diff = y - y_fit
    err = np.absolute(diff)
    return err

def s11_err_func_w_tilt(p, x, y):
    y_fit = s11_func_w_tilt(x, *p)
    diff = y - y_fit
    err = np.absolute(diff)
    return err

def est_linewidth(freq, data):
    ptp = np.ptp(data)
    min_val = np.min(data)
    idxs = np.where(data < min_val + ptp/2)[0]
    return (freq[idxs[-1]] - freq[idxs[0]])

def est_phase_linewidth(freq, phase, idx_c=None):
    if idx_c == None:
        idx_c = np.argmin(abs(phase- (np.max(phase)+np.min(phase))/2))
    x = np.unwrap(phase)
    idx_p = np.where(abs(x[idx_c+1:] - x[idx_c])>np.pi/4)[0]
    idx_n = np.where(abs(x[:idx_c] - x[idx_c])>np.pi/4)[0]
    if len(idx_p) > 0 and len(idx_n) > 0:
        return (freq[idx_p[0]+idx_c+1] - freq[idx_n[-1]])
    else:
        return (freq[-1] - freq[0])

class resonator_fitter(fitter_base):
    def __init__(self, with_tilt=True):
        super().__init__() # for python3
        #super(resonator_fitter, self).__init__() # for python2
        
        ## init param name
        if with_tilt:
            self.p_dict = [('freq_center', Unit('xx')),
                           ('k_ex', Unit('xx')),
                           ('k_in', Unit('xx')),
                           ('amplitude', Unit('yy')),
                           ('phase', Unit('rad')),
                           ('electrical_delay', Unit('rad/xx')),
                           ('tilt_angle', Unit('rad'))
                          ]
            self.fit_func = lambda p, x: s11_func_w_tilt(x, *p)
        else:
            self.p_dict = [('freq_center', Unit('xx')),
                           ('k_ex', Unit('xx')),
                           ('k_in', Unit('xx')),
                           ('amplitude', Unit('yy')),
                           ('phase', Unit('rad')),
                           ('electrical_delay', Unit('rad/xx'))
                          ]
            self.fit_func = lambda p, x: s11_func(x, *p)
        self.with_tilt = with_tilt
        self.overcoupled = True

    def set_overcoupled(overcoupled=True):
        self.overcoupled = overcoupled
        
    def err_func(self, p, x, y):
        y_fit = self.fit_func(p, x)
        diff = y - y_fit
        err = np.absolute(diff)
        return err

    def p_initializer(self):
        freq = self.x
        cplx = self.y
        amp = np.abs(cplx)
        phase = np.unwrap(np.angle(cplx))
        
        numtap = 5
        if numtap%2 == 0:
            freq_lp = (freq[int(numtap/2)-1:-int(numtap/2)] + freq[int(numtap/2):int(-numtap/2)+1])/2
        else:
            freq_lp = freq[int(numtap/2):int(-numtap/2)]
        
        if self.overcoupled:         
            alpha_init = np.max(amp)
            tau_init = -((phase[-1] - phase[0]) - 2*np.pi)/(freq[-1] - freq[0])
            
            cplx_n = cplx*np.exp(1j*tau_init*freq)
            cplx_lp = np.convolve(cplx_n, signal.firwin(numtap, 0.1), mode="valid")
            diff_amp = np.abs(cplx_lp[1:] - cplx_lp[:-1])
            diff_freq = (freq_lp[1:] + freq_lp[:-1])/2
            
            #theta_init = -1*(phase[np.argmin(amp)] + tau_init*freq[np.argmin(amp)])
            #freq_c_init = freq[np.argmin(amp)] # MHz
            freq_c_init = diff_freq[np.argmax(diff_amp)]
            k_ex_init = est_phase_linewidth(freq, phase + tau_init*freq, idx_c=np.argmax(diff_amp))
            #k_ex_init = est_phase_linewidth(freq, phase + tau_init*freq)
            k_in_init = k_ex_init * (np.ptp(amp)/alpha_init)
        else:
            alpha_init = np.max(amp)
            tau_init = -((phase[-1] - phase[0]))/(freq[-1] - freq[0])
            
            cplx_n = cplx*np.exp(1j*tau_init*freq)
            cplx_lp = np.convolve(cplx_n, signal.firwin(numtap, 0.1), mode="valid")
            diff_amp = np.abs(cplx_lp[1:] - cplx_lp[:-1])
            diff_freq = (freq_lp[1:] + freq_lp[:-1])/2
            #theta_init = phase[np.argmin(amp)] + tau_init*freq[np.argmin(amp)]
            #freq_c_init = freq[np.argmin(amp)] # MHz
            
            freq_c_init = diff_freq[np.argmax(diff_amp)]
            k_in_init = est_linewidth(freq, amp**2)
            k_ex_init = 2*k_in_init * (np.ptp(amp)/alpha_init)

        if self.with_tilt:
            theta_init = phase[0] - np.angle(self.fit_func([freq_c_init, k_ex_init, k_in_init, alpha_init, 0, tau_init, 0], freq[0]))
            p_init = [freq_c_init, k_ex_init, k_in_init, alpha_init, theta_init, tau_init, 0]
        else:
            theta_init = phase[0] - np.angle(self.fit_func([freq_c_init, k_ex_init, k_in_init, alpha_init, 0, tau_init], freq[0]))
            p_init = [freq_c_init, k_ex_init, k_in_init, alpha_init, theta_init, tau_init]

        return p_init

    def fit_twice(self, p_init=None, ftol=1.49012e-08, maxfev=0):
        self.fit(p_init=p_init, ftol=ftol, maxfev=maxfev)
        fit_res1 = self.fit_res
        p_opt1 = self.p_opt
        p_cov_raw1 = self.p_cov_raw
        p_cov1 = self.p_cov
        p_stdev1 = self.p_stdev
        
        self.overcoupled = not self.overcoupled
        self.fit(p_init=p_init, ftol=ftol, maxfev=maxfev)
        fit_res2 = self.fit_res
        self.overcoupled = not self.overcoupled
    
        fval1 = np.linalg.norm(fit_res1[2]['fvec'])
        fval2 = np.linalg.norm(fit_res2[2]['fvec'])
    
        if fval1 < fval2:
            self.fit_res = fit_res1
            self.p_opt = p_opt1
            self.p_cov_raw = p_cov_raw1
            self.p_cov = p_cov1
            self.p_stdev = p_stdev1
        return self.p_opt, self.p_cov
    
