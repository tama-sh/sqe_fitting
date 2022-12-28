#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from labrad.units import Unit
from .base import fitter_base, cplx_loader
from .lorentzian_fitter import lorentzian_fitter
from .mod_linear_fitter import mod_linear_fitter
from .util import est_el_delay, correct_el_delay, sorted_partial_average, sorted_partial_idxs
from scipy.signal import firwin

def s11_func(freq, freq_c, k_ex, k_in, alpha, theta, tau, phi):
    return alpha*np.exp(1j*(theta-freq*tau))*(1-(2*k_ex*np.exp(1j*phi))/(k_ex+k_in-2j*(freq-freq_c)))

# def s11_func(freq, freq_c, k_ex, k_in, alpha, theta, tau):
#     return alpha*np.exp(1j*(theta-freq*tau))*(1-(2*k_ex)/(k_ex+k_in-2j*(freq-freq_c)))

def s21_func(freq, freq_c, k_ex, k_in, alpha, theta, tau, phi):
    return alpha*np.exp(1j*(theta-freq*tau))*(1-(k_ex*np.exp(1j*phi))/(k_ex+k_in-2j*(freq-freq_c)))

# def s21_func(freq, freq_c, k_ex, k_in, alpha, theta, tau):
#     return alpha*np.exp(1j*(theta-freq*tau))*(1-(k_ex)/(k_ex+k_in-2j*(freq-freq_c)))

# def est_linewidth(freq, data):
#     ptp = np.ptp(data)
#     min_val = np.min(data)
#     idxs = np.where(data < min_val + ptp/2)[0]
#     return (freq[idxs[-1]] - freq[idxs[0]])

# def est_phase_linewidth(freq, phase, idx_c=None):
#     if idx_c == None:
#         idx_c = np.argmin(abs(phase- (np.max(phase)+np.min(phase))/2))
#     x = np.unwrap(phase)
#     idx_p = np.where(abs(x[idx_c+1:] - x[idx_c])>np.pi/4)[0]
#     idx_n = np.where(abs(x[:idx_c] - x[idx_c])>np.pi/4)[0]
#     if len(idx_p) > 0 and len(idx_n) > 0:
#         return (freq[idx_p[0]+idx_c+1] - freq[idx_n[-1]])
#     else:
#         return (freq[-1] - freq[0])

class resonator_fitter(fitter_base):
    def __init__(self, s_type="S11", with_tilt=True):
        super().__init__() # for python3
        #super(resonator_fitter, self).__init__() # for python2
        
        ## init param name
        self.p_dict = [('freq_center', Unit('xx')),
                       ('k_ex/2pi', Unit('xx')),
                       ('k_in/2pi', Unit('xx')),
                       ('amplitude', Unit('yy')),
                       ('phase', Unit('rad')),
                       ('electrical_delay', Unit('rad/xx'))]

        if s_type == "S11":
            if with_tilt:
                self.fit_func = lambda p, x: s11_func(x, *p)
                self.p_dict.append(('tilt_angle', Unit('rad')))
            else:
                self.fit_func = lambda p, x: s11_func(x, *p, 0)
        elif s_type == "S21":
            if with_tilt:
                self.fit_func = lambda p, x: s21_func(x, *p)
                self.p_dict.append(('tilt_angle', Unit('rad')))
            else:
                self.fit_func = lambda p, x: s21_func(x, *p, 0)
        else:
            raise ValueError("s_type should be either S11 or S21.")
                
        self.with_tilt = with_tilt
        
    def err_func(self, p, x, y):
        y_fit = self.fit_func(p, x)
        diff = y - y_fit
        err = np.absolute(diff)
        return err

    def p_initializer(self):
        freq = self.x
        cplx = self.y
        
        # Electrical delay fit
        phase_unwrap = np.unwrap(np.angle(cplx))
        fitter_el_delay = mod_linear_fitter()
        fitter_el_delay.load_data(freq, phase_unwrap)
        
        ### uc = under coupoing, oc = over coupling
        uc_el_delay, uc_phase_offset = np.polyfit(freq, phase_unwrap, 1)
        oc_el_delay = uc_el_delay - 2*np.pi/(freq[-1]-freq[0])
        oc_phase_offset = uc_phase_offset + np.pi*(freq[0]+freq[-1])/(freq[-1]-freq[0]) - np.pi
        
        fitter_el_delay.fit([uc_el_delay, uc_phase_offset])
        fit_el_delay = fitter_el_delay.fit_func(fitter_el_delay.p_opt, freq)
        uc_p_opt = fitter_el_delay.p_opt
        uc_err = np.sum(np.square(fitter_el_delay.fit_res[2]['fvec']))

        fitter_el_delay.fit([oc_el_delay, oc_phase_offset])
        fit_el_delay = fitter_el_delay.fit_func(fitter_el_delay.p_opt, freq)
        oc_p_opt = fitter_el_delay.p_opt
        oc_err = np.sum(np.square(fitter_el_delay.fit_res[2]['fvec']))
        
        uc_tau_init = -uc_p_opt[0]
        uc_theta_init = uc_p_opt[1]
        oc_tau_init = -oc_p_opt[0]
        oc_theta_init = oc_p_opt[1]
            
        # Lorentzian fit
        cplx_c = correct_el_delay(freq, cplx, oc_tau_init)
        
        numtap = 11
        if numtap%2 == 0:
            freq_lp = (freq[int(numtap/2)-1:-int(numtap/2)] + freq[int(numtap/2):int(-numtap/2)+1])/2
        else:
            freq_lp = freq[int(numtap/2):int(-numtap/2)]

        cplx_lp = np.convolve(cplx_c, firwin(numtap, 0.1), mode="valid")
        
        freq_diff = (freq_lp[1:]+freq_lp[:-1])/2
        del_f = freq[1] - freq[0]
        diff_cplx = cplx_lp[1:] - cplx_lp[:-1]
        v_cplx = np.abs(diff_cplx/del_f)

        fitter_lorentz = lorentzian_fitter()
        fitter_lorentz.load_data(freq_diff, v_cplx)
        fitter_lorentz.fit()

        p_lorentz_opt = fitter_lorentz.p_opt
        # end of Lorentzian fit
        
        alpha_init = sorted_partial_average(np.abs(cplx), 0.75, 1)  # avoid peak
        freq_c_init = p_lorentz_opt[0]
        k_total_init = p_lorentz_opt[1]
        k_ex_init = p_lorentz_opt[2]/alpha_init
        k_in_init = k_total_init-k_ex_init
    
        if self.with_tilt:
            p_init = [[freq_c_init, k_ex_init, k_in_init, alpha_init, uc_theta_init, uc_tau_init, 1e-10],
                      [freq_c_init, k_ex_init, k_in_init, alpha_init, oc_theta_init, oc_tau_init, 1e-10]]
        else:
            p_init = [[freq_c_init, k_ex_init, k_in_init, alpha_init, uc_theta_init, uc_tau_init],
                      [freq_c_init, k_ex_init, k_in_init, alpha_init, oc_theta_init, oc_tau_init]]
        return p_init
    
