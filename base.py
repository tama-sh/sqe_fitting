#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from labrad.units import Unit, WithUnit, NumberDict

# unit
def subs_unit1(u_target, u_holder, u_real):
    """Unit substitution function
    
    Replace a temporary unit variable (xx, yy etc.) with a real unit.
       
    Args:
        u_target (Unit): target of unit substitution
        u_holder (Unit, str): a temporary unit variable to be replaced with a real unit "u_real"
        u_real (Unit, str): a reaul unit variable to be substituted
        
    Return:
        Unit: u_target with the substitution of u_holder -> u_real
        
    Example:
        u_target = "m*xx^2/s"
        u_holder = "xx"
        u_real = "s"
        -> Return "m*s"
    """
    if isinstance(u_holder, Unit):
        holder_name = u_holder.name
    elif isinstance(u_holder, str):
        holder_name = u_holder
    else:
        raise Exception('u_holder should be an instance of Unit or str')

    if holder_name not in u_target.lex_names:
        return u_target
    
    u_real = Unit(u_real)
    pow_factor = u_target.lex_names[holder_name]

    names = NumberDict(u_target.names)
    names.pop(holder_name)
    names = names + u_real.names*pow_factor
    factor = pow(u_real.factor, pow_factor)
    powers = [pt + pr*pow_factor for (pt, pr) in zip(u_target.powers, u_real.powers)]
    offset = u_target.offset
    lex_names = u_target.lex_names.copy()
    lex_names.pop(holder_name)
    
    return Unit(names, factor, powers, offset, lex_names)

def subs_unit(target, u_pair_list):
    """Multiple unit substitution function
    
    Replace temporary unit variables (xx, yy etc.) with real units.
    
    Args:
        target (Unit): target unit string
        u_pair_list (list[tuple[Unit, Unit]]): a list of (temparary unit, real unit) pairs
        
    Return:
        Unit: target with unit substitution
        
    Example:
        target = "m*s*xx/yy"
        u_pair_list = [("xx", "s"), ("yy", "m")]
        -> Return "s^2"
    """
    for (u_holder, u_real) in u_pair_list:
        target = subs_unit1(target, u_holder, u_real)
    return target

# projector   
def find_major_axis(cplx):
    """Find major axis from complex data
    
    Find major axis from complex data and return the angle indicating the direction of major axis
    
    Args:
        cplx (np.ndarray(complex)): IQ data as complex number
        
    Return:
        theta: angle of the major axis
        
    """
    X = np.stack((cplx.real, cplx.imag), axis=0)
    cov = np.cov(X, bias=0)
    eigval, eigvec = np.linalg.eig(cov)
    idx = np.argmin(eigval)
    theta = np.arctan2(eigvec[idx, 0], eigvec[idx, 1])
    return theta

class cplx_projector(object):
    """Mix-in class to project complex data
    
    """
    def __init__(self, angle=0, amplitude=1, offset=0):
        self.configure(angle, amplitude, offset)
        
    def configure(self, angle=None, amplitude=None, offset=None):
        """
            angle: projection angle [rad], the angle or rotating axis
            amplitude: projection amplitude, the amplitude after rotating axis
            offset: projection offset, the offset of origin after rotation axis
        """
        if not angle is None:
            self.angle = angle
        if not amplitude is None:
            self.amplitude = amplitude
        if not offset is None:
            self.offset = offset
    def project(self, cplx):
        return ((cplx*np.exp(-1j*self.angle)).real - self.offset)/self.amplitude

class cplx_loader(object):   # mixin
    def load_cplx_data(self, x, cplx, proj_angle=None):
        if isinstance(x, WithUnit):   ## unit string
            self.x = np.array(x[x.unit])
            self.x_unit = x.unit
        else:
            self.x = np.array(x)
        if isinstance(cplx, WithUnit):   ## unit string
            self.cplx = np.array(cplx[cplx.units])
            self.y_unit = cplx.unit
        else:
            self.cplx = np.array(cplx)
            self.y_unit = Unit('')
        
        if proj_angle is not None:
            self.proj_angle = proj_angle
        else:
            self.proj_angle = find_major_axis(self.cplx)
        self.cplx_n = self.cplx*np.exp(-1j*self.proj_angle)
        self.y = self.cplx_n.real

# fitter
class fitter_base(object):
    def __init__(self):
        """
            param name should be initialized in subclass
            * self.p_dict = [(p0 name, p0 unit), ..., (pn name, pn unit)]
            you can use dummy unit to indicate x and y units as 'xx' and 'yy'

        """
        self.p_dict = []
        self.p_init = None
        self.p_opt = None
        self.p_stdev = None
        self.p_cov = None
        self.x_unit = Unit('')
        self.y_unit = Unit('')
    
    def get_p_dict_subs(self):
        return [(name, subs_unit(unit, [('xx', self.x_unit), ('yy', self.y_unit)])) for (name, unit) in self.p_dict]
    
    def print_p_names(self, with_dummy_unit=False):
        if with_dummy_unit:
            name_unit_pairs = self.p_dict
        else:
            name_unit_pairs = self.get_p_dict_subs()
        for idx, (p, unit) in enumerate(name_unit_pairs):
            if unit == '':
                print ("param{0}: {1}".format(idx, p))
            else:
                print ("param{0}: {1} ({2})".format(idx, p, unit))                

    def remove_unit_from_p(self, p):
        p_dict = self.get_p_dict_subs()
        p_wo_unit = []
        for idx in range(len(p)):
            if isinstance(p[idx], WithUnit):
                if p[idx].isCompatible(p_dict[idx][1]):
                    p_wo_unit.append(p_init[idx][p_dict[idx][1]])
                else:
                    raise Exception('parameter {0} should have compatible unit with {1}'.format(p_dict[idx][0], p_dict[idx][1]))
            else:
                p_wo_unit.append(p[idx])
        return p_wo_unit

    def set_x_unit(self, unit):
        unit = Unit(unit)
        self.x_unit = unit
    
    def conv_x_unit(self, unit):
        unit = Unit(unit)
        if hasattr(self, 'x_unit') and self.x_unit.isCompatible(unit):
            factor = self.x_unit.conversionFactorTo(unit)
            self.x = self.x*factor
            self._scale_p('xx', factor)
        self.x_unit = unit
    
    def set_y_unit(self, unit):
        unit = Unit(unit)
        self.y_unit = unit

    def conv_y_unit(self, unit):
        unit = Unit(unit)
        if hasattr(self, 'y_unit') and self.y_unit.isCompatible(self.y_unit):
            factor = self.x_unit.conversionFactorTo(unit)
            self.y = self.y*factor
            self._scale_p('yy', factor)
        self.y_unit = unit
    
    def _scale_p(self, lex_unit, scale):
        p_scale_list = []
        for idx, (p_name, p_unit) in enumerate(self.p_dict):
            p_scale = 1
            if lex_unit in p_unit.lex_names:
                p_scale *= pow(scale, p_unit.lex_names[lex_unit])
            p_scale_list.append(p_scale)
        p_scale_arr = np.array(p_scale_list)
        
        if isinstance(self.p_init, np.ndarray):
            self.p_init *= p_scale_arr
        if isinstance(self.p_opt, np.ndarray):
            self.p_opt *= p_scale_arr
        if isinstance(self.p_stdev, np.ndarray):
            self.p_stdev *= p_scale_arr
        if isinstance(self.p_cov, np.ndarray):
            self.p_cov *= np.outer(p_scale_arr, p_scale_arr)
    
    def load_data(self, x, y):
        if isinstance(x, WithUnit):   ## unit string
            self.x = np.array(x[x.units])
            self.x_unit = x.unit
        else:
            self.x = np.array(x)
            self.x_unit = Unit('')

        if isinstance(y, WithUnit):   ## unit string
            self.y = np.array(y[y.units])
            self.y_unit = y.unit
        else:
            self.y = np.array(y)
            self.y_unit = Unit('')
    
    def fit_func(self, p, x):  ## defined in child class
        pass
    def err_func(self, p, x, y):
        return y - self.fit_func(p, x)
    def fit_curve_opt(self, x):
        if self.p_opt is None:
            raise Exception("Fit has not been done yet.")
        else:
            return self.fit_func(self.p_opt, x)
    def err_curve_opt(self, x, y):
        if self.p_opt is None:
            raise Exception("Fit has not been done yet.")
        else:
            return self.err_func(self.p_opt, x)
    def p_initializer(self):  ## defined in child class
        pass
    def post_process(self):  ## defined in child class
        pass
        
    def fit(self, p_init=None, **kwargs):
        if p_init is not None:
            if not len(p_init) == len(self.p_dict):
                raise Exception('initial paramter length is not matached')
        else:
            p_init = self.p_initializer()

        if isinstance(p_init[0], np.ndarray) or isinstance(p_init[0], list):  # allow multiple initial values
            self.p_init = []
            for idx in range(len(p_init)):
                self.p_init.append(self.remove_unit_from_p(p_init[idx]))
        else:
            self.p_init = self.remove_unit_from_p(p_init)

        self.p_init = np.array(self.p_init)
        
        residual = np.inf
        if self.p_init.ndim > 1:
            for idx in range(len(self.p_init)):
                p_opt1, p_cov1, residual1, fit_res1 = self._fit_main(self.p_init[idx], **kwargs)
                if residual1 <= residual:
                    p_opt = p_opt1
                    p_cov = p_cov1
                    residual = residual1
                    fit_res = fit_res1
        else:
            p_opt, p_cov, residual, fit_res = self._fit_main(self.p_init, **kwargs)
        
        self.p_opt = p_opt
        self.p_cov = p_cov
        self.residual = residual
        self.fit_res = fit_res
        if isinstance(self.p_cov, np.ndarray):
            self.p_stdev = np.sqrt(np.diag(self.p_cov))
        else:
            self.p_stdev = None
        
        self.post_process()
        return self.p_opt, self.p_cov, self.residual
    
    def _fit_main(self, p_init, **kwargs):
        fit_res = leastsq(self.err_func, p_init, args=(self.x, self.y), full_output=True)

        p_opt = fit_res[0]
        p_cov_raw = fit_res[1]
        residual = np.sum(fit_res[2]['fvec']**2)

        N = len(self.x)
        M = len(p_opt)
        y_var = np.sum(np.square(self.err_func(p_opt, self.x, self.y)))
        if isinstance(p_cov_raw, np.ndarray):
            p_cov = p_cov_raw * (y_var/(N-M)) # see the help of scipy.optimize.curve_fit
        else:
            p_cov = None
    
        return p_opt, p_cov, residual, fit_res
    
    def get_p_opt(self, with_unit=False):
        if not hasattr(self, 'p_opt'):
            raise Exception('Fitting is not done.')
    
        if not with_unit:
            return self.p_opt
        
        p_dict = self.get_p_dict_subs()
        p_opt_w_unit = []
        for idx in range(len(self.p_opt)):
            p_opt_w_unit.append(self.p_opt[idx]*Unit(p_dict[idx][1]))
            
        return p_opt_w_unit
    
    def get_p_stdev(self):
        if not hasattr(self, 'p_stdev'):
            raise Exception('Fitting is not done.')
        return self.p_stdev
    
    def get_p_unit(self):
        p_dict = self.get_p_dict_subs()
        return list(map(lambda x: x[1], p_dict))
    
    def print_p_opt(self):
        if not hasattr(self, 'p_opt'):
            raise Exception('Fitting is not done.')
        p_dict = self.get_p_dict_subs()
        for idx in range(len(self.p_opt)):
            if self.p_stdev is not None:
                print("{0} = {1:.2f} +/- {2:.2f} {3}".format(p_dict[idx][0], self.p_opt[idx], self.p_stdev[idx], p_dict[idx][1]))
            else:
                print("{0} = {1:.2f} {2}".format(p_dict[idx][0], self.p_opt[idx], p_dict[idx][1]))
    
    def plot(self, ax=None, data_label="data", fit_label="fit"):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.x, self.y, 'o', label=data_label)
        if hasattr(self, "p_opt") and (self.p_opt is not None):
            ax.plot(self.x, self.fit_func(self.p_opt, self.x), "--")
        #ax.set_xlabel('x [{0:s}]'.format(self.x_unit))
        #ax.set_ylabel('y [{0:s}]'.format(self.y_unit))
        return ax