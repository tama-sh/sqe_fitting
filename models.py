import numpy as np
import lmfit 

from lmfit.models import (
    LorentzianModel,
    ConstantModel,
    update_param_vals
)

from .util import percentile_range_data
from .signal_util import middle_points, derivative, smoothen
from .lorentzian_fitter import lorentzian_fitter, guess_linewidth_from_peak
from .electrical_delay_fitter import estimate_electrical_delay_resonator, correct_electrical_delay
from .circle_fitter import algebric_circle_fit


def damped_oscillation(x, amplitude, decay, frequency, phase):
    return amplitude*np.exp(-x/decay)*np.cos(2*np.pi*frequency*x + phase)

def resonator_reflection(omega, omega_0, kappa_ex, kappa_in, a, tau, theta, phi, reflection_factor=1):
    """Frequency-dependent referection form a resonator
    Note that the function is written by using the unit of imaginary number 'j', instead of 'i' in quantum mechanics.
    You can obtain this function by substituting i = -j.

    Args:
        omega: frequency, independent value
        omega_0: center frequency of resonator
        kappa_ex: external decay rate
        kappa_in: internal decay rates
        a: amplitude
        tau: electrical delay
        theta: phase offset
        phi: tilt angle of the circle fits
        reflection_factor: 1 for normal reflection, 0.5 for hanger
    """
    return a*np.exp(1j*(theta-omega*tau))*(1-(reflection_factor*2*kappa_ex*np.exp(1j*phi))/(kappa_ex+kappa_in+2j*(omega-omega_0)))

class DampedOscillationModel(lmfit.model.Model):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(damped_oscillation, **kwargs)
        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('phase', min=-np.pi, max=np.pi)
        
    def guess(self, data, x, **kwargs):
        N = len(data)
        data_no_dc = data - np.mean(data)  # need to remove dc peak from fft
        fft_data = np.fft.fft(data_no_dc)[:int(N/2)] # take only positive freq
        fft_freq = np.fft.fftfreq(N, x[1]-x[0])[:int(N/2)]
        
        peak_idx = np.argmax(np.abs(fft_data))
        peak_amp = fft_data[peak_idx]
        peak_freq = fft_freq[peak_idx]
        sigma = guess_linewidth_from_peak(2*np.pi*fft_freq, abs(fft_data)**2) # estimate decay rate from linewidth of fft peak
        
        # estimate parameters
        amplitude = 2*np.abs(peak_amp)*sigma  # factor 2 because of cos(omega*t) = (1/2)*(exp(omega*t)+exp(-omega*t))
        phase = np.angle(peak_amp)
        frequency = peak_freq
        decay = 1/sigma
        
        pars = self.make_params()
        pars['amplitude'].set(value=amplitude)
        pars['decay'].set(value=decay)
        pars['frequency'].set(value=frequency)
        pars['phase'].set(value=phase)
        
        return pars

class ResonatorReflectionModel(lmfit.model.Model):
    def __init__(self, independent_vars=['omega'], prefix='', nan_policy='raise', reflection_type='normal', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        if reflection_type == 'normal':
            self.reflection_type = reflection_type
            self.reflection_factor = 1
        elif reflection_type == 'hanger':
            self.reflection_type = reflection_type
            self.reflection_factor = 0.5
        else:
            ValueError(f"Reflection type '{reflection_type}' is not supprted")
        super().__init__(resonator_reflection, **kwargs)
        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        self.set_param_hint('omega_0', min=0)
        self.set_param_hint('kappa_ex', min=0)
        self.set_param_hint('kappa_in', min=0)
        self.set_param_hint('a', min=0)
        self.set_param_hint('reflection_factor', value=self.reflection_factor, vary=False)
        
    def guess(self, cplx, omega, smoothing_width=10, fix_electrical_delay=False, **kwargs):
        pars = self.make_params()

        # estimate electrical delay
        if not fix_electrical_delay:
            electrical_delay = estimate_electrical_delay_resonator(omega, cplx)
            cplx_c = correct_electrical_delay(omega, cplx, electrical_delay)
        else:
            cplx_c = cplx

        # estimate amplitude baseline
        a = np.mean(percentile_range_data(abs(cplx_c), (0.75, 1)))
        
        # derivative-based guess
        omega_mid = middle_points(omega)
        cplx_lp = smoothen(cplx_c, smoothing_width=smoothing_width)
        s_lorentz = np.abs(derivative(cplx_lp, omega)) # this derivative should be Lorentzian if electrical delay is well calibrated

        fitter_lorentz = lorentzian_fitter()
        rst = fitter_lorentz.fit(omega_mid, s_lorentz)

        amp = rst.params['amplitude'].value
        mu = rst.params['center'].value
        sigma = rst.params['sigma'].value
        
        omega_0 = mu
        kappa_tot = 2*sigma
        kappa_ex = self.reflection_factor*amp*sigma/(np.pi*a)
        kappa_in = max(0, kappa_tot-kappa_ex)

        # estimate phase and tilt
        z_inf = 0.5*(cplx_c[0]+cplx_c[-1])
        theta = np.angle(z_inf) # angle of infinite point
        rst = algebric_circle_fit(cplx_c.real, cplx_c.imag)
        x_c, y_c, r_0 = rst.params['x_c'], rst.params['y_c'], rst.params['r_0']
        z_c = x_c+1j*y_c
        phi = np.angle((z_inf-z_c)*np.conj(z_inf))
        
        # parepare parameters
        pars = self.make_params()
        pars['a'].set(value=a)
        pars['omega_0'].set(value=omega_0)
        pars['kappa_ex'].set(value=kappa_ex)
        pars['kappa_in'].set(value=kappa_in)   

        if fix_electrical_delay:
            pars['tau'].set(value=0, vary=False)
        else:
            pars['tau'].set(value=electrical_delay)
        pars['phi'].set(value=phi)
        pars['theta'].set(value=theta)
        
        return update_param_vals(pars, self.prefix, **kwargs)
    