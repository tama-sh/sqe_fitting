import numpy as np
import operator
import lmfit

from lmfit.models import (
    LorentzianModel,
    ExponentialModel,
    ConstantModel,
    GaussianModel,
    update_param_vals
)

from .util import percentile_range_data
from .signal_util import middle_points, derivative, smoothen, find_peaks, guess_peak_or_dip, guess_linewidth_from_peak
from .electrical_delay_fitter import (
    estimate_electrical_delay_resonator,
    estimate_electrical_delay_unwrap,
    estimate_electrical_delay_from_group_delay,
    estimate_electrical_delay_from_edge_delay,
    correct_electrical_delay)
from .circle_fitter import algebric_circle_fit
import scipy.signal as scisig

# Models
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
        self.set_param_hint('frequency', min=0)
        #self.set_param_hint('phase', min=-np.pi, max=np.pi) # tend to stack at the boundary
        self.set_param_hint('phase', min=-2*np.pi-1e-5, max=2*np.pi+1e-5) 
        
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
        # amplitude = 2*np.abs(peak_amp)*sigma  # factor 2 because of cos(omega*t) = (1/2)*(exp(omega*t)+exp(-omega*t))
        amplitude = 0.5*np.ptp(data)
        frequency = peak_freq
        phase = np.angle(peak_amp*np.exp(-1j*2*np.pi*frequency*x[0]))
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
            raise ValueError(f"Reflection type '{reflection_type}' is not supprted")
        super().__init__(resonator_reflection, **kwargs)
        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        self.set_param_hint('kappa_ex', min=0)
        self.set_param_hint('kappa_in', min=0)
        self.set_param_hint('a', min=0)
        self.set_param_hint('reflection_factor', value=self.reflection_factor, vary=False)
        
    def guess(self, cplx, omega, smoothing_width=10, fix_electrical_delay=False, electrical_delay_estimation="default", **kwargs):
        pars = self.make_params()
        
        # estimate electrical delay
        if not fix_electrical_delay:
            if electrical_delay_estimation == "default":
                 electrical_delay = estimate_electrical_delay_resonator(cplx, omega)
            elif electrical_delay_estimation == "group delay":
                electrical_delay = estimate_electrical_delay_from_group_delay(cplx, omega)
            elif electrical_delay_estimation == "edge group delay":
                electrical_delay = estimate_electrical_delay_from_edge_delay(cplx, omega)
            elif electrical_delay_estimation == "unwrap overcoupled":
                electrical_delay = estimate_electrical_delay_unwrap(cplx, omega, accumulated_phase=-2*np.pi)
            elif electrical_delay_estimation == "unwrap double resonator overcoupled":
                electrical_delay = estimate_electrical_delay_unwrap(cplx, omega, accumulated_phase=-4*np.pi)
            elif electrical_delay_estimation == "unwrap undercoupled":
                electrical_delay = estimate_electrical_delay_unwrap(cplx, omega, accumulated_phase=0)
            elif electrical_delay_estimation == "none":
                electrical_delay = 0
            else:
                raise ValueError(f"Estimation method '{electrical_delay_estimation}' is not supprted")
            cplx_c = correct_electrical_delay(cplx, omega, electrical_delay)
        else:
            cplx_c = cplx

        # estimate amplitude baseline
        a = np.mean(percentile_range_data(abs(cplx_c), (0.75, 1)))
        
        # derivative-based guess
        omega_mid = middle_points(omega)
        cplx_lp = smoothen(cplx_c, smoothing_width=smoothing_width)
        s_lorentz = np.abs(derivative(cplx_lp, omega)) # this derivative should be Lorentzian if electrical delay is well calibrated

        lmodel = Lorentzian_plus_ConstantModel()
        lparams = lmodel.guess(s_lorentz, omega_mid, negative=False)
        rst = lmodel.fit(s_lorentz, params=lparams, x=omega_mid)

        amp = rst.params['amplitude'].value
        mu = rst.params['center'].value
        sigma = rst.params['sigma'].value
        
        omega_0 = mu
        kappa_tot = 2*sigma
        kappa_ex = amp*sigma/(np.pi*a)/self.reflection_factor
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

def double_resonator_reflection(omega, omega_0, kappa_ex_0, kappa_in_0, phi_0, omega_1, kappa_ex_1, kappa_in_1, phi_1, a, tau, theta, reflection_factor=1):
    return a*np.exp(1j*(theta-omega*tau))*(1-(reflection_factor*2*kappa_ex_0*np.exp(1j*phi_0))/(kappa_ex_0+kappa_in_0+2j*(omega-omega_0))
                                            -(reflection_factor*2*kappa_ex_1*np.exp(1j*phi_1))/(kappa_ex_1+kappa_in_1+2j*(omega-omega_1)))

class DoubleResonatorReflectionModel_Parallel(lmfit.model.Model):
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
            raise ValueError(f"Reflection type '{reflection_type}' is not supprted")
        super().__init__(double_resonator_reflection, **kwargs)
        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        self.set_param_hint('kappa_ex_0', min=0)
        self.set_param_hint('kappa_in_0', min=0)
        self.set_param_hint('kappa_ex_1', min=0)
        self.set_param_hint('kappa_in_0', min=0)
        self.set_param_hint('kappa_in_1', min=0)
        self.set_param_hint('a', min=0)
        self.set_param_hint('reflection_factor', value=self.reflection_factor, vary=False)
        
    def guess(self, cplx, omega, smoothing_width=10, fix_electrical_delay=True, **kwargs):
        pars = self.make_params()
        
        electrical_delay = estimate_electrical_delay_from_edge_delay(cplx, omega)
        cplx_c = correct_electrical_delay(cplx, omega, electrical_delay)

        # estimate amplitude baseline
        a = np.mean(percentile_range_data(abs(cplx_c), (0.75, 1)))
        
        # derivative-based guess
        omega_mid = middle_points(omega)
        cplx_lp = smoothen(cplx_c, smoothing_width=smoothing_width)
        s_lorentz = np.abs(derivative(cplx_lp, omega)) # this derivative should be Lorentzian if electrical delay is well calibrated
        
        double_lorentzian_model = LorentzianModel(prefix='r0_') + LorentzianModel(prefix='r1_')
        pars = double_lorentzian_model.make_params()
        
        peaks, properties = find_peaks(s_lorentz, omega_mid, height=10, prominence=10)
        sigmas = 0.5*scisig.peak_widths(s_lorentz, peaks, rel_height=0.5)[0]*(omega_mid[1]-omega_mid[0])
        heights = properties['peak_heights']
        
        print('TESTING: peaks:', peaks)
        print('omega_mid[peaks[0]] :', omega_mid[peaks[0]] )

        if len(peaks) == 2:
            pars['r0_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r1_amplitude'].set(value=heights[1]*(sigmas[1]*np.pi))
            pars['r0_center'].set(value=omega_mid[peaks[0]]) 
            pars['r1_center'].set(value=omega_mid[peaks[1]])
            pars['r0_sigma'].set(value=sigmas[0])
            pars['r1_sigma'].set(value=sigmas[1])
        else:
            pars['r0_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r1_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r0_center'].set(value=omega_mid[peaks[0]]) 
            pars['r1_center'].set(value=omega_mid[peaks[0]] + 50) ### add 50 MHz
            pars['r0_sigma'].set(value=sigmas[0])
            pars['r1_sigma'].set(value=sigmas[0])
        
        rst = double_lorentzian_model.fit(s_lorentz, x=omega_mid, params=pars)
        
        amp_0 = rst.params['r0_amplitude'].value
        mu_0 = rst.params['r0_center'].value
        sigma_0 = rst.params['r0_sigma'].value
        
        amp_1 = rst.params['r1_amplitude'].value
        mu_1 = rst.params['r1_center'].value
        sigma_1 = rst.params['r1_sigma'].value
        
        omega_0 = mu_0
        kappa_tot_0 = 2*sigma_0
        kappa_ex_0 = amp_0*sigma_0/(np.pi*a)/self.reflection_factor
        kappa_in_0 = max(0, kappa_tot_0-kappa_ex_0)
        omega_1 = mu_1
        kappa_tot_1 = 2*sigma_1
        kappa_ex_1 = amp_1*sigma_1/(np.pi*a)/self.reflection_factor
        kappa_in_1 = max(0, kappa_tot_1-kappa_ex_1)
        
        # parepare parameters
        pars = self.make_params()

        print('HERE: pars:', pars)

        pars['a'].set(value=a)
        pars['omega_0'].set(value=omega_0)
        pars['kappa_ex_0'].set(value=kappa_ex_0)
        pars['kappa_in_0'].set(value=kappa_in_0)
        pars['omega_1'].set(value=omega_1)
        pars['kappa_ex_1'].set(value=kappa_ex_1)
        pars['kappa_in_1'].set(value=kappa_in_1)   
        pars['phi_0'].set(value=0)
        pars['phi_1'].set(value=0)
        
        if fix_electrical_delay:
            pars['tau'].set(value=0, vary=False)
        else:
            pars['tau'].set(value=electrical_delay)
        pars['theta'].set(value=0)

        return update_param_vals(pars, self.prefix, **kwargs)

def double_resonator_reflection_series(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, a, a_grad, tau, theta, reflection_factor=1):

    val = (a+a_grad*(omega))*np.exp(1j*(theta-omega*tau))*(1-(-1j*4*reflection_factor*kappa_p*(omega_r - omega))/(4 * J**2 + (2*-1j*(omega_p - omega) + kappa_p + kappa_in_p) * (2*-1j*(omega_r - omega) + kappa_in_r)))  

    #val = np.unwrap(np.angle(val))

    return val

class DoubleResonatorReflectionModel_Series(lmfit.model.Model):
    def __init__(self, independent_vars=['omega'], prefix='', nan_policy='raise', reflection_type='normal', fit_phase_only = False, **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        self.fit_phase_only = fit_phase_only
        if reflection_type == 'normal':
            self.reflection_type = reflection_type
            self.reflection_factor = 1
        elif reflection_type == 'hanger':
            self.reflection_type = reflection_type
            self.reflection_factor = 0.5
        else:
            raise ValueError(f"Reflection type '{reflection_type}' is not supported")
        
        if self.fit_phase_only:

            def return_phase(func):

                def wrapper(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, a, a_grad, tau, theta, reflection_factor=1):

                    val = np.unwrap(np.angle(func(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, a, a_grad, tau, theta, reflection_factor=reflection_factor)))

                    return val
                
                return wrapper
            
            unwrapped_double_resonator_reflection_series = return_phase(double_resonator_reflection_series)
            super().__init__(unwrapped_double_resonator_reflection_series, **kwargs)

        else:
            super().__init__(double_resonator_reflection_series, **kwargs)

        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        # attributes = dir(self)
        # print('attributes:', attributes)
        # print('attribute keys:', self.__dict__.values())
        self.set_param_hint('kappa_p', min=40, max = 150)
        self.set_param_hint('J', min=5, max = 50)
        self.set_param_hint('omega_r', min=0)
        self.set_param_hint('omega_p', min=0)
        if self.fit_phase_only:
            self.set_param_hint('kappa_in_r', min = 0, max = 0.5)
            self.set_param_hint('kappa_in_p', min = 0, max = 0.5)
        else:
            self.set_param_hint('kappa_in_r', min = 0, max = 2)
            self.set_param_hint('kappa_in_p', min = 0, max = 2)
        #self.set_param_hint('a', min=1e9, max = 2*10e9)
        self.set_param_hint('a_grad', min=-2*10e9, max = 2*10e9)
        self.set_param_hint('reflection_factor', value=self.reflection_factor, vary=False)
        self.set_param_hint('theta', min=-0.1*np.pi, max = 0.1*np.pi)
        self.set_param_hint('tau', min=-0.04, max = 0.04)

    def guess(self, cplx, omega, smoothing_width=8, fix_electrical_delay=True, **kwargs):
        pars = self.make_params()
        
        electrical_delay = estimate_electrical_delay_from_edge_delay(cplx, omega)
        cplx_c = correct_electrical_delay(cplx, omega, electrical_delay)

        # estimate amplitude baseline
        a = np.mean(percentile_range_data(abs(cplx_c), (0.75, 1)))
        
        # derivative-based guess
        omega_mid = middle_points(omega)

        cplx_lp = smoothen(cplx_c, smoothing_width=smoothing_width)
        s_lorentz = np.abs(derivative(cplx_lp, omega)) # this derivative should be Lorentzian if electrical delay is well calibrated

        start_a = np.abs(cplx_lp[0])
        end_a = np.abs(cplx_lp[-1])

        grad_a_guess = (end_a - start_a)/(omega[-1] - omega[0])

        double_lorentzian_model = LorentzianModel(prefix='r0_') + LorentzianModel(prefix='r1_')
        pars = double_lorentzian_model.make_params()

        import matplotlib.pyplot as plt

        peaks, properties = find_peaks(s_lorentz, omega_mid, height=4, prominence=4)
        sigmas = 0.5*scisig.peak_widths(s_lorentz, peaks, rel_height=0.5)[0]*(omega_mid[1]-omega_mid[0])
        heights = properties['peak_heights']

        if len(peaks) == 2:
            pars['r0_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r1_amplitude'].set(value=heights[1]*(sigmas[1]*np.pi))
            pars['r0_center'].set(value=omega_mid[peaks[0]]+ 10) ### add 10 MHz
            pars['r1_center'].set(value=omega_mid[peaks[1]])
            pars['r0_sigma'].set(value=sigmas[0])
            pars['r1_sigma'].set(value=sigmas[1])
        else:
            pars['r0_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r1_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r0_center'].set(value=omega_mid[peaks[0]] + 10) 
            pars['r1_center'].set(value=omega_mid[peaks[0]] + 10 + 25) ### add 25 MHz
            pars['r0_sigma'].set(value=sigmas[0]) 
            pars['r1_sigma'].set(value=2*sigmas[0]) # multiply by 2 as filter resonance likely to be much wider
        

        rst = double_lorentzian_model.fit(s_lorentz, x=omega_mid, params=pars)

        #freq_fit = np.linspace(min(freq), max(freq), fit_slice)
            
        response_fit = rst.eval(params=rst.params, omega=omega)
        
        # plt.plot(omega[:-1], s_lorentz)
        # plt.plot(omega[:-1], response_fit)
        # plt.show()

        amp_0 = rst.params['r0_amplitude'].value
        mu_0 = rst.params['r0_center'].value
        sigma_0 = rst.params['r0_sigma'].value
        
        amp_1 = rst.params['r1_amplitude'].value
        mu_1 = rst.params['r1_center'].value
        sigma_1 = rst.params['r1_sigma'].value

        print('omega0:',  pars['r0_center'])
        print('omega1:',  pars['r1_center'])

        ##
        omega_l = mu_0
        kappa_tot_l = 2*sigma_0
        # kappa_ex_l = amp_0*sigma_0/(np.pi*a)/self.reflection_factor
        # kappa_in_l = max(0, kappa_tot_r-kappa_ex_r)
        omega_h = mu_1
        kappa_tot_h = 2*sigma_1
        # kappa_ex_h = amp_1*sigma_1/(np.pi*a)/self.reflection_factor
        # kappa_in_h = max(0, kappa_tot_p-kappa_ex_p)
        ##

        print('kappa_tot_l, kappa_tot_h:', kappa_tot_l, kappa_tot_h)

        kappa_p = kappa_tot_l + kappa_tot_h
        print('kappa_p:',  kappa_p)
        #input('...')
        # parepare parameters
        pars = self.make_params()

        #print('HERE: pars:', pars)

        omega_middle = (omega[-1] + omega[0])/2

        pars['a'].set(value=a - grad_a_guess*omega_middle, vary = False)
        pars['a_grad'].set(value=grad_a_guess, vary=False)

        pars['omega_r'].set(value=omega_l + 10) # add 10 MHz to account for hybridization avoided crossing
        pars['kappa_in_r'].set(value = 0) ## assume no internal loss
        pars['omega_p'].set(value=omega_h - 10) # remove 10 MHz to account for hybridization avoided crossing
        pars['kappa_in_p'].set(value = 0)
        pars['kappa_p'].set(value=kappa_p)
        pars['J'].set(value= 20) # assume J value of 25 MHz

        if fix_electrical_delay:
            pars['tau'].set(value=0, vary=False)
        else:
            pars['tau'].set(value=electrical_delay)
        pars['theta'].set(value=0, vary = False)

        return update_param_vals(pars, self.prefix, **kwargs)


# Composite models

class DoubleResonatorReflectionModel_Series_g_e():
    
    def __init__(self, omega_r, omega_p, kappa_in_r, kappa_in_p, kappa_p, J, fit_phase_only = False, reflection_type = 'normal'):

        self.params = lmfit.Parameters()
        self.omega_r = omega_r
        self.omega_p = omega_p
        self.kappa_in_r = kappa_in_r
        self.kappa_in_p = kappa_in_p
        self.kappa_p = kappa_p
        self.J = J
        self.fit_phase_only = fit_phase_only
        if reflection_type == 'normal':
            self.reflection_type = reflection_type
            self.reflection_factor = 1
        elif reflection_type == 'hanger':
            self.reflection_type = reflection_type
            self.reflection_factor = 0.5
        else:
            raise ValueError(f"Reflection type '{reflection_type}' is not supported")
                
        ## set the g parameters
        state = 'g'
        self.params.add('{}_kappa_p'.format(state), value = self.kappa_p, min=max(0, self.kappa_p - 20), max = self.kappa_p + 20)
        self.params.add('{}_J'.format(state), value = self.J, min=max(0, self.J - 15), max = self.J + 15)
        self.params.add('{}_omega_r'.format(state), value = self.omega_r, min=self.omega_r - 15, max = self.omega_r + 15)
        self.params.add('{}_omega_p'.format(state), value = self.omega_p, min=self.omega_p - 15, max = self.omega_p + 15)
        if self.fit_phase_only:
            self.params.add('{}_kappa_in_r'.format(state), value = self.kappa_in_r, min=0, max = 0.5)
            self.params.add('{}_kappa_in_p'.format(state), value = self.kappa_in_p, min=0, max = 0.5)
        else:
            self.params.add('{}_kappa_in_r'.format(state), value = self.kappa_in_r, min = 0, max = 2)
            self.params.add('{}_kappa_in_p'.format(state), value = self.kappa_in_p, min = 0, max = 2)

        self.params.add('{}_a'.format(state), min=1e9, max = 30*10e9)
        self.params.add('{}_a_grad'.format(state), min=-2*10e9, max = 2*10e9)
        self.params.add('{}_reflection_factor'.format(state), value=self.reflection_factor, vary=False)
        self.params.add('{}_theta'.format(state), min=-0.1*np.pi, max = 0.1*np.pi)
        self.params.add('{}_tau'.format(state), min=-0.075, max = 0.075) ## 0.04

        ### constrain the 'e' parameters
        state = 'e'
        self.params.add('{}_kappa_p'.format(state), expr='g_kappa_p')
        self.params.add('{}_J'.format(state), expr='g_J')
        self.params.add('chi'.format(state), value = 1, min = -10, max = 0)
        self.params.add('{}_omega_r'.format(state), expr='g_omega_r + 2*chi')
        self.params.add('{}_omega_p'.format(state), expr='g_omega_p')
        if self.fit_phase_only:
            self.params.add('{}_kappa_in_r'.format(state), expr='g_kappa_in_r')
            self.params.add('{}_kappa_in_p'.format(state), expr='g_kappa_in_p')
        else:
            self.params.add('{}_kappa_in_r'.format(state), expr='g_kappa_in_r')
            self.params.add('{}_kappa_in_p'.format(state), expr='g_kappa_in_p')
        self.params.add('{}_a'.format(state), expr='g_a')
        self.params.add('{}_a_grad'.format(state), expr='g_a_grad')
        self.params.add('{}_reflection_factor'.format(state), expr='g_reflection_factor')
        self.params.add('{}_theta'.format(state), expr='g_theta')
        self.params.add('{}_tau'.format(state), expr='g_tau')

    def guess(self, cplx, omega, smoothing_width=8, fix_electrical_delay=True, **kwargs):
                
        electrical_delay = estimate_electrical_delay_from_edge_delay(cplx, omega)
        cplx_c = correct_electrical_delay(cplx, omega, electrical_delay)

        # estimate amplitude baseline
        a = np.mean(percentile_range_data(abs(cplx_c), (0.75, 1)))
        
        # derivative-based guess
        omega_mid = middle_points(omega)

        cplx_lp = smoothen(cplx_c, smoothing_width=smoothing_width)
        s_lorentz = np.abs(derivative(cplx_lp, omega)) # this derivative should be Lorentzian if electrical delay is well calibrated

        start_a = np.abs(cplx_lp[0])
        end_a = np.abs(cplx_lp[-1])

        grad_a_guess = (end_a - start_a)/(omega[-1] - omega[0])

        double_lorentzian_model = LorentzianModel(prefix='r0_') + LorentzianModel(prefix='r1_')
        pars = double_lorentzian_model.make_params()

        import matplotlib.pyplot as plt

        peaks, properties = find_peaks(s_lorentz, omega_mid, height=4, prominence=4)
        sigmas = 0.5*scisig.peak_widths(s_lorentz, peaks, rel_height=0.5)[0]*(omega_mid[1]-omega_mid[0])
        heights = properties['peak_heights']

        if len(peaks) == 2:
            pars['r0_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r1_amplitude'].set(value=heights[1]*(sigmas[1]*np.pi))
            pars['r0_center'].set(value=omega_mid[peaks[0]]+ 10) ### add 10 MHz
            pars['r1_center'].set(value=omega_mid[peaks[1]])
            pars['r0_sigma'].set(value=sigmas[0])
            pars['r1_sigma'].set(value=sigmas[1])
        else:
            pars['r0_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r1_amplitude'].set(value=heights[0]*(sigmas[0]*np.pi))
            pars['r0_center'].set(value=omega_mid[peaks[0]] + 10) 
            pars['r1_center'].set(value=omega_mid[peaks[0]] + 10 + 25) ### add 25 MHz
            pars['r0_sigma'].set(value=sigmas[0]) 
            pars['r1_sigma'].set(value=2*sigmas[0]) # multiply by 2 as filter resonance likely to be much wider
        
        rst = double_lorentzian_model.fit(s_lorentz, x=omega_mid, params=pars)

        #freq_fit = np.linspace(min(freq), max(freq), fit_slice)
            
        response_fit = rst.eval(params=rst.params, omega=omega)
        
        # plt.plot(omega[:-1], s_lorentz)
        # plt.plot(omega[:-1], response_fit)
        # plt.show()

        amp_0 = rst.params['r0_amplitude'].value
        mu_0 = rst.params['r0_center'].value
        sigma_0 = rst.params['r0_sigma'].value
        
        amp_1 = rst.params['r1_amplitude'].value
        mu_1 = rst.params['r1_center'].value
        sigma_1 = rst.params['r1_sigma'].value

        print('omega0:',  pars['r0_center'])
        print('omega1:',  pars['r1_center'])

        ##
        omega_l = mu_0
        kappa_tot_l = 2*sigma_0
        # kappa_ex_l = amp_0*sigma_0/(np.pi*a)/self.reflection_factor
        # kappa_in_l = max(0, kappa_tot_r-kappa_ex_r)
        omega_h = mu_1
        kappa_tot_h = 2*sigma_1
        # kappa_ex_h = amp_1*sigma_1/(np.pi*a)/self.reflection_factor
        # kappa_in_h = max(0, kappa_tot_p-kappa_ex_p)
        ##

        print('kappa_tot_l, kappa_tot_h:', kappa_tot_l, kappa_tot_h)

        kappa_p = kappa_tot_l + kappa_tot_h
        print('kappa_p:',  kappa_p)

        omega_middle = (omega[-1] + omega[0])/2

        self.params['g_a'].set(value= a - grad_a_guess*omega_middle, vary = False)

        # print('a - grad_a_guess*omega_middle:', a - grad_a_guess*omega_middle)
        #input('...')
        
        self.params['g_a_grad'].set(value=grad_a_guess, vary=False)

        if fix_electrical_delay:
            self.params['g_tau'].set(value=0, vary=False)
        else:
            self.params['g_tau'].set(value=electrical_delay)
        self.params['g_theta'].set(value=0, vary = False)

        return self.params

    ## constrain the parameters

    def simultaneous_fit(self, data, omega, params = None, method = 'basinhopping'):

        from lmfit import minimize

        def fit_func(params, i, omega):

            if i == 0:
                state = 'g'
            elif i == 1:
                state = 'e'
        
            omega_r = params['{}_omega_r'.format(state)].value
            kappa_in_r = params['{}_kappa_in_r'.format(state)].value
            omega_p = params['{}_omega_p'.format(state)].value
            kappa_in_p = params['{}_kappa_in_p'.format(state)].value
            kappa_p = params['{}_kappa_p'.format(state)].value
            J = params['{}_J'.format(state)].value
            a = params['{}_a'.format(state)].value
            a_grad = params['{}_a_grad'.format(state)].value
            tau = params['{}_tau'.format(state)].value
            theta = params['{}_theta'.format(state)].value
            reflection_factor = params['{}_reflection_factor'.format(state)].value
                    
            val = double_resonator_reflection_series(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, a, a_grad, tau, theta, reflection_factor=reflection_factor)

            if self.fit_phase_only:
                val = np.unwrap(np.angle(val))

            return val

        def objective(params, omega, data):
            """ calculate total residual for fits to several data sets held
            in a 2-D array, and modeled by Gaussian functions"""
            ndata, nx = data.shape
            resid = 0.0*data[:]
            # make residual per data set
            for i in range(ndata):
                resid[i, :] = data[i, :] - fit_func(params, i, omega)
            # now flatten this to a 1D array, as minimize() needs
            return np.abs(resid.flatten())

        if params is None:    
            params = self.self.params

        result = minimize(objective, params, args=(omega, data), method = method)

        return result

    def eval(self, omega, params = None):
        
        if params is None:    
            params = self.params

        vals = []

        states = ['g', 'e']
        
        param_dict = params.valuesdict()

        for state in states:

            state_param_dict = {key[2:]:value for key, value in param_dict.items() if (state +'_') in key}

            val = double_resonator_reflection_series(omega, **state_param_dict)
            
            if self.fit_phase_only:
                val = np.unwrap(np.angle(val))
            
            vals.append(val)

        return np.array(vals)

class Lorentzian_plus_ConstantModel(lmfit.model.CompositeModel):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        super().__init__(LorentzianModel(**kwargs), ConstantModel(**kwargs), operator.add)
    
    def guess(self, data, x, negative=None, **kwargs):
        if negative is None:
            negative = not guess_peak_or_dip(data)
            
        if negative:
            c_init = max(data)
            data_peak = -data
        else:
            c_init = min(data)
            data_peak = data

        # # use guess function of LorentzianModel
        # params = self.left.guess(data-c_init, x=x, negative=self.negative, **kwargs)

        sigma = guess_linewidth_from_peak(x, data_peak)
        idx_c = np.argmax(data_peak)
        mu = x[idx_c]
        A = np.pi*sigma*data[idx_c]
        
        params = self.make_params()
        params['amplitude'].set(value=A)
        params['center'].set(value=mu)
        params['sigma'].set(value=sigma)
        params['c'].set(value=c_init)
        
        return params 
   
class Exponential_plus_ConstantModel(lmfit.model.CompositeModel):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        super().__init__(ExponentialModel(**kwargs), ConstantModel(**kwargs), operator.add)
    
    def guess(self, data, x, negative=None, **kwargs):
        if negative is None:
            N = len(data)
            if np.mean(data[:N//2]) < np.mean(data[N//2:]):
                negative = True
            else:
                negative = False
        
        params = self.left.make_params()
        if negative:
            c_init = max(data)
            factor = -1
        else:
            c_init = min(data)
            factor = 1
        data_exp = factor*(data - c_init)
        idxs = np.where(data_exp < data_exp[0] - 0.5*np.ptp(data_exp))[0]
        if len(idxs) > 0:
            idx = np.where(data_exp < data_exp[0] - 0.5*np.ptp(data_exp))[0][0]
        else:
            idx = len(data)//2
        decay = x[idx]*np.log(2)
        amplitude = np.ptp(data_exp)
        params['amplitude'].set(value=factor*amplitude)
        params['decay'].set(value=decay, min=0)
        params.add('c', value=c_init)
        return params

class DampedOscillation_plus_ConstantModel(lmfit.model.CompositeModel):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        super().__init__(DampedOscillationModel(**kwargs), ConstantModel(**kwargs), operator.add)
    
    def guess(self, data, x, **kwargs):
        c_init = np.mean(percentile_range_data(data, (0.25, 0.75)))
        
        params = self.left.guess(data-c_init, x=x)
        params.add('c', value=c_init)
        return params
