import numpy as np
import operator
import lmfit 
from functools import reduce
import inspect

from lmfit.models import (
    LorentzianModel,
    ExponentialModel,
    ConstantModel,
    update_param_vals
)

from .util import percentile_range_data
from .signal_util import middle_points, derivative, smoothen, find_peaks, guess_peak_or_dip, guess_linewidth_from_peak
from .electrical_delay_fitter import (
    estimate_electrical_delay_resonator,
    estimate_electrical_delay_unwrap,
    estimate_electrical_delay_from_group_delay,
    correct_electrical_delay)
from .circle_fitter import algebric_circle_fit
import scipy.signal as scisig

# Models
def damped_oscillation(x, amplitude, decay, frequency, phase):
    return amplitude*np.exp(-x/decay)*np.cos(2*np.pi*frequency*x + phase)

def electrical_delay(omega, tau, theta):
    """Frequency-dependent phase caused by electrical delay
    
    Args:
        omega: frequency, independent value
        tau: electrical delay
        theta: phase offset
    """
    return np.exp(1j*(theta-omega*tau))

def cable_attenuation(omega, attn_coeff):
    """Frequency-dependent loss of coaxial cable
    The loss of the coaxial cable normaly depend like
    * attn (dB) = attn_coeff*sqrt(f)
    """
    return 10**(-attn_coeff*np.sqrt(omega)/10)

def resonator_reflection_base(omega, omega_0, kappa_ex, kappa_in, phi, reflection_factor=1):
    return 1-(reflection_factor*kappa_ex*np.exp(1j*phi))/(0.5*kappa_ex+kappa_in+1j*(omega-omega_0))

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
    return a*electrical_delay(omega, tau, theta)*resonator_reflection_base(omega, omega_0, kappa_ex, kappa_in, phi, reflection_factor)

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

class PolarWeightModel(lmfit.model.Model):
    def __init__(self, func, independent_vars=None, param_names=None, nan_policy='raise', prefix='', name=None, **kws):
        super().__init__(func, independent_vars, param_names, nan_policy, prefix, name, **kws)

    def _residual(self, params, data, weights, **kwargs):
        """Return the residual.
        Weight is applyed for amplitude (real part) and phase (imaginary part) direction based.
        This is done by rotating the diff by the phase of the model
        """
        if not np.issubdtype(data.dtype, np.complexfloating): # "diff.dtype is complex" in lmfit was not working well for complex128
            raise ValueError("The data type should be complex.")

        model = self.eval(params, **kwargs)
        if self.nan_policy == 'raise' and not np.all(np.isfinite(model)):
            msg = ('The model function generated NaN values and the fit '
                   'aborted! Please check your model function and/or set '
                   'boundaries on parameters where applicable. In cases like '
                   'this, using "nan_policy=\'omit\'" will probably not work.')
            raise ValueError(msg)

        diff = data - model

        model_is_zero = (model == 0)
        rotation_factors = np.where(
            np.invert(model_is_zero),  # If model is not zero
            np.conj(model) / np.abs(model),  # Normal rotation factor
            1  # Use a default factor of 1
        )
        diff = diff * rotation_factors
        diff = diff.ravel().view(float)

        if weights is not None:
            if np.isscalar(weights): 
                weights = np.full(len(data), weights, dtype=complex)
            if np.iscomplexobj(weights): # in lmfit.model they are using "if weights.dtype is complex" but it returns False for complex128 type
                # weights are complex
                weights = weights.ravel().view(float)
            else:
                # real weights but complex data
                weights = weights.astype(complex).ravel().view(float)
            diff *= weights
        return diff

class ResonatorReflectionModel(PolarWeightModel):
    def __init__(self, independent_vars=['omega'], prefix='', nan_policy='raise', reflection_type='normal', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
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
            elif electrical_delay_estimation == "unwrap overcoupled":
                electrical_delay = estimate_electrical_delay_unwrap(cplx, omega, accumulated_phase=-2*np.pi)
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

def s2y(s, z0=50):
    return (1-s)/(1+s)/z0

def y2s(y, z0=50):
    return (1-y*z0)/(1+y*z0)

def parallel_sparameter(s_func_list):
    """Return parallel sparametr function
    
    Each s_func should have "omega" variable as the first argument.
    The parameter name for each s_func replaced with the one with prefix, like amplitude -> e0_amplitude where e0 means element 0.
    As lmfit.model.Model parse the arguments by using inspect.signature, this function set the __signature__ attribute.
    """
    kwmap_list = []
    new_parameters = [inspect.Parameter("omega", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for k, s_func in enumerate(s_func_list):
        sig = inspect.signature(s_func)
        parameters = sig.parameters
        kwmap = {f"e{k}_{param}": param for param in parameters if param != 'omega'}
        kwmap_list.append(kwmap)
        new_parameters.extend([inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD) for param in kwmap])
    
    def paralell_s_func(omega, **kwargs):
        y_list = []
        for k, s_func in enumerate(s_func_list):
            kwargs_each = {}
            for kw in kwmap_list[k]:
                kw_org = kwmap_list[k][kw]
                kw_val = kwargs.get(kw)
                kwargs_each[kw_org] = kw_val
            s = s_func(omega, **kwargs_each)
            y = s2y(s)
            y_list.append(y)
        y_tot = reduce(operator.add, y_list)  # add up the admittances and convert back to the s parameter to calculate parallelly connected response.
        s_tot = y2s(y_tot)
        return s_tot   
    
    paralell_s_func.__signature__ = inspect.Signature(new_parameters)
    
    return paralell_s_func

def with_cable(s_func):
    def s_func_with_cable(omega, **kwargs):
        a = kwargs.get('a')
        attn_coeff = kwargs.get('attn_coeff')
        tau = kwargs.get('tau')
        theta = kwargs.get('theta')
        return a*cable_attenuation(omega, attn_coeff)*electrical_delay(omega, tau, theta)*s_func(omega, **kwargs)
    sig = inspect.signature(s_func)
    parameters = list(sig.parameters.values())
    parameters.append(inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD))
    parameters.append(inspect.Parameter('attn_coeff', inspect.Parameter.POSITIONAL_OR_KEYWORD))
    parameters.append(inspect.Parameter('tau', inspect.Parameter.POSITIONAL_OR_KEYWORD))
    parameters.append(inspect.Parameter('theta', inspect.Parameter.POSITIONAL_OR_KEYWORD))
    new_sig = inspect.Signature(parameters)
    s_func_with_cable.__signature__ = new_sig
    return s_func_with_cable

class ParallelResonatorReflectionModel(PolarWeightModel):
    def __init__(self, n_resonators, independent_vars=['omega'], prefix='', nan_policy='raise', reflection_type='normal', with_cable_attn_coeff=False, **kwargs):
        """
        Args
            n: number of resonators
        """
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        if reflection_type == 'normal':
            self.reflection_type = reflection_type
            self.reflection_factor = 1
        elif reflection_type == 'hanger':
            self.reflection_type = reflection_type
            self.reflection_factor = 0.5
        else:
            raise ValueError(f"Reflection type '{reflection_type}' is not supprted")
        
        self.with_cable_attn_coeff = with_cable_attn_coeff
        self.n_resonators = n_resonators
        s_func_list = [resonator_reflection_base]*self.n_resonators
        parallel_s_func = parallel_sparameter(s_func_list)
        parallel_s_func_with_cable = with_cable(parallel_s_func)
        super().__init__(parallel_s_func_with_cable, **kwargs)
        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        for k in range(self.n_resonators):
            self.set_param_hint(f'e{k}_kappa_ex', min=0)
            self.set_param_hint(f'e{k}_kappa_in', min=0)
            self.set_param_hint(f'e{k}_reflection_factor', value=self.reflection_factor, vary=False)
        
        self.set_param_hint('a', min=0)
        if self.with_cable_attn_coeff:
            self.set_param_hint('attn_coeff', min=0)
        else:
            self.set_param_hint('attn_coeff', value=0, vary=False)

def resonator_filter_reflection_base(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, phi, reflection_factor=1):
    # parameters:
    # omega : drive freq
    # omega_r : readout resonator freq
    # kappa_in_r : internal decay rate of readout resonator (normally set to zero)
    # omega_p : filter resonator freq
    # kappa_in_p : internal decay rate of filter resonator (normally set to zero)
    # kappa_p : external decay rate of filter resonator
    # J : coupling strength between readout resonator and filter resonator
    # phi: reflection phase offset
    # reflection_factor: 1 for normal reflection, 1/2 for hanger type resonator
    return 1-reflection_factor*np.exp(1j*phi)*kappa_p*(0.5*kappa_in_r + 1j*(omega - omega_r))/((0.5*kappa_p + 0.5*kappa_in_p + 1j*(omega - omega_p)) * (0.5*kappa_in_r + 1j*(omega - omega_r)) + J**2)

def resonator_filter_reflection(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, a, attn_coeff, tau, theta, phi, reflection_factor=1):
    return a*cable_attenuation(omega, attn_coeff)*electrical_delay(omega, tau, theta)*resonator_filter_reflection_base(omega, omega_r, kappa_in_r, omega_p, kappa_in_p, kappa_p, J, phi, reflection_factor)

class ResonatorFilterReflectionModel(PolarWeightModel):
    def __init__(self, independent_vars=['omega'], prefix='', nan_policy='raise', reflection_type='normal', with_cable_attn_coeff=False, **kwargs):
        """
        Args
            n: number of resonators
        """
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        if reflection_type == 'normal':
            self.reflection_type = reflection_type
            self.reflection_factor = 1
        elif reflection_type == 'hanger':
            self.reflection_type = reflection_type
            self.reflection_factor = 0.5
        else:
            raise ValueError(f"Reflection type '{reflection_type}' is not supprted")
        
        self.with_cable_attn_coeff=with_cable_attn_coeff
        super().__init__(resonator_filter_reflection, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('kappa_p', min=0)
        self.set_param_hint('kappa_in_p', min=0)
        self.set_param_hint('kappa_in_r', min=0)
        self.set_param_hint('a', min=0)
        self.set_param_hint('reflection_factor', value=self.reflection_factor, vary=False)
        if self.with_cable_attn_coeff:
            self.set_param_hint('attn_coeff', min=0)
        else:
            self.set_param_hint('attn_coeff', value=0, vary=False)

class ParallelResonatorFilterReflectionModel(PolarWeightModel):
    def __init__(self, n_resonators, independent_vars=['omega'], prefix='', nan_policy='raise', reflection_type='normal', with_cable_attn_coeff=False, **kwargs):
        """
        Args
            n: number of resonators
        """
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        if reflection_type == 'normal':
            self.reflection_type = reflection_type
            self.reflection_factor = 1
        elif reflection_type == 'hanger':
            self.reflection_type = reflection_type
            self.reflection_factor = 0.5
        else:
            raise ValueError(f"Reflection type '{reflection_type}' is not supprted")
        
        self.with_cable_attn_coeff = with_cable_attn_coeff
        self.n_resonators = n_resonators
        s_func_list = [resonator_filter_reflection_base]*self.n_resonators
        parallel_s_func = parallel_sparameter(s_func_list)
        parallel_s_func_with_cable = with_cable(parallel_s_func)
        super().__init__(parallel_s_func_with_cable, **kwargs)
        self._set_paramhints_prefix()
    
    def _set_paramhints_prefix(self):
        for k in range(self.n_resonators):
            self.set_param_hint(f'e{k}_kappa_p', min=0)
            self.set_param_hint(f'e{k}_kappa_in_p', min=0)
            self.set_param_hint(f'e{k}_kappa_in_r', min=0)
            self.set_param_hint(f'e{k}_reflection_factor', value=self.reflection_factor, vary=False)
        
        self.set_param_hint('a', min=0)
        if self.with_cable_attn_coeff:
            self.set_param_hint('attn_coeff', min=0)
        else:
            self.set_param_hint('attn_coeff', value=0, vary=False)

# Composite models
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
    