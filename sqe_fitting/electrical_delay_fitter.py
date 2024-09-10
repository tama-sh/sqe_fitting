from typing import Tuple

import numpy as np
import lmfit
from .util import percentile_range_data
from .signal_util import group_delay, smoothen
from .circle_fitter import algebric_circle_fit

def correct_electrical_delay(cplx: np.ndarray, omega: np.ndarray, electrical_delay=None, phase_offset=None, phase_auto_correct=False):
    """Correct electrical delay of complex data
    
    Args:
        cplx (np.ndarray): complex data with electrical delay
        omega (np.ndarray): angular frequency
        electrical_delay (bool): the value of electrical_delay. if None, automatically estimate the electrical delay from group delay. default is None
        phase_offset (float): phase offset to be corrected. if not None, the phase offset is corrected with this value. default is None.
        phase_auto_correct (bool): if True, automatically correct phase offset as well. if phase_offset is given, this parameter is not active. default is False
    
    Returns:
        np.ndarray: complex data with electrical delay corrected
    """
    if electrical_delay is None:
        electrical_delay = estimate_electrical_delay_from_group_delay(cplx, omega)
    cplx_c = cplx*np.exp(1j*electrical_delay*omega)
    if phase_offset:
        return cplx_c*np.exp(-1j*phase_offset)
    elif phase_auto_correct:
        return cplx_c*np.exp(-1j*np.angle(cplx_c[0]+cplx_c[-1]))
    else:
        return cplx_c

def estimate_electrical_delay_unwrap(cplx: np.ndarray, omega: np.ndarray, accumulated_phase=0):
    """Estimate electrical delay by unwraping the phase
    
    Args:
        omega (np.ndarray): angular frequency
        cplx (np.ndarray): complex data with electrical delay
        accumulated_phase: expected accumulated phase. default is 0 (if there are n overcoupled resonator, this value should be -2*np.pi*n)
    
    Returns:
        float: electrical delay
        float: phase offset
    """
    phase = np.unwrap(np.angle(cplx))
    electrical_delay = -(phase[-1]-phase[0]-accumulated_phase)/(omega[-1]-omega[0])
    return electrical_delay

def estimate_electrical_delay_from_group_delay(cplx: np.ndarray, omega: np.ndarray, percentile_range: Tuple[float, float] = (0, 0.5)):
    """Estimate electridal delay from the group delay
    
    Args:
        cplx (np.ndarray): complex data with electrical delay
        omega (np.ndarray): angular frequency
        percentile_range (Tupple[float, float]): range of percentile used for estimation, default is (0, 0.5) to neglect the peak caused by resonance

    Returns:
        float: electrical delay
    """
    delay = group_delay(cplx, omega)
    return np.mean(percentile_range_data(delay, percentile_range))

def estimate_electrical_delay_circle_fit(cplx: np.ndarray, omega: np.ndarray, electrical_delay_init=0, return_minimizer_result=False):
    """Estimate electridal delay from algebric circle fit
    
    Args:
        cplx (np.ndarray): complex data with electrical delay
        omega (np.ndarray): angular frequency
        electrical_delay_init (float): initial guess of electrical delay. default is 0
        return_minimizer_result (bool): if True, the function return lmfit.MinimizerResult otherwise return the value of electrical delay. default is False
        
    Return:
        float or lmfit.MinimizerResult: electrical delay or result of fitting
    """
    def residual(pars: lmfit.Parameters, cplx: np.ndarray, omega: np.ndarray):
        """Caldulate residual of algebric circle fit, with electrical delay as a parameter
    
        Args:
            pars (lmfit.Parameters): Parameters which has an electrical delay
            cplx (np.ndarray): complex data
            omega (np.ndarray): angular frequency
        
        Returns:
            np.ndarray: residual
        """
        parvals = pars.valuesdict()
        electrical_delay = parvals['electrical_delay']
        
        cplx_c = cplx*np.exp(1j*omega*electrical_delay)
        x = cplx_c.real
        y = cplx_c.imag
        rst = algebric_circle_fit(x, y)
        return rst.residual
    
    pars = lmfit.Parameters()
    pars.add('electrical_delay', value=electrical_delay_init)
    
    fitter = lmfit.Minimizer(residual, pars, fcn_args=(cplx, omega))
    rst = fitter.minimize()
    
    if return_minimizer_result:
        return rst
    else: 
        return rst.params['electrical_delay'].value
    
def estimate_electrical_delay_resonator(cplx: np.ndarray, omega: np.ndarray):
    """Estimate electridal delay for resonator data
    
    Use estimate_electrical_delay_unwrap to give an initial guess and use estimate_electrical_delay_circle_fit after that
    
    Args:
        cplx (np.ndarray): complex data with electrical delay
        omega (np.ndarray): angular frequency
        
    Returns:
        float: electrical delay
    """
    
    electrical_delay_init = estimate_electrical_delay_unwrap(cplx, omega, accumulated_phase=-2*np.pi)
    return estimate_electrical_delay_circle_fit(cplx, omega, electrical_delay_init=electrical_delay_init)