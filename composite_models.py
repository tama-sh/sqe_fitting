import numpy as np
import operator
import lmfit
from lmfit.models import (LorentzianModel,
                          ExponentialModel,
                          ConstantModel)
from .models import DampedOscillationModel
from .util import percentile_range_data

def guess_peak_or_dip(data):
    """ Guess data has a peak or a dip
        Args:
            data (np.ndarray): data
        Return:
            bool: True if data has a peak
    """
    return np.median(data) < (np.min(data) + np.ptp(data)/2)

def guess_linewidth_from_peak(freq, data, r=2):
    """ Estimate line width sigma (half of FWHM) from peak, for LorentzianModel
    
        Args:
            freq (np.ndarray): frequency
            data (np.ndarray): data with peak of dip (peak, dip is automatically estimated)
            r (float): Theshold value to estimate the peak. The peak width is estimated from the range of data where data > (1-1/r)*max(data)
            
        Return:
            float: sigma (half of FWHM)
    """
    length = len(data)
    idx_c = np.argmax(data[1:-1]) + 1 # avoid the peak placing at borders
    
    ptp = np.ptp(data)
    max_val = data[idx_c]
    cond = data > (max_val - ptp/r)

    i = 0
    while (idx_c+i+1 < (length-1) and cond[idx_c+i+1]):
        i += 1
    j = 0
    while (idx_c-(j+1) > 0 and cond[idx_c-(j+1)]):
        j += 1
    idx_l = idx_c - j
    idx_r = idx_c + i
    
    # redefine r parameter
    r = ptp/(data[idx_c]-0.5*(data[idx_l-1]+data[idx_r+1]))
    return np.sqrt(r-1)*(freq[idx_r+1] - freq[idx_l-1])/2

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
    
    def guess(self, data, x, negative=False, **kwargs):
        if negative:
            c_init = max(data)
        else:
            c_init = min(data)
        
        params = self.left.guess(data-c_init, x=x, negative=negative, **kwargs)
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
    