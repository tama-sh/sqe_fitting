import numpy as np
from lmfit.models import (
    LorentzianModel,
    ConstantModel
)

class lorentzian_fitter(object):
    def __init__(self, negative=None):
        self.model = LorentzianModel() + ConstantModel()
        self.negative = negative
    
    def guess(self, data, x, **kwargs):
        if self.negative is None:
            negative = not guess_peak_or_dip(data)
            
        if negative:
            c_init = max(data)
            data_peak = -data
        else:
            c_init = min(data)
            data_peak = data

        # # use guess function of LorentzianModel
        # params = self.model.left.guess(data-c_init, x=x, negative=self.negative, **kwargs)

        sigma = guess_linewidth_from_peak(x, data_peak)
        idx_c = np.argmax(data_peak)
        mu = x[idx_c]
        A = np.pi*sigma*data[idx_c]
        
        params = self.model.make_params()
        params['amplitude'].set(value=A)
        params['center'].set(value=mu)
        params['sigma'].set(value=sigma)
        params['c'].set(value=c_init)
        
        return params
    
    def fit(self, data, x, params=None, **kwargs):
        if params is None:
            params = self.guess(data, x)
        return self.model.fit(data, params, x=x, **kwargs)

def guess_peak_or_dip(data):
    """ Guess data has a peak or a dip
        Args:
            data (np.ndarray): data
        Return:
            bool: True if data has a peak
    """
    return np.median(data) < (np.min(data) + np.ptp(data)/2)

def guess_linewidth_from_peak(freq, data, r=2):
    """ Estimate line width sigma (half of FWHM) from peak
    
        Args:
            freq (np.ndarray): frequency
            data (np.ndarray): data with peak of dip (peak, dip is automatically estimated)
            r (float): Theshold value to estimate the peak. The peak width is estimated from the range of data where data > (1-1/r)*max(data)
            
        Return:
            float: sigma (half of FWHM)
    """
    ptp = np.ptp(data)
    idx_c = np.argmax(data)
    max_val = data[idx_c]
    cond = data > (max_val - ptp/r)

    length = len(data)
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