import numpy as np
from lmfit.models import ConstantModel
from .models import DampedOscillationModel
from .util import percentile_range_data

class decay_osc_fitter(object):
    def __init__(self):
        self.model = DampedOscillationModel() + ConstantModel()
    
    def guess(self, data, x, **kwargs):
        c_init = np.mean(percentile_range_data(data, (0.25, 0.75)))
        
        # make parameters
        params = self.model.left.guess(data-c_init, x=x)
        params.add('c', value=c_init)
        return params
    
    def fit(self, data, x, params=None, **kwargs):
        if params is None:
            params = self.guess(data, x)
        return self.model.fit(data, params, x=x, **kwargs)