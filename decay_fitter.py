from lmfit.models import (
    ExponentialModel,
    ConstantModel
)

class decay_fitter(object):
    def __init__(self, negative=False):
        self.model = ExponentialModel() + ConstantModel()
        self.negative = negative
    
    def guess(self, x, data, **kwargs):
        if self.negative:
            c_init = max(data)
        else:
            c_init = min(data)
        
        params = self.model.left.guess(data-c_init, x=x, negative=self.negative, **kwargs)
        params.add('c', value=c_init)
        return params
    
    def fit(self, x, data, params=None, **kwargs):
        if params is None:
            params = self.guess(x, data)
        return self.model.fit(data, params, x=x, **kwargs)
    