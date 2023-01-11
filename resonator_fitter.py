from .models import ResonatorReflectionModel

class resonator_fitter(object):
    def __init__(self, reflection_type='normal'):
        self.model = ResonatorReflectionModel(reflection_type=reflection_type)
    
    def guess(self, cplx, omega, fix_electrical_delay=False, **kwargs):
        pars = self.model.guess(cplx, omega=omega, fix_electrical_delay=fix_electrical_delay, **kwargs)
        return pars
    
    def fit(self, cplx, omega, params=None, fix_electrical_delay=False, **kwargs):
        if params is None:
            params = self.guess(cplx, omega, fix_electrical_delay=False)
        return self.model.fit(cplx, params, omega=omega, **kwargs)
