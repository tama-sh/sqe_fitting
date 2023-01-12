from .models import ResonatorReflectionModel

class resonator_fitter(object):
    def __init__(self, reflection_type='normal'):
        self.model = ResonatorReflectionModel(reflection_type=reflection_type)
    
    def guess(self, cplx, omega, fix_electrical_delay=True, electrical_delay_estimation="default", **kwargs):
        pars = self.model.guess(cplx, omega=omega, fix_electrical_delay=fix_electrical_delay, electrical_delay_estimation=electrical_delay_estimation, **kwargs)
        return pars
    
    def fit(self, cplx, omega, params=None, fix_electrical_delay=True, electrical_delay_estimation="default", **kwargs):
        if params is None:
            params = self.guess(cplx, omega, fix_electrical_delay=fix_electrical_delay, electrical_delay_estimation=electrical_delay_estimation)
        return self.model.fit(cplx, params, omega=omega, **kwargs)
