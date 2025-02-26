# SQE fitting

[lmfit](https://lmfit.github.io/lmfit-py/)-based fitting codes for superconducting qubit experiments.

## Fitting models

The following are fitting models provided in this module:

* **ResonatorReflectionModel**: Fitting of reflection from resonator (normal reflection or hanger-type resonator)
* **DampedOscillationModel**: Fitting for Rabi or Ramsey
* **ComplexDampedRotationModel**: Fitting for Ramsey with IQ data to identify the sign of detuning
* **ParallelResonatorReflectionModel**: Fitting of reflection from resonators connected in parallel
* **ParallelResonatorFilterReflectionModel**: Fitting of reflection from resonators with individual Purcell filter connected in parallel

The build-in models from lmfit are also customized to have better `guess` functions with constant offset.
The following are CompositeModel with customized guess functions

* **Lorentzian_plus_ConstantModel**: LorentzianModel + ConstantModel
* **Exponential_plus_ConstantModel**: ExponentialModel + ConstantModel
* **DampedOscillation_plus_ConstantModel**: DampedOscillationModel + ConstantModel
* **ComplexDampedRotationModel_plus_ConstantModel**: ComplexDampedRotationModel + ConstantModel

## Installation

```bash
pip install git+pip install git+https://github.com/tama-sh/sqe_fitting.git
```
