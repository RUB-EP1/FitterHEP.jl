# FitterHEP.jl Documentation

`FitterHEP.jl` is a small wrapper layer for fitting objective functions with
interchangeable Julia optimizer backends.

The package currently focuses on:

- one `fit` interface for `Optim.jl` and `Minuit2.jl`
- common result and diagnostic objects
- finite-difference and backend covariance paths
- vector, `NamedTuple`, and `ComponentArray` parameter inputs
- bounds and fixed-parameter handling

## Pages

- [API](api.md)
- [Parameters](parameters.md)
- [Covariance](covariance.md)
- [Examples](examples.md)
- [Limitations](limitations.md)
