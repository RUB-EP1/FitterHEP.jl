# FitterHEP.jl

Universal fitting wrappers for Julia HEP analyses.

`FitterHEP.jl` provides a small, backend-independent interface for fitting
objective functions while keeping optimizer details, covariance estimation, and
diagnostics in one place.

Phase 1 supports vector-valued fits through `Optim.jl` and finite-difference
Hessian covariance estimates.

```julia
using FitterHEP

objective(x) = (x[1] - 1.5)^2 + (x[2] + 2.0)^2

result = fit(
    objective,
    [0.0, 0.0];
    backend = OptimBackend(:bfgs),
    covariance = :finite_diff,
    errordef = 1.0,
)

result.minimizer
errors(result)
correlation(result)
```

## Current API

- `fit(objective, initial; backend=OptimBackend(:lbfgs), ...)`
- `OptimBackend(:lbfgs)`, `OptimBackend(:bfgs)`, `OptimBackend(:nelder_mead)`
- `hesse(objective, point; method=:finite_diff, errordef=0.5)`
- `covariance(result)`, `errors(result)`, `correlation(result)`

## Roadmap

- Minuit2 backend with an Optim-like interface
- fixed parameters, bounds, and step sizes
- support for `NamedTuple` and `ComponentArray` parameters
- backend-native and fallback HESSE/covariance paths
- X2VV-style benchmark examples
