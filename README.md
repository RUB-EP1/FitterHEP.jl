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

The same objective can be minimized with Minuit/MIGRAD:

```julia
result = fit(
    objective,
    [0.0, 0.0];
    backend = MinuitBackend(strategy = 1, tolerance = 0.1),
    names = [:x, :y],
    step_sizes = [0.1, 0.1],
    covariance = :backend,
    errordef = 1.0,
)
```

Initial parameters can also be structured. The optimizer still sees a flat
vector internally, while the objective and result use the original shape:

```julia
initial = (signal = (mu = 0.0, sigma = 1.0), background = (slope = 0.0,))
objective(p) = (p.signal.mu - 1.5)^2 + (p.background.slope + 0.2)^2

result = fit(objective, initial; backend = OptimBackend(:bfgs))
result.minimizer.signal.mu
```

Bounds and fixed parameters use flat parameter order for now:

```julia
result = fit(
    objective,
    initial;
    backend = OptimBackend(:bfgs),
    bounds = ([0.0, -Inf, -10.0], [10.0, Inf, 10.0]),
    fixed = [false, true, false],
)
```

For `OptimBackend`, bounds are handled with `Fminbox` by default. Set
`bound_strategy = :transform` to optimize unconstrained raw parameters that are
mapped into the bounded physical space.

## Current API

- `fit(objective, initial; backend=OptimBackend(:lbfgs), ...)`
- `OptimBackend(:lbfgs)`, `OptimBackend(:bfgs)`, `OptimBackend(:nelder_mead)`
- `MinuitBackend(; strategy=1, tolerance=0.1, ncall=0, iterate=5)`
- `hesse(objective, point; method=:finite_diff, errordef=0.5)`
- `covariance(result)`, `errors(result)`, `correlation(result)`

See [`docs/`](docs/README.md) for API details, parameter conventions,
covariance conventions, examples, and current limitations.

## Examples

Run the toy examples:

```sh
julia --project=. examples/switch_backends.jl
julia --project=. examples/bounded_fraction_fit.jl
julia --project=. examples/fixed_parameter_fit.jl
julia --project=. examples/massfit_nll.jl
```

- `switch_backends.jl`: same structured objective with `OptimBackend` and
  `MinuitBackend`.
- `bounded_fraction_fit.jl`: bounded one-parameter likelihood fit with
  `bound_strategy = :transform`.
- `fixed_parameter_fit.jl`: chi-square line fit with one fixed parameter.
- `massfit_nll.jl`: Gaussian signal plus exponential background mixture with a
  hand-written NLL and finite-difference covariance estimation.

## Roadmap

- Minuit2 backend with an Optim-like interface
- fallback HESSE/covariance paths
- X2VV-style benchmark examples
