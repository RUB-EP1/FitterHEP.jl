# API

## Main Entry Point

```julia
result = fit(objective, initial; backend = OptimBackend(:lbfgs), kwargs...)
```

`objective` is a callable that receives parameters in the same shape as
`initial`. Internally, `FitterHEP.jl` flattens parameters for the optimizer and
rebuilds them before calling the objective.

## Backends

### OptimBackend

```julia
OptimBackend(:lbfgs)
OptimBackend(:bfgs)
OptimBackend(:nelder_mead)
```

Common options:

```julia
fit(
    objective,
    initial;
    backend = OptimBackend(:bfgs),
    iterations = 10_000,
    autodiff = :forward,
    bounds = nothing,
    fixed = nothing,
    bound_strategy = :native,
    covariance = :finite_diff,
)
```

When bounds are supplied and `bound_strategy = :native`, `Optim.Fminbox` is
used. With `bound_strategy = :transform`, unconstrained raw parameters are
mapped into the bounded physical space.

### MinuitBackend

```julia
MinuitBackend(;
    strategy = 1,
    tolerance = 0.1,
    ncall = 0,
    iterate = 5,
    use_simplex = true,
    errordef = 0.5,
)
```

Common options:

```julia
fit(
    objective,
    initial;
    backend = MinuitBackend(),
    names = nothing,
    step_sizes = nothing,
    fixed = nothing,
    bounds = nothing,
    covariance = :backend,
)
```

`covariance = :backend` runs Minuit HESSE after MIGRAD.

## Results

`fit` returns a `FitResult`:

```julia
result.minimizer
result.minimum
result.converged
result.diagnostics
```

Convenience accessors:

```julia
covariance(result)
errors(result)
correlation(result)
```

`result.raw_result` stores the backend-native result object.
