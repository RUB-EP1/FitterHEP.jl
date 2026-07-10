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
    step_sizes = nothing,
    covariance = :finite_diff,
)
```

`step_sizes` are parameter scales in flat parameter order. If omitted,
FitterHEP uses `max(0.1 * abs(x0), 0.1)` for each parameter. For
`OptimBackend(:bfgs)`, these scales define the diagonal initial inverse Hessian,
`Diagonal(step_sizes .^ 2)`. For `OptimBackend(:lbfgs)`, they define a diagonal
Hessian preconditioner, `Diagonal(1 ./ step_sizes .^ 2)`. Custom Optim method
objects that already define an initial inverse Hessian or preconditioner are
left unchanged.

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
    fixed = nothing,
    bounds = nothing,
    step_sizes = nothing,
    covariance = :backend,
)
```

For `MinuitBackend`, `step_sizes` are passed as the initial Minuit parameter
errors. The same default `max(0.1 * abs(x0), 0.1)` is used when they are omitted.

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
