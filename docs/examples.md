# Examples

Run examples from the package root:

```sh
julia --project=. examples/switch_backends.jl
julia --project=. examples/bounded_fraction_fit.jl
julia --project=. examples/fixed_parameter_fit.jl
julia --project=. examples/massfit_nll.jl
```

## Backend Switching

`examples/switch_backends.jl`

Fits the same structured Rosenbrock objective with:

- `OptimBackend(:bfgs)`
- `MinuitBackend()`

This demonstrates the main purpose of the package: switch optimizer backends
without rewriting the objective.

## Bounded Fraction Fit

`examples/bounded_fraction_fit.jl`

Fits a one-parameter binomial negative log likelihood with:

```julia
bounds = ([1e-6], [1 - 1e-6])
bound_strategy = :transform
```

## Fixed Parameter Fit

`examples/fixed_parameter_fit.jl`

Fits a small line-model chi-square while keeping the intercept fixed:

```julia
fixed = [true, false]
```

## Toy Mass Fit

`examples/massfit_nll.jl`

Generates a toy mass sample and fits a Gaussian signal plus exponential
background mixture with a hand-written NLL.
