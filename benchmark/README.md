# Fit-Only Backend Comparisons

These scripts compare minimization behavior only. Covariance/HESSE timing and
quality are intentionally left for a separate comparison.

Run from the package root:

```sh
julia --project=. benchmark/quadratic.jl
julia --project=. benchmark/rosenbrock.jl
julia --project=. benchmark/bounded_fraction_toys.jl
julia --project=. benchmark/massfit_toys.jl
julia --project=. benchmark/make_tables.jl
```

Each script writes CSV files under `benchmark/results/` by default. Pass a first
argument to choose another output directory:

```sh
julia --project=. benchmark/massfit_toys.jl /tmp/fitterhep_bench 50
```

## Compared Backends

- `OptimBackend(:bfgs)`
- `OptimBackend(:lbfgs)`
- `OptimBackend(:nelder_mead)`
- `MinuitBackend(strategy = 1, tolerance = 0.1)`
- `MinuitBackend(strategy = 0, tolerance = 0.2)`

## Recorded Quantities

- convergence flag
- minimum objective value
- fitted flat parameters
- parameter difference from known truth, when available
- runtime in milliseconds
- backend-reported function calls, when available
- backend-reported iterations, when available
- backend status

`common.jl` warms each fit once before timing so the tables focus on fit runtime
rather than first-call compilation.

## Scripts

- `quadratic.jl`: analytic convex reference with known optimum.
- `rosenbrock.jl`: nonlinear banana-shaped minimization test.
- `bounded_fraction_toys.jl`: bounded one-parameter binomial likelihood toys.
- `massfit_toys.jl`: simple Gaussian-plus-exponential unbinned mass-fit toys.
