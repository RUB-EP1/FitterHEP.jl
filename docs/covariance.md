# Covariance

`FitterHEP.jl` provides two covariance paths.

## Finite-Difference Hessian

```julia
result = fit(
    objective,
    initial;
    backend = OptimBackend(:bfgs),
    covariance = :finite_diff,
    errordef = 1.0,
)
```

The covariance convention is:

```text
covariance ≈ 2 * errordef * inv(H)
```

where `H` is the Hessian of the minimized objective.

Common choices:

- `errordef = 0.5` for negative log likelihoods
- `errordef = 1.0` for chi-square objectives

If direct inversion fails or the Hessian is not positive definite,
`FitterHEP.jl` falls back to a pseudo-inverse in `inversion = :auto`.

## Minuit HESSE

```julia
result = fit(
    objective,
    initial;
    backend = MinuitBackend(),
    covariance = :backend,
)
```

For `MinuitBackend`, `covariance = :backend` runs Minuit HESSE and reads the
backend covariance matrix.

Accepted aliases are:

```julia
covariance = :backend
covariance = :hesse
covariance = :auto
covariance = true
```

## Fixed Parameters

For `OptimBackend`, fixed parameters are removed from the active optimization
vector. Hessian covariance is computed in active space and embedded back into
the full flat parameter space. Rows and columns for fixed parameters are zero.

For `MinuitBackend`, fixed parameters are passed to Minuit directly.
