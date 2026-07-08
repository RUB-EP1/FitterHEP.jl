# Parameters

`FitterHEP.jl` supports three parameter input shapes.

## Vector

```julia
objective(x) = (x[1] - 1)^2 + (x[2] + 2)^2
result = fit(objective, [0.0, 0.0]; backend = OptimBackend(:bfgs))
```

The result minimizer is a `Vector{Float64}`.

## NamedTuple

```julia
initial = (signal = (mu = 0.0, sigma = 1.0), background = (slope = 0.0,))

objective(p) = (p.signal.mu - 5.28)^2 + (p.background.slope + 1.0)^2

result = fit(objective, initial; backend = OptimBackend(:bfgs))
```

The result minimizer has the same nested `NamedTuple` shape:

```julia
result.minimizer.signal.mu
```

For `MinuitBackend`, generated default names follow dotted paths such as
`signal.mu`.

## ComponentArray

```julia
using ComponentArrays

initial = ComponentArray(signal = [0.0, 1.0], background = (slope = 0.0,))
objective(p) = (p.signal[1] - 1)^2 + (p.background.slope + 0.5)^2

result = fit(objective, initial; backend = OptimBackend(:bfgs))
```

The result minimizer is rebuilt as a `ComponentArray` with the original axes.

## Flat Option Order

Options such as `bounds`, `fixed`, and `step_sizes` currently use the internal
flat parameter order.

For nested `NamedTuple`s, this order is depth-first in field order:

```julia
initial = (signal = (mu = 0.0, sigma = 1.0), background = (slope = 0.0,))
```

Flat order:

```text
signal.mu
signal.sigma
background.slope
```

So:

```julia
fixed = [false, false, true]
bounds = ([5.0, 0.001, -10.0], [6.0, 1.0, 10.0])
```
