# Limitations

Current limitations are intentional and keep the package small while the API
settles.

## Direct Dependencies

`Minuit2` and `ComponentArrays` are direct dependencies. A future version may
move these integrations into package extensions so the base package is lighter.

## Flat Option Order

`bounds`, `fixed`, and `step_sizes` are specified in flat parameter order. This
is simple and backend-neutral, but not yet as ergonomic as named constraints.

## Gradients

`MinuitBackend` accepts a raw Minuit-style `grad` keyword, but gradients are not
yet adapted from structured parameter shapes. In practice, use gradient-free
Minuit first, or pass a backend-native gradient knowingly.

## Covariance With Transformed Bounds

For `OptimBackend(bound_strategy = :transform)`, finite-difference covariance is
computed in the active raw parameter space and embedded in flat order. It is not
yet transformed to physical parameter covariance.

## Error Propagation For Structured Parameters

`covariance(result)`, `errors(result)`, and `correlation(result)` currently
return flat arrays. They are not rebuilt into the shape of a `NamedTuple` or
`ComponentArray`.
