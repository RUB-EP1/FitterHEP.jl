using FitterHEP
using Printf
using Random

function generate_counting_sample(n::Integer; fraction::Real = 0.35, seed::Integer = 42)
    rng = MersenneTwister(seed)
    return [rand(rng) < fraction for _ in 1:n]
end

function binomial_nll(p, data)
    f = p.fraction
    !(0 < f < 1) && return Inf
    n_success = count(data)
    n_total = length(data)
    return -(n_success * log(f) + (n_total - n_success) * log1p(-f))
end

function main()
    data = generate_counting_sample(1_000)
    objective(p) = binomial_nll(p, data)

    result = fit(
        objective,
        (fraction = 0.5,);
        backend = OptimBackend(:bfgs),
        bounds = ([1e-6], [1 - 1e-6]),
        bound_strategy = :transform,
        covariance = :finite_diff,
        errordef = 0.5,
    )

    @printf("converged=%s fraction=%.5f error(raw)=%.5f\n",
        string(result.converged),
        result.minimizer.fraction,
        errors(result)[1],
    )
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
