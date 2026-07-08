using FitterHEP
using Printf

function rosenbrock(p)
    return (1 - p.x)^2 + 100 * (p.y - p.x^2)^2
end

function run_backend(label, backend)
    initial = (x = -1.2, y = 1.0)
    result = if backend isa MinuitBackend
        fit(
            rosenbrock,
            initial;
            backend,
            step_sizes = [0.1, 0.1],
            covariance = :backend,
            errordef = 1.0,
        )
    else
        fit(
            rosenbrock,
            initial;
            backend,
            covariance = :finite_diff,
            errordef = 1.0,
            iterations = 10_000,
        )
    end
    @printf("%-8s converged=%s minimum=%.3g x=%.5f y=%.5f\n",
        label,
        string(result.converged),
        result.minimum,
        result.minimizer.x,
        result.minimizer.y,
    )
    return result
end

function main()
    optim = run_backend("Optim", OptimBackend(:bfgs))
    minuit = run_backend("Minuit", MinuitBackend(strategy = 1, tolerance = 0.1))
    return (; optim, minuit)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
