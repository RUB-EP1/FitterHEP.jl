using FitterHEP
using Printf

function line_chi2(p)
    x = 0:4
    y = [1.0, 2.1, 2.9, 4.2, 5.1]
    σ = fill(0.2, length(y))
    return sum(((p.intercept .+ p.slope .* x .- y) ./ σ) .^ 2)
end

function main()
    result = fit(
        line_chi2,
        (intercept = 0.0, slope = 1.0);
        backend = OptimBackend(:bfgs),
        fixed = [true, false],
        covariance = :finite_diff,
        errordef = 1.0,
    )

    @printf("converged=%s intercept=%.3f slope=%.3f slope_error=%.3f\n",
        string(result.converged),
        result.minimizer.intercept,
        result.minimizer.slope,
        errors(result)[2],
    )
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
