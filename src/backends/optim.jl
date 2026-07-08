struct OptimBackend{M} <: AbstractFitBackend
    method::M
end

OptimBackend() = OptimBackend(:lbfgs)

function OptimBackend(method::Symbol)
    method === :lbfgs && return OptimBackend(LBFGS())
    method === :bfgs && return OptimBackend(BFGS())
    method === :nelder_mead && return OptimBackend(NelderMead())
    method === :neldermead && return OptimBackend(NelderMead())
    throw(ArgumentError("unsupported Optim method: $method"))
end

function _optim_options(; iterations::Integer = 10_000, kwargs...)
    opts = Dict{Symbol, Any}(:iterations => iterations)
    for (key, value) in kwargs
        value === nothing && continue
        opts[key] = value
    end
    return Optim.Options(; opts...)
end

function _fit(
    backend::OptimBackend,
    objective,
    initial;
    iterations::Integer = 10_000,
    autodiff = nothing,
    covariance = :none,
    errordef::Real = 0.5,
    inversion::Symbol = :auto,
    g_tol = nothing,
    x_tol = nothing,
    f_tol = nothing,
    x_abstol = nothing,
    f_abstol = nothing,
    x_reltol = nothing,
    f_reltol = nothing,
    show_trace::Bool = false,
    callback = nothing,
)
    adapter = _parameter_adapter(initial)
    x0 = adapter.x0
    objective_flat(x) = objective(adapter.rebuild(x))
    options = _optim_options(;
        iterations,
        g_tol,
        x_tol,
        f_tol,
        x_abstol,
        f_abstol,
        x_reltol,
        f_reltol,
        show_trace,
        callback,
    )

    result = if autodiff === nothing
        optimize(objective_flat, x0, backend.method, options)
    else
        optimize(objective_flat, x0, backend.method, options; autodiff)
    end
    xmin = collect(Float64, Optim.minimizer(result))
    fmin = Float64(Optim.minimum(result))
    hess, cov, err, corr, valid_cov = _maybe_covariance(
        objective_flat,
        xmin;
        covariance_method = covariance,
        errordef,
        inversion,
    )
    diagnostics = FitDiagnostics(
        status = Optim.converged(result) ? :converged : :not_converged,
        message = "",
        iterations = try Optim.iterations(result) catch; nothing end,
        nfcn = try result.f_calls catch; nothing end,
        valid_covariance = valid_cov,
        backend_status = result,
    )
    return FitResult(backend, adapter.rebuild(xmin), fmin, Optim.converged(result), cov, err, corr, result, diagnostics)
end
