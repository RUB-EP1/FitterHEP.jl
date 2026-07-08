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
    fixed = nothing,
    bounds = nothing,
    bound_strategy::Symbol = :native,
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
    problem = _active_parameter_problem(x0; fixed, bounds, bound_strategy)
    objective_active(z) = objective(adapter.rebuild(problem.expand(z)))
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

    if isempty(problem.active)
        xmin = x0
        fmin = Float64(objective(adapter.rebuild(xmin)))
        diagnostics = FitDiagnostics(status = :converged, message = "all parameters are fixed", iterations = 0, nfcn = 1)
        return FitResult(backend, adapter.rebuild(xmin), fmin, true, nothing, nothing, nothing, nothing, diagnostics)
    end

    result = if problem.has_bounds
        if autodiff === nothing
            optimize(objective_active, problem.zlower, problem.zupper, problem.z0, Fminbox(backend.method), options)
        else
            optimize(objective_active, problem.zlower, problem.zupper, problem.z0, Fminbox(backend.method), options; autodiff)
        end
    elseif autodiff === nothing
        optimize(objective_active, problem.z0, backend.method, options)
    else
        optimize(objective_active, problem.z0, backend.method, options; autodiff)
    end
    zmin = collect(Float64, Optim.minimizer(result))
    xmin = problem.expand(zmin)
    fmin = Float64(Optim.minimum(result))
    hess, active_cov, active_err, active_corr, valid_cov = _maybe_covariance(
        objective_active,
        zmin;
        covariance_method = covariance,
        errordef,
        inversion,
    )
    cov = _embed_active_matrix(active_cov, problem.active, length(x0))
    err = cov === nothing ? nothing : _errors_from_covariance(cov)
    corr = cov === nothing ? nothing : _correlation_from_covariance(cov)
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
