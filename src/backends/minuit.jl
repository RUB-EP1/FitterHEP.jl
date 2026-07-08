Base.@kwdef struct MinuitBackend <: AbstractFitBackend
    strategy::Int = 1
    tolerance::Float64 = 0.1
    ncall::Int = 0
    iterate::Int = 5
    use_simplex::Bool = true
    errordef::Float64 = 0.5
end

function _default_names(n::Integer)
    return ["p$i" for i in 1:n]
end

function _default_step_sizes(x0::AbstractVector)
    return [max(0.1 * abs(x), 0.1) for x in x0]
end

function _normalize_vector_option(value, n::Integer, name::AbstractString)
    value === nothing && return nothing
    length(value) == n || throw(ArgumentError("$name must have length $n"))
    return collect(value)
end

function _normalize_names(names, n::Integer)
    values = names === nothing ? _default_names(n) : collect(names)
    length(values) == n || throw(ArgumentError("names must have length $n"))
    return string.(values)
end

function _normalize_bounds(bounds, n::Integer)
    bounds === nothing && return nothing
    if bounds isa Tuple && length(bounds) == 2 && all(x -> x isa AbstractVector, bounds)
        lower, upper = bounds
        length(lower) == n || throw(ArgumentError("lower bounds must have length $n"))
        length(upper) == n || throw(ArgumentError("upper bounds must have length $n"))
        return collect(zip(lower, upper))
    end
    length(bounds) == n || throw(ArgumentError("bounds must have length $n"))
    return collect(bounds)
end

function _minuit_status(m)
    m.is_valid && return :converged
    m.has_reached_call_limit && return :call_limit
    m.is_above_max_edm && return :above_max_edm
    return :not_converged
end

function _minuit_covariance(m, strategy::Integer)
    hesse!(m; strategy)
    C = matrix(m)
    s = _covariance_summary(C)
    return (s.covariance, s.errors, s.correlation, true)
end

function _fit(
    backend::MinuitBackend,
    objective,
    initial;
    names = nothing,
    step_sizes = nothing,
    fixed = nothing,
    bounds = nothing,
    covariance = :none,
    errordef = nothing,
    grad = nothing,
)
    x0 = collect(Float64, initial)
    n = length(x0)
    param_names = _normalize_names(names, n)
    errors0 = something(_normalize_vector_option(step_sizes, n, "step_sizes"), _default_step_sizes(x0))
    fixed_parameters = something(_normalize_vector_option(fixed, n, "fixed"), fill(false, n))
    limits = _normalize_bounds(bounds, n)

    f_array(raw) = objective(collect(raw))
    active_errordef = errordef === nothing ? backend.errordef : Float64(errordef)
    result_backend = MinuitBackend(
        strategy = backend.strategy,
        tolerance = backend.tolerance,
        ncall = backend.ncall,
        iterate = backend.iterate,
        use_simplex = backend.use_simplex,
        errordef = active_errordef,
    )
    minuit_kwargs = Dict{Symbol, Any}(
        :error => errors0,
        :names => param_names,
        :fixed => Bool.(fixed_parameters),
        :errordef => active_errordef,
        :tolerance => backend.tolerance,
        :arraycall => true,
    )
    limits === nothing || (minuit_kwargs[:limits] = limits)
    grad === nothing || (minuit_kwargs[:grad] = grad)

    m = Minuit(f_array, x0; minuit_kwargs...)
    migrad!(
        m,
        backend.strategy;
        ncall = backend.ncall,
        iterate = backend.iterate,
        use_simplex = backend.use_simplex,
    )

    xmin = collect(Float64, m.values)
    fmin = Float64(m.fval)
    cov = nothing
    err = nothing
    corr = nothing
    valid_cov = false
    if covariance in (:backend, :hesse, :auto, true)
        try
            cov, err, corr, valid_cov = _minuit_covariance(m, backend.strategy)
        catch
            valid_cov = false
        end
    elseif !(covariance in (:none, nothing, false))
        throw(ArgumentError("MinuitBackend supports covariance=:backend, :hesse, :auto, true, or :none"))
    end

    diagnostics = FitDiagnostics(
        status = _minuit_status(m),
        message = "",
        iterations = try m.niter catch; nothing end,
        nfcn = try m.nfcn catch; nothing end,
        edm = try Float64(m.edm) catch; nothing end,
        reached_call_limit = try m.has_reached_call_limit catch; nothing end,
        valid_covariance = valid_cov,
        backend_status = m,
    )
    return FitResult(result_backend, xmin, fmin, m.is_valid, cov, err, corr, m, diagnostics)
end
