struct ParameterAdapter{F,N}
    x0::Vector{Float64}
    rebuild::F
    names::N
end

function _parameter_adapter(initial::AbstractVector)
    x0 = collect(Float64, initial)
    return ParameterAdapter(x0, x -> collect(Float64, x), _default_names(length(x0)))
end

function _parameter_adapter(initial::NamedTuple)
    values = Float64[]
    names = String[]
    _flatten_namedtuple!(values, names, (), initial)
    rebuild = x -> first(_rebuild_namedtuple(initial, collect(Float64, x), 1))
    return ParameterAdapter(values, rebuild, names)
end

function _parameter_adapter(initial::ComponentArray)
    x0 = collect(Float64, getdata(initial))
    axes = getaxes(initial)
    rebuild = x -> ComponentArray(collect(Float64, x), axes)
    return ParameterAdapter(x0, rebuild, _default_names(length(x0)))
end

function _flatten_namedtuple!(values::Vector{Float64}, names::Vector{String}, prefix::Tuple, nt::NamedTuple)
    for key in keys(nt)
        value = getproperty(nt, key)
        path = (prefix..., key)
        if value isa Real
            push!(values, Float64(value))
            push!(names, join(string.(path), "."))
        elseif value isa NamedTuple
            _flatten_namedtuple!(values, names, path, value)
        else
            throw(ArgumentError("NamedTuple parameters must contain only Real values or nested NamedTuples; got $(typeof(value)) at $(join(string.(path), "."))"))
        end
    end
    return values
end

function _rebuild_namedtuple(template::NamedTuple, values::AbstractVector{<:Real}, index::Integer)
    pairs = Pair{Symbol, Any}[]
    next_index = index
    for key in keys(template)
        value = getproperty(template, key)
        if value isa Real
            push!(pairs, key => Float64(values[next_index]))
            next_index += 1
        elseif value isa NamedTuple
            nested, next_index = _rebuild_namedtuple(value, values, next_index)
            push!(pairs, key => nested)
        else
            throw(ArgumentError("unsupported NamedTuple parameter value type: $(typeof(value))"))
        end
    end
    return (; pairs...), next_index
end

function _normalize_vector_option(value, n::Integer, name::AbstractString)
    value === nothing && return nothing
    length(value) == n || throw(ArgumentError("$name must have length $n"))
    return collect(value)
end

function _default_step_sizes(x0::AbstractVector)
    return [max(0.1 * abs(x), 0.1) for x in x0]
end

function _normalize_step_sizes(step_sizes, x0::AbstractVector)
    values = something(_normalize_vector_option(step_sizes, length(x0), "step_sizes"), _default_step_sizes(x0))
    out = Float64.(values)
    all(x -> isfinite(x) && x > 0, out) || throw(ArgumentError("step_sizes must contain positive finite values"))
    return out
end

function _normalize_fixed(fixed, n::Integer)
    fixed === nothing && return fill(false, n)
    values = _normalize_vector_option(fixed, n, "fixed")
    return Bool.(values)
end

function _normalize_bounds(bounds, n::Integer)
    if bounds === nothing
        return fill(-Inf, n), fill(Inf, n)
    end
    if bounds isa Tuple && length(bounds) == 2 && all(x -> x isa AbstractVector, bounds)
        lower, upper = bounds
        length(lower) == n || throw(ArgumentError("lower bounds must have length $n"))
        length(upper) == n || throw(ArgumentError("upper bounds must have length $n"))
        return Float64.(lower), Float64.(upper)
    end
    length(bounds) == n || throw(ArgumentError("bounds must have length $n"))
    lower = Float64[first(b) for b in bounds]
    upper = Float64[last(b) for b in bounds]
    return lower, upper
end

function _validate_bounds(x0::AbstractVector, lower::AbstractVector, upper::AbstractVector)
    for i in eachindex(x0)
        lower[i] <= upper[i] || throw(ArgumentError("lower bound exceeds upper bound for parameter $i"))
        lower[i] <= x0[i] <= upper[i] || throw(ArgumentError("initial parameter $i is outside its bounds"))
    end
    return nothing
end

function _forward_transform(raw::Real, lower::Real, upper::Real)
    isfinite(lower) && isfinite(upper) && return lower + (upper - lower) / (1 + exp(-raw))
    isfinite(lower) && return lower + log1p(exp(-abs(raw))) + max(raw, 0)
    isfinite(upper) && return upper - (log1p(exp(-abs(raw))) + max(raw, 0))
    return Float64(raw)
end

function _inverse_transform(x::Real, lower::Real, upper::Real)
    if isfinite(lower) && isfinite(upper)
        y = clamp((x - lower) / (upper - lower), eps(Float64), 1 - eps(Float64))
        return log(y / (1 - y))
    elseif isfinite(lower)
        y = max(x - lower, eps(Float64))
        y > 40 && return y
        return log(expm1(y))
    elseif isfinite(upper)
        y = max(upper - x, eps(Float64))
        y > 40 && return -y
        return -log(expm1(y))
    end
    return Float64(x)
end

function _active_parameter_problem(x0::AbstractVector; fixed = nothing, bounds = nothing, bound_strategy::Symbol = :native)
    fixed_mask = _normalize_fixed(fixed, length(x0))
    lower, upper = _normalize_bounds(bounds, length(x0))
    _validate_bounds(x0, lower, upper)
    active = findall(!, fixed_mask)
    bound_strategy in (:native, :transform) || throw(ArgumentError("unsupported bound_strategy: $bound_strategy"))

    if bound_strategy === :native
        z0 = x0[active]
        zlower = lower[active]
        zupper = upper[active]
        expand = z -> begin
            x = copy(x0)
            x[active] .= z
            x
        end
        has_bounds = any(isfinite, zlower) || any(isfinite, zupper)
    else
        z0 = [_inverse_transform(x0[i], lower[i], upper[i]) for i in active]
        zlower = fill(-Inf, length(active))
        zupper = fill(Inf, length(active))
        expand = z -> begin
            x = copy(x0)
            for (j, i) in enumerate(active)
                x[i] = _forward_transform(z[j], lower[i], upper[i])
            end
            x
        end
        has_bounds = false
    end

    return (;
        active,
        fixed = fixed_mask,
        lower,
        upper,
        z0,
        zlower,
        zupper,
        expand,
        has_bounds,
        bound_strategy,
    )
end

function _embed_active_matrix(C::Union{Nothing, AbstractMatrix}, active::AbstractVector{<:Integer}, n::Integer)
    C === nothing && return nothing
    length(active) == n && return Matrix{Float64}(C)
    out = zeros(Float64, n, n)
    for (ia, i) in enumerate(active), (ja, j) in enumerate(active)
        out[i, j] = C[ia, ja]
    end
    return out
end
