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
