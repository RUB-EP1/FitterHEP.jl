using FitterHEP
using Printf
using Random

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results")

function ensure_dir(path::AbstractString)
    mkpath(path)
    return path
end

function csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    ensure_dir(dirname(path))
    isempty(rows) && error("cannot write empty CSV: $path")
    columns = collect(keys(first(rows)))
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, col)) for col in columns), ","))
        end
    end
    return path
end

function flatten_values(x::AbstractVector)
    return Float64.(x)
end

function flatten_values(x::NamedTuple)
    values = Float64[]
    _flatten_namedtuple_values!(values, x)
    return values
end

function _flatten_namedtuple_values!(values::Vector{Float64}, nt::NamedTuple)
    for key in keys(nt)
        value = getproperty(nt, key)
        if value isa Real
            push!(values, Float64(value))
        elseif value isa NamedTuple
            _flatten_namedtuple_values!(values, value)
        else
            error("unsupported benchmark parameter value type: $(typeof(value))")
        end
    end
    return values
end

function parameter_columns(values; prefix = "p", max_parameters = 6)
    out = NamedTuple()
    n = min(length(values), max_parameters)
    for i in 1:n
        out = merge(out, NamedTuple{(Symbol(prefix, i),)}((values[i],)))
    end
    return out
end

function backend_specs(; bounded::Bool = false, transform_bounds::Bool = false)
    optim_extra = bounded ? (bound_strategy = transform_bounds ? :transform : :native,) : NamedTuple()
    return [
        (
            label = "optim_bfgs",
            backend = OptimBackend(:bfgs),
            kwargs = merge((iterations = 10_000,), optim_extra),
        ),
        (
            label = "optim_lbfgs",
            backend = OptimBackend(:lbfgs),
            kwargs = merge((iterations = 10_000,), optim_extra),
        ),
        (
            label = "optim_nelder_mead",
            backend = OptimBackend(:nelder_mead),
            kwargs = merge((iterations = 20_000,), optim_extra),
        ),
        (
            label = "minuit_strategy1",
            backend = MinuitBackend(strategy = 1, tolerance = 0.1),
            kwargs = NamedTuple(),
        ),
        (
            label = "minuit_strategy0",
            backend = MinuitBackend(strategy = 0, tolerance = 0.2),
            kwargs = NamedTuple(),
        ),
    ]
end

function run_fit(objective, initial, spec; kwargs...)
    return fit(
        objective,
        initial;
        backend = spec.backend,
        covariance = :none,
        spec.kwargs...,
        kwargs...,
    )
end

function timed_fit(objective, initial, spec; warmup::Bool = true, kwargs...)
    warmup && run_fit(objective, initial, spec; kwargs...)
    elapsed = @elapsed result = run_fit(objective, initial, spec; kwargs...)
    return result, 1_000 * elapsed
end

function fit_row(case_name::AbstractString, spec, result, time_ms::Real; truth = Float64[])
    values = flatten_values(result.minimizer)
    deltas = isempty(truth) ? fill(NaN, length(values)) : values .- truth
    base = (
        case = case_name,
        backend = spec.label,
        converged = result.converged,
        minimum = result.minimum,
        time_ms = Float64(time_ms),
        nfcn = result.diagnostics.nfcn === nothing ? missing : result.diagnostics.nfcn,
        iterations = result.diagnostics.iterations === nothing ? missing : result.diagnostics.iterations,
        status = result.diagnostics.status,
    )
    return merge(base, parameter_columns(values), parameter_columns(deltas; prefix = "dp"))
end

function summarize_rows(rows::Vector{<:NamedTuple}, group_key::Symbol)
    labels = unique(getproperty.(rows, group_key))
    summaries = NamedTuple[]
    for label in labels
        subset = filter(row -> getproperty(row, group_key) == label, rows)
        times = Float64[row.time_ms for row in subset]
        nfcns = skipmissing([row.nfcn for row in subset])
        converged = count(row -> row.converged, subset)
        push!(summaries, (
            backend = label,
            n = length(subset),
            converged = converged,
            success_rate = converged / length(subset),
            median_time_ms = median_no_stats(times),
            mean_time_ms = sum(times) / length(times),
            median_nfcn = isempty(collect(nfcns)) ? missing : median_no_stats(Float64.(collect(skipmissing([row.nfcn for row in subset])))),
        ))
    end
    return summaries
end

function median_no_stats(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    sorted = sort(Float64.(values))
    n = length(sorted)
    isodd(n) && return sorted[(n + 1) ÷ 2]
    return 0.5 * (sorted[n ÷ 2] + sorted[n ÷ 2 + 1])
end
