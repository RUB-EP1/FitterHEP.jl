include("common.jl")

function read_simple_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return Symbol[], NamedTuple[]
    header = Symbol.(split(lines[1], ','))
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ',')
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return header, rows
end

function maybe_number(x)
    s = string(x)
    s == "missing" && return s
    v = tryparse(Float64, s)
    return v === nothing ? s : v
end

function fmt(x; digits = 4)
    x isa Missing && return ""
    if x isa AbstractString
        v = maybe_number(x)
        v isa Number || return x
        return fmt(v; digits)
    elseif x isa Bool
        return string(x)
    elseif x isa Integer
        return string(x)
    elseif x isa Number
        !isfinite(x) && return string(x)
        if abs(x) != 0 && (abs(x) < 1e-3 || abs(x) >= 1e4)
            return Printf.format(Printf.Format("%." * string(digits) * "e"), x)
        end
        return string(round(x; digits = digits))
    end
    return string(x)
end

function markdown_table(io, headers, rows)
    println(io, "| ", join(headers, " | "), " |")
    println(io, "| ", join(fill("---", length(headers)), " | "), " |")
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
end

function deterministic_table(path)
    _, rows = read_simple_csv(path)
    return [
        [
            row.backend,
            row.converged,
            fmt(row.minimum; digits = 3),
            fmt(row.time_ms; digits = 3),
            row.nfcn,
            row.iterations,
            fmt(row.p1; digits = 6),
            fmt(row.p2; digits = 6),
            fmt(row.dp1; digits = 3),
            fmt(row.dp2; digits = 3),
        ] for row in rows
    ]
end

function summary_table(path)
    _, rows = read_simple_csv(path)
    return [
        [
            row.backend,
            row.n,
            row.converged,
            fmt(row.success_rate; digits = 3),
            fmt(row.median_time_ms; digits = 3),
            fmt(row.mean_time_ms; digits = 3),
            row.median_nfcn,
        ] for row in rows
    ]
end

function write_comparison_tables(; results_dir = joinpath(@__DIR__, "results"))
    out = joinpath(results_dir, "comparison_tables.md")
    ensure_dir(results_dir)
    open(out, "w") do io
        println(io, "# Fit-Only Comparison Tables")
        println(io)
        println(io, "Generated from CSV files in `benchmark/results/`.")
        println(io, "Covariance calculations are not included in these timings.")
        println(io)

        println(io, "## Quadratic")
        markdown_table(
            io,
            ["backend", "converged", "minimum", "time_ms", "nfcn", "iterations", "p1", "p2", "dp1", "dp2"],
            deterministic_table(joinpath(results_dir, "quadratic_fits.csv")),
        )
        println(io)

        println(io, "## Rosenbrock")
        markdown_table(
            io,
            ["backend", "converged", "minimum", "time_ms", "nfcn", "iterations", "p1", "p2", "dp1", "dp2"],
            deterministic_table(joinpath(results_dir, "rosenbrock_fits.csv")),
        )
        println(io)

        println(io, "## Bounded Fraction Toys")
        markdown_table(
            io,
            ["backend", "n", "converged", "success_rate", "median_time_ms", "mean_time_ms", "median_nfcn"],
            summary_table(joinpath(results_dir, "bounded_fraction_summary.csv")),
        )
        println(io)

        println(io, "## Mass-Fit Toys")
        markdown_table(
            io,
            ["backend", "n", "converged", "success_rate", "median_time_ms", "mean_time_ms", "median_nfcn"],
            summary_table(joinpath(results_dir, "massfit_summary.csv")),
        )
    end
    println("wrote ", out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results")
    write_comparison_tables(; results_dir)
end
