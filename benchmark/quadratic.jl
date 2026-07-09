include("common.jl")

function quadratic_objective(x)
    μ = [1.5, -2.0]
    σ = [2.0, 3.0]
    return sum(((x .- μ) ./ σ) .^ 2)
end

function run_quadratic(; output_dir = DEFAULT_RESULTS_DIR)
    rows = NamedTuple[]
    initial = [0.0, 0.0]
    truth = [1.5, -2.0]
    for spec in backend_specs()
        result, time_ms = timed_fit(quadratic_objective, initial, spec)
        push!(rows, fit_row("quadratic", spec, result, time_ms; truth))
    end
    path = write_csv(joinpath(output_dir, "quadratic_fits.csv"), rows)
    println("wrote ", path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_RESULTS_DIR
    run_quadratic(; output_dir)
end
