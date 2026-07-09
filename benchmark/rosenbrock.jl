include("common.jl")

rosenbrock(p) = (1 - p.x)^2 + 100 * (p.y - p.x^2)^2

function run_rosenbrock(; output_dir = DEFAULT_RESULTS_DIR)
    rows = NamedTuple[]
    initial = (x = -1.2, y = 1.0)
    truth = [1.0, 1.0]
    for spec in backend_specs()
        kwargs = spec.backend isa MinuitBackend ? (step_sizes = [0.1, 0.1],) : NamedTuple()
        result, time_ms = timed_fit(rosenbrock, initial, spec; kwargs...)
        push!(rows, fit_row("rosenbrock", spec, result, time_ms; truth))
    end
    path = write_csv(joinpath(output_dir, "rosenbrock_fits.csv"), rows)
    println("wrote ", path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_RESULTS_DIR
    run_rosenbrock(; output_dir)
end
