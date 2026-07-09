include("common.jl")

function generate_counting_sample(rng::AbstractRNG, n::Integer, fraction::Real)
    return [rand(rng) < fraction for _ in 1:n]
end

function binomial_nll(p, data)
    f = p.fraction
    !(0 < f < 1) && return Inf
    n_success = count(data)
    n_total = length(data)
    return -(n_success * log(f) + (n_total - n_success) * log1p(-f))
end

function run_bounded_fraction_toys(;
    output_dir = DEFAULT_RESULTS_DIR,
    n_toys::Integer = 50,
    n_events::Integer = 1_000,
    true_fraction::Real = 0.35,
    seed::Integer = 314159,
)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    specs = backend_specs(; bounded = true, transform_bounds = true)
    for toy in 1:n_toys
        data = generate_counting_sample(rng, n_events, true_fraction)
        objective(p) = binomial_nll(p, data)
        for spec in specs
            kwargs = spec.backend isa MinuitBackend ? (bounds = ([1e-6], [1 - 1e-6]), step_sizes = [0.05]) :
                (bounds = ([1e-6], [1 - 1e-6]),)
            result, time_ms = timed_fit(objective, (fraction = 0.5,), spec; kwargs...)
            row = fit_row("bounded_fraction", spec, result, time_ms; truth = [true_fraction])
            push!(rows, merge((toy = toy, n_events = n_events, true_fraction = true_fraction), row))
        end
    end
    path = write_csv(joinpath(output_dir, "bounded_fraction_toys.csv"), rows)
    summary_path = write_csv(joinpath(output_dir, "bounded_fraction_summary.csv"), summarize_rows(rows, :backend))
    println("wrote ", path)
    println("wrote ", summary_path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_RESULTS_DIR
    n_toys = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
    run_bounded_fraction_toys(; output_dir, n_toys)
end
