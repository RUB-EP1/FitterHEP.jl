include("common.jl")

const MASS_MIN = 5.0
const MASS_MAX = 7.0
const MASS_WIDTH = MASS_MAX - MASS_MIN
const INV_SQRT2PI_BENCH = inv(sqrt(2π))
const TRUE_MU = 5.28
const TRUE_SIGMA = 0.06
const TRUE_TAU = 1.0
const TRUE_SIGNAL_FRACTION = 0.30

function generate_mass_sample(rng::AbstractRNG, n::Integer)
    data = Vector{Float64}(undef, n)
    bg_norm = 1 - exp(-MASS_WIDTH / TRUE_TAU)
    for i in eachindex(data)
        if rand(rng) < TRUE_SIGNAL_FRACTION
            x = TRUE_MU + TRUE_SIGMA * randn(rng)
            while !(MASS_MIN <= x <= MASS_MAX)
                x = TRUE_MU + TRUE_SIGMA * randn(rng)
            end
            data[i] = x
        else
            data[i] = MASS_MIN - TRUE_TAU * log1p(-rand(rng) * bg_norm)
        end
    end
    return data
end

function unpack_mass(raw)
    return (
        mu = raw[1],
        sigma = exp(raw[2]),
        tau = exp(raw[3]),
        signal_fraction = inv(1 + exp(-raw[4])),
    )
end

function signal_pdf(x, mu, sigma)
    z = (x - mu) / sigma
    return INV_SQRT2PI_BENCH * exp(-0.5 * z^2) / sigma
end

function background_pdf(x, tau)
    norm = tau * (1 - exp(-MASS_WIDTH / tau))
    return exp((MASS_MIN - x) / tau) / norm
end

function mass_nll(data, raw)
    p = unpack_mass(raw)
    !(MASS_MIN < p.mu < MASS_MAX) && return Inf
    !(1e-3 < p.sigma < 1.0) && return Inf
    !(1e-3 < p.tau < 20.0) && return Inf
    nll = 0.0
    for x in data
        sig = signal_pdf(x, p.mu, p.sigma)
        bg = background_pdf(x, p.tau)
        pdf = p.signal_fraction * sig + (1 - p.signal_fraction) * bg
        (!isfinite(pdf) || pdf <= 0) && return Inf
        nll -= log(pdf)
    end
    return nll
end

function run_massfit_toys(;
    output_dir = DEFAULT_RESULTS_DIR,
    n_toys::Integer = 20,
    n_events::Integer = 2_000,
    seed::Integer = 271828,
)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    initial = [5.25, log(0.08), log(0.8), 0.0]
    truth = [TRUE_MU, log(TRUE_SIGMA), log(TRUE_TAU), log(TRUE_SIGNAL_FRACTION / (1 - TRUE_SIGNAL_FRACTION))]
    for toy in 1:n_toys
        data = generate_mass_sample(rng, n_events)
        objective(raw) = mass_nll(data, raw)
        for spec in backend_specs()
            kwargs = spec.backend isa MinuitBackend ? (step_sizes = [0.01, 0.05, 0.05, 0.1],) : NamedTuple()
            result, time_ms = timed_fit(objective, initial, spec; kwargs...)
            row = fit_row("massfit", spec, result, time_ms; truth)
            push!(rows, merge((toy = toy, n_events = n_events), row))
        end
    end
    path = write_csv(joinpath(output_dir, "massfit_toys.csv"), rows)
    summary_path = write_csv(joinpath(output_dir, "massfit_summary.csv"), summarize_rows(rows, :backend))
    println("wrote ", path)
    println("wrote ", summary_path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_RESULTS_DIR
    n_toys = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    run_massfit_toys(; output_dir, n_toys)
end
