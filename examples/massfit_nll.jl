using FitterHEP
using Printf
using Random

const XMIN = 5.0
const XMAX = 7.0
const WIDTH = XMAX - XMIN
const INV_SQRT2PI = inv(sqrt(2π))

const TRUE_MU = 5.28
const TRUE_SIGMA = 0.06
const TRUE_TAU = 1.0
const TRUE_SIGNAL_FRACTION = 0.30

function generate_mass_sample(n::Integer; seed::Integer = 12345)
    rng = MersenneTwister(seed)
    data = Vector{Float64}(undef, n)
    bg_norm = 1 - exp(-WIDTH / TRUE_TAU)
    for i in eachindex(data)
        if rand(rng) < TRUE_SIGNAL_FRACTION
            x = TRUE_MU + TRUE_SIGMA * randn(rng)
            while !(XMIN <= x <= XMAX)
                x = TRUE_MU + TRUE_SIGMA * randn(rng)
            end
            data[i] = x
        else
            data[i] = XMIN - TRUE_TAU * log1p(-rand(rng) * bg_norm)
        end
    end
    return data
end

function signal_pdf(x, mu, sigma)
    z = (x - mu) / sigma
    return INV_SQRT2PI * exp(-0.5 * z^2) / sigma
end

function background_pdf(x, tau)
    norm = tau * (1 - exp(-WIDTH / tau))
    return exp((XMIN - x) / tau) / norm
end

function unpack(raw)
    mu = raw[1]
    sigma = exp(raw[2])
    tau = exp(raw[3])
    signal_fraction = inv(1 + exp(-raw[4]))
    return (; mu, sigma, tau, signal_fraction)
end

function mass_nll(data, raw)
    p = unpack(raw)
    !(XMIN < p.mu < XMAX) && return Inf
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

function main()
    data = generate_mass_sample(10_000)
    initial = [5.25, log(0.08), log(0.8), 0.0]
    objective(raw) = mass_nll(data, raw)

    result = fit(
        objective,
        initial;
        backend = OptimBackend(:bfgs),
        autodiff = :forward,
        covariance = :finite_diff,
        errordef = 0.5,
        iterations = 5_000,
    )

    p = unpack(result.minimizer)
    println("converged: ", result.converged)
    @printf("minimum NLL: %.3f\n", result.minimum)
    @printf("mu              = %.5f\n", p.mu)
    @printf("sigma           = %.5f\n", p.sigma)
    @printf("tau             = %.5f\n", p.tau)
    @printf("signal fraction = %.5f\n", p.signal_fraction)
    println("raw-parameter errors: ", round.(errors(result); digits = 5))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
