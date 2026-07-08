abstract type AbstractFitBackend end

"""
    FitDiagnostics

Backend-independent optimizer status information. Fields that are not meaningful
for a backend are set to `nothing`.
"""
Base.@kwdef struct FitDiagnostics
    status::Symbol = :unknown
    message::String = ""
    iterations::Union{Nothing, Int} = nothing
    nfcn::Union{Nothing, Int} = nothing
    edm::Union{Nothing, Float64} = nothing
    reached_call_limit::Union{Nothing, Bool} = nothing
    valid_covariance::Union{Nothing, Bool} = nothing
    backend_status::Any = nothing
end

"""
    FitResult

Common fit result returned by `fit`.
"""
struct FitResult{P,R,B}
    backend::B
    minimizer::P
    minimum::Float64
    converged::Bool
    covariance::Union{Nothing, Matrix{Float64}}
    errors::Union{Nothing, Vector{Float64}}
    correlation::Union{Nothing, Matrix{Float64}}
    raw_result::R
    diagnostics::FitDiagnostics
end

"""
    fit(objective, initial; backend=OptimBackend(:lbfgs), kwargs...)

Minimize `objective` starting from `initial` with the selected backend.

Vector-like numeric parameters are supported through `OptimBackend` and
`MinuitBackend`.
"""
function fit(objective, initial; backend::AbstractFitBackend = OptimBackend(:lbfgs), kwargs...)
    return _fit(backend, objective, initial; kwargs...)
end

function _fit end

covariance(result::FitResult) = result.covariance
errors(result::FitResult) = result.errors
correlation(result::FitResult) = result.correlation
