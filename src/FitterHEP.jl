module FitterHEP

using FiniteDiff
using LinearAlgebra
using Optim

export AbstractFitBackend,
    OptimBackend,
    FitResult,
    FitDiagnostics,
    fit,
    covariance,
    errors,
    correlation,
    hesse

include("core.jl")
include("covariance.jl")
include("backends/optim.jl")

end
