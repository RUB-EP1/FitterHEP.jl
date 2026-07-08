module FitterHEP

using FiniteDiff
using LinearAlgebra
using Minuit2
using Optim

export AbstractFitBackend,
    OptimBackend,
    MinuitBackend,
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
include("backends/minuit.jl")

end
