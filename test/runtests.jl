using FitterHEP
using LinearAlgebra
using Test

@testset "FitterHEP" begin
    @testset "OptimBackend constructors" begin
        @test OptimBackend() isa OptimBackend
        @test OptimBackend(:lbfgs) isa OptimBackend
        @test OptimBackend(:bfgs) isa OptimBackend
        @test OptimBackend(:nelder_mead) isa OptimBackend
        @test_throws ArgumentError OptimBackend(:unknown)
    end

    @testset "quadratic fit with covariance" begin
        μ = [1.5, -2.0]
        σ = [2.0, 3.0]
        objective(x) = sum(((x .- μ) ./ σ) .^ 2)

        result = fit(
            objective,
            [0.0, 0.0];
            backend = OptimBackend(:bfgs),
            covariance = :finite_diff,
            errordef = 1.0,
            iterations = 1_000,
        )

        @test result.converged
        @test result.minimizer ≈ μ atol = 1e-6
        @test result.minimum ≈ 0.0 atol = 1e-10
        @test covariance(result) ≈ Diagonal(σ .^ 2) atol = 1e-5
        @test errors(result) ≈ σ atol = 1e-5
        @test correlation(result) ≈ Matrix(I, 2, 2) atol = 1e-5
        @test result.diagnostics.status == :converged
        @test result.diagnostics.valid_covariance == true
    end

    @testset "Nelder Mead works without covariance" begin
        objective(x) = (x[1] - 3.0)^2 + (x[2] + 4.0)^2
        result = fit(objective, [10.0, 10.0]; backend = OptimBackend(:nelder_mead), iterations = 5_000)

        @test result.converged
        @test result.minimizer ≈ [3.0, -4.0] atol = 2e-4
        @test covariance(result) === nothing
        @test errors(result) === nothing
        @test correlation(result) === nothing
    end

    @testset "hesse helper" begin
        objective(x) = (x[1] - 1)^2 + 4 * (x[2] + 1)^2
        h = hesse(objective, [1.0, -1.0]; errordef = 1.0)

        @test h.hessian ≈ [2.0 0.0; 0.0 8.0] atol = 1e-5
        @test h.covariance ≈ [1.0 0.0; 0.0 0.25] atol = 1e-5
        @test h.errors ≈ [1.0, 0.5] atol = 1e-5
    end

    @testset "mass-fit example loads" begin
        include(joinpath(@__DIR__, "..", "examples", "massfit_nll.jl"))
        data = generate_mass_sample(100; seed = 7)
        @test length(data) == 100
        @test all(x -> 5.0 <= x <= 7.0, data)
        @test isfinite(mass_nll(data, [5.28, log(0.06), log(1.0), 0.0]))
    end
end
