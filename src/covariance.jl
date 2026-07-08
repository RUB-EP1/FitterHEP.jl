function _covariance_from_hessian(H::AbstractMatrix; errordef::Real = 0.5, inversion::Symbol = :auto)
    scale = 2 * Float64(errordef)
    C = if inversion === :inv
        inv(H)
    elseif inversion === :pinv
        pinv(H)
    elseif inversion === :auto
        try
            isposdef(Symmetric(H)) ? inv(H) : pinv(H)
        catch
            pinv(H)
        end
    else
        throw(ArgumentError("unsupported covariance inversion policy: $inversion"))
    end
    return scale .* Matrix(C)
end

function _correlation_from_covariance(C::AbstractMatrix)
    n = size(C, 1)
    R = Matrix{Float64}(I, n, n)
    for i in 1:n, j in 1:n
        denom = C[i, i] * C[j, j]
        R[i, j] = denom > 0 ? C[i, j] / sqrt(denom) : NaN
    end
    return R
end

function _errors_from_covariance(C::AbstractMatrix)
    return [C[i, i] >= 0 ? sqrt(C[i, i]) : NaN for i in axes(C, 1)]
end

"""
    hesse(objective, point; method=:finite_diff, errordef=0.5, inversion=:auto)

Compute a Hessian-derived covariance summary at `point`.

For standalone objective functions, `method=:finite_diff` is supported.
"""
function hesse(
    objective,
    point;
    method::Symbol = :finite_diff,
    errordef::Real = 0.5,
    inversion::Symbol = :auto,
)
    method === :finite_diff || throw(ArgumentError("Phase 1 supports only method=:finite_diff"))
    x = collect(Float64, point)
    H = FiniteDiff.finite_difference_hessian(objective, x)
    C = _covariance_from_hessian(H; errordef, inversion)
    return (hessian = H, covariance = C, errors = _errors_from_covariance(C), correlation = _correlation_from_covariance(C))
end

function _covariance_summary(C::AbstractMatrix)
    cov = Matrix{Float64}(C)
    return (covariance = cov, errors = _errors_from_covariance(cov), correlation = _correlation_from_covariance(cov))
end

function _maybe_covariance(objective, point; covariance_method, errordef, inversion)
    covariance_method in (:none, nothing, false) && return (nothing, nothing, nothing, nothing, false)
    method = covariance_method === :auto ? :finite_diff : covariance_method
    try
        h = hesse(objective, point; method, errordef, inversion)
        return (h.hessian, h.covariance, h.errors, h.correlation, true)
    catch
        return (nothing, nothing, nothing, nothing, false)
    end
end
