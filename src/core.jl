# Computes the SVD of a 2-by-2 matrix M, returning U, S, and V such that A == U*S*V'.
# Reference:
# Blinn, J. (1996). Consider the lowly 2 x 2 matrix. IEEE Computer Graphics and Applications, 16(2), 82-88.
# https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
function svd2x2(M::AbstractMatrix{T}) where T <: Number
    E = (M[1,1] + M[2,2]) / 2
    F = (M[1,1] - M[2,2]) / 2
    G = (M[2,1] + M[1,2]) / 2
    H = (M[2,1] - M[1,2]) / 2
    Q = sqrt(E^2 +H^2)
    R = sqrt(F^2 + G^2)
    sx = Q + R
    sy = Q - R
    a₁ = atan(G,F)
    a₂ = atan(H,E)
    θ = (a₂ - a₁) / 2
    ϕ = (a₂ + a₁) / 2
    s = sign(sy)

    sinϕ, cosϕ = sin(ϕ), cos(ϕ)
    sinθ, cosθ = sin(θ), cos(θ)

    U = SMatrix{2, 2, T}(cosϕ, sinϕ, -s * sinϕ, s * cosϕ)
    S = SMatrix{2, 2, T}(sx, 0.0, 0.0, abs(sy))
    V = SMatrix{2, 2, T}(cosθ, -sinθ, sinθ, cosθ)
    U, S, V
end

# Computes the Moore-Penrose pseudoinverse for a 2-by-2 matrix M by only inverting
# singular values above the threshold `tol = sqrt(eps(real(float(one(eltype(M))))))`
function pinv2x2(M)
    U, S, V = svd2x2(M)
    pinv2x2(U, S, V)
end

function pinv2x2(
    U::AbstractMatrix{T}, S::AbstractMatrix{T}, V::AbstractMatrix{T},
) where T <: Number
    tol = √eps(real(float(one(T))))
    D = SMatrix{2, 2, T}(
        S[1, 1] > tol ? (1.0 / S[1, 1]) : 0.0, 0.0, 0.0,
        S[2, 2] > tol ? (1.0 / S[2, 2]) : 0.0,
    )
    U * D * V'
end
