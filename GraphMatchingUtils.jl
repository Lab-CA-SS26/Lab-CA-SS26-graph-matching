module GraphMatchingUtils
    using LinearAlgebra
    export isPerm, sqd_frob, f0, ∇f0!, f1, ∇f1!, fλ, fλ_QAP, ∇fλ!, ∇fλ_QAP!, qapVal
    export FλForP, ∇FλForP!, FλForP_QAP, ∇FλForP_QAP!

   # permute matrix A by permutation matrix p
   # A is matrix
   # p is Vector where i1-->p[1], i2-->p[2], etc.
   function permute(A,p)
    A = A[p,:]
    A = A[:,p]
    return A
   end

   # returns true if P contains only zeros and ones and false if not
   function isPerm(P)
    return all(x -> x == 0.0 || x == 1.0, P)
   end

   # returns the squared frobenius norm of matrix A
   function sqd_frob(A)
    val = norm(A,2)
    return val^2
   end

   # returns the diagonal degree matrix of G
   # column by column is quicker to go through in julia
   function diagonal_degree(G)
    D = zeros(size(G))
    for j = 1:size(D,1)
        sum = 0.0
        for i = 1:size(D,1)
            sum += G[i, j]
        end
        D[j, j] = sum
    end
    return D
   end

   # returns the matrix Δ as stated in the paper
   function Δ(G,H)
    D_G = diagonal_degree(G)
    D_H = diagonal_degree(H)

    Δ_G_H = zeros(size(G))
    for i = 1:size(G,1)
        for j = 1:size(G,1)
            Δ_G_H[i, j] = D_H[j, j] - D_G[i, i]
        end
    end
    return Δ_G_H .^ 2
   end

   # returns the laplacian matrix of G
   function laplacian(G)
    return diagonal_degree(G) .- G
   end

   # function F0 as stated in the paper
   function f0(P,G,H)
    return sqd_frob(G*P .- P*H)
   end

   # gradient of F0 as stated in the paper
   # save solution value in variable "storage" for space economy
   function ∇f0!(storage, P, G, H)
       storage .= 2.0 .* ((G^2) * P .- 2.0 .* G * P * H .+ P * (H^2))
   end

   # function F1 as stated in the paper
   function f1(P, G, H)
    constantTerm = tr(GraphMatchingUtils.laplacian(G)^2)+tr(GraphMatchingUtils.laplacian(H)^2)
    return -tr(Δ(G,H)'*P) - 2.0 * (vec(P)' * vec(laplacian(G) * P * laplacian(H))) + constantTerm
   end

   # gradient of F1 as stated in the paper
   # save solution value in variable "storage" for space economy
   function ∇f1!(storage, P, G, H)
    storage .= -Δ(G,H)' .- 2.0 .* laplacian(G) * P * laplacian(H)
   end

   function fλ(P, λ, G, H)
    return (1-λ) * GraphMatchingUtils.f0(P, G, H)  +  λ * GraphMatchingUtils.f1(P, G, H)
   end

    struct FλForP
        λ::Float64
        G::Matrix{Float64}
        H::Matrix{Float64}
    end
    function(fλ_struct::FλForP)(P) 
        return fλ(P, fλ_struct.λ, fλ_struct.G, fλ_struct.H)
    end

    # function flipped for maximization and solving QAP
   function fλ_QAP(P, λ, G, H)
    return (1-λ) * (-GraphMatchingUtils.f1(P, G, H))  +  λ * (-GraphMatchingUtils.f0(P, G, H))
   end

    struct FλForP_QAP
        λ::Float64
        G::Matrix{Float64}
        H::Matrix{Float64}
    end
    function(fλ_struct::FλForP_QAP)(P) 
        return fλ_QAP(P, fλ_struct.λ, fλ_struct.G, fλ_struct.H)
    end

   # gradient of F1 as stated in the paper
   # save solution value in variable "storage" for space economy
   function ∇fλ!(storageλ, storage0, storage1, P, λ, G, H)
    GraphMatchingUtils.∇f0!(storage0, P, G, H)
    GraphMatchingUtils.∇f1!(storage1, P, G, H)
    storageλ .= (1.0-λ) .* storage0 .+ λ .* storage1
   end

   struct ∇FλForP!
        storage0::Matrix{Float64}
        storage1::Matrix{Float64}
        λ::Float64
        G::Matrix{Float64}
        H::Matrix{Float64}
    end
    function(∇fλ_struct::∇FλForP!)(storageλ, P)
        ∇fλ!(storageλ, ∇fλ_struct.storage0, ∇fλ_struct.storage1, P, ∇fλ_struct.λ, ∇fλ_struct.G, ∇fλ_struct.H)
    end

   # gradient flipped for maximization and solving QAP
   # save solution value in variable "storage" for space economy
   function ∇fλ_QAP!(storageλ, storage0, storage1, P, λ, G, H)
    GraphMatchingUtils.∇f0!(storage0, P, G, H)
    GraphMatchingUtils.∇f1!(storage1, P, G, H)
    storageλ .= (1.0-λ) .* (-storage1) .+ λ .* (-storage0)
   end

   struct ∇FλForP_QAP!
        storage0::Matrix{Float64}
        storage1::Matrix{Float64}
        λ::Float64
        G::Matrix{Float64}
        H::Matrix{Float64}
    end
    function(∇fλ_struct::∇FλForP_QAP!)(storageλ, P)
        ∇fλ_QAP!(storageλ, ∇fλ_struct.storage0, ∇fλ_struct.storage1, P, ∇fλ_struct.λ, ∇fλ_struct.G, ∇fλ_struct.H)
    end

   function qapVal(P,G,H)
    return tr(G*P*H'*P')
   end 
end