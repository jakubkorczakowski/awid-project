using LinearAlgebra

function HessenbergReduction( A::Union{Matrix, Symmetric} )
    n = size(A, 1)
    H = A
    if ( n > 2 )
        a1 = A[2:n, 1]
        e1 = zeros(n-1); e1[1] = 1
        sgn = sign(a1[1])
        v = (a1 + sgn*norm(a1)*e1); v = v./norm(v)
        Q1 = eye(n-1) - 2*(v*v')
        A[2:n,1] = Q1*A[2:n,1]
        A[1,2:n] = Q1*A[1,2:n]
        A[2:n,2:n] = Q1*A[2:n,2:n]*Q1' 
        H = HessenbergReduction( A[2:n,2:n] )
    else
        H = copy(A)
    end
    return A
end

function eye(n)
    return Matrix{Float64}(I,n,n)
end

function sorteigen(evals::Vector{T}, evecs::Matrix{T}) where {T}
    p = sortperm(evals)
    evals[p], evecs[:, p]
end