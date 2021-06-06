using LinearAlgebra

function HessenbergReduction( A::Matrix )
    X = copy(A)
    n = size(X, 1)
    H = A  # TODO Nie rozumiem jaki jest cel tej linii - można ją usunąć?
    if ( n > 2 )
        a1 = X[2:n, 1]
        e1 = zeros(n-1); e1[1] = 1
        sgn = sign(a1[1])
        v = (a1 + sgn*norm(a1)*e1); v = v./norm(v)
        Q1 = eye(n-1) - 2*(v*v')
        X[2:n,1] = Q1*X[2:n,1]
        X[1,2:n] = Q1*X[1,2:n]
        X[2:n,2:n] = Q1*X[2:n,2:n]*Q1' 
        H = HessenbergReduction( X[2:n,2:n] )
    else
        H = copy(X)
    end
    return X
end

function eye(n)
    return Matrix{Float64}(I,n,n)
end
