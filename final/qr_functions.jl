using LinearAlgebra

include("hessenberg_reduction.jl")

function mgs(A::Array)
    n = size(A,1);
    R = zeros(n, n);
    Q = zeros(n, n);
    for j = 1:n
        v = A[:,j];
        for i = 1:j-1
            R[i,j] = Q[:,i]'*v;
            v = v - R[i,j]*Q[:,i];
        end
        R[j,j] = norm(v)
        Q[:,j] = v / R[j,j]
    end
    return Q,R
end

function mgramschmidt(A::Array) 
    # gram-schmidt for nxm matrix, used in SVD script
    n, m = size(A)
    Q = zeros(n, m)
    R = zeros(m, m)
    for j = 1:m
        v = A[:,j]
        for k = 1:j-1
            R[k,j] = dot(Q[:,k]',v)
            v -= R[k,j]*Q[:,k]
        end
        R[j,j] = norm(v)
        Q[:,j] = v/R[j,j]
    end
    return Q, R
end


function QR_eigen(A::Array, l::Integer)  
    # QR function without calculated eigenvectors
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
    end
    sort(diag(A))
end


function QR_eigen_err(A::Array, l::Integer, expected_eigenvalues::Vector)  
    # QR function without calculated eigenvectors
    ΔΛ::Array{Array} = [];
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        Λ = sort!(diag(A));
        push!(ΔΛ, abs.(Λ - expected_eigenvalues))
    end
    sort(diag(A)), ΔΛ
end


function QR_eigen_hessen(A::Array, l::Integer)  
    # QR function without calculated eigenvectors with transformation to Hessenberg form
    A = HessenbergReduction(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        A = HessenbergReduction(A);
    end
    sort(diag(A))
end


function QR_eigen_hessen_err(A::Array, l::Integer, expected_eigenvalues::Vector)  
    # QR function without calculated eigenvectors with transformation to Hessenberg form
    ΔΛ::Array{Array} = [];
    A = HessenbergReduction(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        A = HessenbergReduction(A);
        Λ = sort!(diag(A));
        push!(ΔΛ, abs.(Λ - expected_eigenvalues));
    end
    sort(diag(A)), ΔΛ
end


function QR_eigen_vect(A::Array, l::Integer)  
    # QR function calculating eigenvectors
    E = I(size(A,1))
    Q,R = mgs(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        E = E * Q
    end
    sorteigen(diag(A), E)
end


function QR_eigen_vect_hessen(A::Array, l::Integer)  
    # QR function calculating eigenvectors with transformation to Hessenberg form
    A = HessenbergReduction(A)
    E = I(size(A,1))
    Q,R = mgs(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        A = HessenbergReduction(A);
        E = E * Q;
    end
    sorteigen(diag(A), E)
end


function shiftQRc(A::Array, maxiter=500)  
    #  QR function that is able to calculate complex eigenvalues
    tol=1e-14
    m=size(A,1)
    lam=zeros(m,1)
    n=m
    q,r=mgs(A)
    while n>1
        iter=0
        while sort(abs.(A[n,1:n-1]))[end]>tol && iter<maxiter
            iter=iter+1
            mu=A[n,n]
            q,r=mgs(A-mu*eye(n))
            A=r*q+mu*eye(n)
        end
        if iter<maxiter #block with 1x1
            lam[n]=A[n,n]
            n=n-1
            A=A[1:n,1:n]
        else
            lam=complex(lam)
            disc=(A[n-1,n-1]-A[n,n])^2+4*A[n,n-1]*A[n-1,n]
            temp=sqrt.(disc+0*im)
            lam[n]=(A[n-1,n-1]+A[n,n] +temp)/2
            lam[n-1]=(A[n-1,n-1]+A[n,n]-temp)/2
            n=n-2
            A=A[1:n,1:n]
        end
    end
    if n>0
        lam[1]=A[1,1]
    end
    return lam
end        


function shiftQRc_hessen(A::Array, maxiter=500)  #  QR function that is able to calculate complex eigenvalues with transformation to Hessenberg form
    A = HessenbergReduction(A);
    tol=1e-14
    m=size(A,1)
    lam=zeros(m,1)
    n=m
    q,r=mgs(A)
    while n>1
        iter=0
        while sort(abs.(A[n,1:n-1]))[end]>tol && iter<maxiter
                         iter=iter+1
                         mu=A[n,n]
            q,r=mgs(A-mu*eye(n))
            A=r*q+mu*eye(n)
            # A = HessenbergReduction(A)
        end
        if iter<maxiter #block with 1x1
            lam[n]=A[n,n]
            n=n-1
            A=A[1:n,1:n]
        else
            lam=complex(lam)
                         disc=(A[n-1,n-1]-A[n,n])^2+4*A[n,n-1]*A[n-1,n]
                         temp=sqrt.(disc+0*im)
            lam[n]=(A[n-1,n-1]+A[n,n] +temp)/2
            lam[n-1]=(A[n-1,n-1]+A[n,n]-temp)/2
            n=n-2
            A=A[1:n,1:n]
        end
    end
    if n>0
        lam[1]=A[1,1]
    end
    return lam
end    


function WilkinsonShift( a::Number, b::Number, c::Number )
    # Calculate Wilkinson's shift for symmetric matrices: 
    δ = (a-c)/2
    return c - sign(δ)*b^2/(abs(δ) + sqrt(δ^2+b^2))
end


function QRwithShifts( A::Matrix, iter_number::Int )
    n = size(A,1)
    myeigs = zeros(n)
    if ( n == 1 )
        myeigs[1] = A[1,1]
    else
        I = eye( n )
        A = HessenbergReduction( A )
        for i = 1:iter_number
            mu = WilkinsonShift( A[n-1,n-1], A[n,n], A[n-1,n] )
            (Q,R) = mgs(A - mu*I)
            A = R*Q + mu*I
        end
        myeigs = [A[n,n] ; QRwithShifts( A[1:n-1, 1:n-1], iter_number )]
    end
    return myeigs
end
