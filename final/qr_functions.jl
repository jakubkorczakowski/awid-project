include("hessenberg_reduction.jl")

function mgs(A)
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


function QR_eigen(A, l::Integer)  # QR function without calculated eigenvectors
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
    end
    A
end


function QR_eigen_hessen(A, l::Integer)  # QR function without calculated eigenvectors with transformation to Hessenberg form
    A = HessenbergReduction(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
    end
    A
end


function QR_eigen_vect(A, l::Integer)  # QR function calculating eigenvectors
    E = I(size(A,1))
    Q,R = mgs(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        E = E * Q
    end
    display(E)  # to jest macierz wektorów własnych
    A
end


function QR_eigen_vect_hessen(A, l::Integer)  # QR function calculating eigenvectors with transformation to Hessenberg form
    A = HessenbergReduction(A)
    E = I(size(A,1))
    Q,R = mgs(A)
    for k = 1:l
        Q,R = mgs(A);
        A = R*Q;
        E = E * Q
    end
    display(E)  # that is a matrix of eigenvectors
    A
end


function shiftQRc(A::Array, maxiter=500)  #  QR function that is able to calculate complex eigenvalues
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
    A = HessenbergReduction(A)
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


function WilkinsonShift( a::Number, b::Number, c::Number )
    # Calculate Wilkinson's shift for symmetric matrices: 
    δ = (a-c)/2
    return c - sign(δ)*b^2/(abs(δ) + sqrt(δ^2+b^2))
end


function QRwithShifts( A::Matrix, iter_number::Int )
   # The QR algorithm for symmetric A with Rayleigh shifts and Hessenberg reduction. Please use eigvals() in 
   # Julia for serious applications.
    n = size(A,1)
    myeigs = zeros(n)
    if ( n == 1 )
        myeigs[1] = A[1,1]
    else
        I = eye( n )
        # Reduction to Hessenberg form:
        A = HessenbergReduction( A )
        # Let's start the shifted QR algorithm with 
#         while( norm(A[n,n-1]) > 1e-10 )
        for i = 1:iter_number
            mu = WilkinsonShift( A[n-1,n-1], A[n,n], A[n-1,n] )
            # This line should use faster Hessenberg reduction:
            (Q,R) = mgs(A - mu*I)
            # This line needs speeding up, currently O(n^3) operations!: 
            A = R*Q + mu*I
        end
        # Deflation and recurse:
        myeigs = [A[n,n] ; QRwithShifts( A[1:n-1, 1:n-1], iter_number )]
    end
    return myeigs
end


function QRwithoutShifts( A::Matrix, iter_number::Int )
   # The QR algorithm for symmetric A with Rayleigh shifts and Hessenberg reduction. Please use eigvals() in 
   # Julia for serious applications.
    n = size(A,1)
    myeigs = zeros(n)
    if ( n == 1 )
        myeigs[1] = A[1,1]
    else
        I = eye( n )
        # Reduction to Hessenberg form:
        A = HessenbergReduction( A )
        # Let's start the shifted QR algorithm with 
#         while( norm(A[n,n-1]) > 1e-10 )
        for i = 1:iter_number
            (Q,R) = mgs(A)
            # This line needs speeding up, currently O(n^3) operations!: 
            A = R*Q
        end
        # Deflation and recurse:
        myeigs = [A[n,n] ; QRwithoutShifts( A[1:n-1, 1:n-1], iter_number)]
    end
    return myeigs
end
