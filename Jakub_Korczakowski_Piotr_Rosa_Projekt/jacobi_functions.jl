using LinearAlgebra: Matrix
using LinearAlgebra

include("hessenberg_reduction.jl")

function maxst(A::Array)  
    # function used for locating maximum value of input matrix
    s = 1;
    t = 2;
    n = size(A,1);

    for c = 2:n
        for r = 1:c-1
            if abs(A[r,c]) > abs(A[s,t])
            s = r;
            t = c;
            end
        end
    end
    return s,t
end


function jacobi_A(A::Array, l::Integer)  
    # Jacobi algorithm #1 for computing eigenvalues 
    n = size(A,1);
    for i = 1:l
        s,t = maxst(A);
        d = sqrt((A[s,s] - A[t,t])^2 + 4*A[s,t]^2);
        sin2t = 2*A[s,t]/d;
        cos2t = (A[s,s] - A[t,t]) / d;
        dt = sqrt(2*(1+cos2t));
        sint = abs(sin2t) / dt;
        cost = abs((1+ cos2t) / dt);
        cost = sign(A[s,t]) * cost;

        R = eye(n);
        R[s,s] = cost;
        R[t,t] = cost;
        R[s,t] = -sint;
        R[t,s] = sint;
        A = R'*A*R;
    end
    sort(diag(A))
end

function jacobi_A_vect(A::Array, l::Integer)  
    # Jacobi algorithm #1 for computing eigenvalues 
    n = size(A,1);
    E = eye(n);
    for i = 1:l
        s,t = maxst(A);
        d = sqrt((A[s,s] - A[t,t])^2 + 4*A[s,t]^2);
        sin2t = 2*A[s,t]/d;
        cos2t = (A[s,s] - A[t,t]) / d;
        dt = sqrt(2*(1+cos2t));
        sint = abs(sin2t) / dt;
        cost = abs((1+ cos2t) / dt);
        cost = sign(A[s,t]) * cost;

        R = eye(n);
        R[s,s] = cost;
        R[t,t] = cost;
        R[s,t] = -sint;
        R[t,s] = sint;
        A = R'*A*R;
        E = E*R; 
    end
    sorteigen(diag(A), E)
end



function jacobi_A_err(A::Array, l::Integer,  expected_eigenvalues::Vector)  
    # Jacobi algorithm #1 for computing eigenvalues 
    n = size(A,1);
    ΔΛ::Array{Array} = [];
    for i = 1:l
        s,t = maxst(A);
        d = sqrt((A[s,s] - A[t,t])^2 + 4*A[s,t]^2);
        sin2t = 2*A[s,t]/d;
        cos2t = (A[s,s] - A[t,t]) / d;
        dt = sqrt(2*(1+cos2t));
        sint = abs(sin2t) / dt;
        cost = abs((1+ cos2t) / dt);
        cost = sign(A[s,t]) * cost;

        R = eye(n);
        R[s,s] = cost;
        R[t,t] = cost;
        R[s,t] = -sint;
        R[t,s] = sint;
        A = R'*A*R;
        Λ = sort(diag(A));
        push!(ΔΛ, abs.(Λ - expected_eigenvalues))
    end
    sort(diag(A)), ΔΛ
end


function jacobi_B_vect(A::Array, l::Integer)  
    # Jacobi algorithm #2 for computing eigenvalues
    n = size(A,1);
    X = copy(A);
    H = eye(n);  # H is a eigenvectors matrix
    for iter = 1:l
        i,j = maxst(X);
        Si = X[:, i]
        Sj = X[:, j]
        
        θ = 0.5*atan(2*Si[j], Sj[j]-Si[i])
        c = cos(θ)
        s = sin(θ)
        
        X[i, :] = X[:, i] = c*Si - s*Sj
        X[j, :] = X[:, j] = s*Si + c*Sj
        X[i,j] = 0
        X[j,i] = 0
        X[i,i] = c^2*Si[i] - 2*s*c*Si[j] + s^2*Sj[j]
        X[j,j] = s^2*Si[i] + 2*s*c*Si[j] + c^2*Sj[j]
        
        Hi = H[:, i]
        H[:, i] = c*Hi - s*H[:, j]
        H[:, j] = s*Hi + c*H[:, j]
    end
    sorteigen(diag(X), H)
end


function jacobi_B(A::Array, l::Integer)  
    # Jacobi algorithm #2 for computing only eigenvalues
    n = size(A,1);
    X = copy(A);
    for iter = 1:l
        i,j = maxst(X);
        Si = X[:, i]
        Sj = X[:, j]
        
        θ = 0.5*atan(2*Si[j], Sj[j]-Si[i])
        c = cos(θ)
        s = sin(θ)
        
        X[i, :] = X[:, i] = c*Si - s*Sj
        X[j, :] = X[:, j] = s*Si + c*Sj
        X[i,j] = 0
        X[j,i] = 0
        X[i,i] = c^2*Si[i] - 2*s*c*Si[j] + s^2*Sj[j]
        X[j,j] = s^2*Si[i] + 2*s*c*Si[j] + c^2*Sj[j]
    end
    sort(diag(X))
end


function jacobi_B_err(A::Array, l::Integer, expected_eigenvalues::Vector)  
    # Jacobi algorithm #2 for computing only eigenvalues with error comp
    n = size(A,1);
    X = copy(A);
    ΔΛ::Array{Array} = [];
    for iter = 1:l
        i,j = maxst(X);
        Si = X[:, i]
        Sj = X[:, j]
        
        θ = 0.5*atan(2*Si[j], Sj[j]-Si[i])
        c = cos(θ)
        s = sin(θ)
        
        X[i, :] = X[:, i] = c*Si - s*Sj
        X[j, :] = X[:, j] = s*Si + c*Sj
        X[i,j] = 0
        X[j,i] = 0
        X[i,i] = c^2*Si[i] - 2*s*c*Si[j] + s^2*Sj[j]
        X[j,j] = s^2*Si[i] + 2*s*c*Si[j] + c^2*Sj[j]

        Λ = sort(diag(X));
        push!(ΔΛ, abs.(Λ - expected_eigenvalues))
    end
    sort(diag(X)), ΔΛ
end
