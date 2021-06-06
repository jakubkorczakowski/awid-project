using LinearAlgebra

include("hessenberg_reduction.jl")

function power_method(A::Union{Matrix, Symmetric}, l::Integer)
    # power method 
    n = size(A, 1)
    x = ones(n, 1)
    for i in 1:l
        x = A * x
        x = x / norm(x)
    end
    λ = x'*A*x
    return λ, x
end

function power_method_err(A::Union{Matrix, Symmetric}, l::Integer, expected_eigenvalue::Float64)
    # power method 
    n = size(A, 1)
    x = ones(n, 1)
    Δλ = []
    for i in 1:l
        x = A * x
        x = x / norm(x)
        push!(Δλ, abs(λ - expected_eigenvalue))
    end
    λ = x'*A*x
    return λ, x, Δλ
end


function power_method_hessen(A::Union{Matrix, Symmetric}, l::Integer) 
    # power method with transformation to Hessenberg form
    A = HessenbergReduction(A)
    n = size(A, 1)
    x = ones(n, 1)
    for i in 1:l
        x = A * x
        x = x / norm(x)
    end
    λ = x'*A*x
    return λ, x
end


function power_method_hessen_err(A::Union{Matrix, Symmetric}, l::Integer, expected_eigenvalue::Float64) 
    # power method with transformation to Hessenberg form
    A = HessenbergReduction(A)
    n = size(A, 1)
    x = ones(n, 1)
    Δλ = []
    for i in 1:l
        x = A * x
        x = x / norm(x)
        push!(Δλ, abs(λ - expected_eigenvalue))
    end
    λ = x'*A*x
    return λ, x, Δλ
end

function rayleigh_power_method(A::Union{Matrix, Symmetric}, l::Integer) 
    # rayleigh power method
    n = size(A, 1);
    x = ones(n, 1);
    mI = eye(n);
    mu = rand(1)[1];
        
    x = x / norm(x);
    y = (A - mu*mI) * x;
    
    for i in 1:l
        x = y / norm(y);
        y = (A - mu*mI) * x;
        λ = y' * x;
        mu = mu + 1 / λ[1];
    end
    λ = x'*A*x
    y = y / norm(y);
    return λ, y
end


function rayleigh_power_method_hessen(A::Union{Matrix, Symmetric}, l::Integer) 
    # rayleigh power method with transformation to Hessenberg form
    A = HessenbergReduction(A)
    n = size(A, 1);
    x = ones(n, 1);
    mI = eye(n);
    mu = rand(1)[1];
        
    x = x / norm(x);
    y = (A - mu*mI) * x;
    
    for i in 1:l
        x = y / norm(y);
        y = (A - mu*mI) * x;
        λ = y' * x;
        mu = mu + 1 / λ[1];
    end
    λ = x'*A*x
    y = y / norm(y);
    return λ, y
end
