using LinearAlgebra

include("hessenberg_reduction.jl")

function potegowa(A, l::Integer)  # power method 
    n = size(A, 1)
    x = ones(n, 1)
    for i in 1:l
        x = A * x
        x = x / norm(x)
    end
    λ = x'*A*x
    return λ, x
end


function potegowa_hessen(A, l::Integer)  # power method with transformation to Hessenberg form
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


function rayleigh_power_method(A::Symmetric{Float64}, l::Integer)  # rayleigh power method
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


function rayleigh_power_method_hessen(A::Symmetric{Float64}, l::Integer)  # rayleigh power method with transformation to Hessenberg form
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


function typed_power_method(A::Symmetric{Float64}, l::Integer)  # power method with types
    n::Int64 = size(A, 1)
    x::Array{Float64} = ones(n, 1)
    for i::Int64 in 1:l
        x = A * x
        x = x / norm(x)
    end
    λ::Float64 = (x'*A*x)[1]
    return λ, x
end


function typed_power_method_hessen(A::Symmetric{Float64}, l::Integer)  # power method with types and transformation to Hessenberg form
    n::Int64 = size(A, 1)
    x::Array{Float64} = ones(n, 1)
    for i::Int64 in 1:l
        x = A * x
        x = x / norm(x)
    end
    λ::Float64 = (x'*A*x)[1]
    return λ, x
end