using LinearAlgebra
using Plots
using Random

function load_mat_hw1(n,p)  #From the given file "load_mat_hw1.jl"
    Random.seed!(0);
    p1 = Integer(floor(3*p/4));
    p2 = p-p1;
    Q = randn(n,p1);
    Z1 = Q*Diagonal(randn(1,p1).*exp.(-abs.(1:p1)))*randn(p1,p1);
    I = sortperm(vec(sum(abs.(Z1), dims=1)));
    Z1 = Z1[:,I];
    Q = randn(n,p2);
    Z2 = Q*Diagonal(randn(1,p2).*exp.(-abs.(1:p2)))*randn(p2,p2);
    Z = hcat(Z1,Z2);
    return Z;
end

function QR_std(V)
    A = V
    p = minimum(size(A))
    Q = zeros(size(A)[1],1)
    R = zeros(1, size(A)[2])
    error = zeros(p,1)
    for j in 1:p
        q = A[:,j] / norm(A[:,j])
        rT = q' * A
        Q = [Q q]
        R = [R; rT]
        error[j] = norm(A)
        A = A - q * rT
    end
    Q = Q[:,2:end]
    R = R[2:end,:]
    return Q,R,error
end

function QR_greedy(V, p = minimum(size(V)))
    A = V
    Q = zeros(size(A)[1],1)
    R = zeros(1, size(A)[2])
    error = zeros(p,1)
    for j in 1:p
        _,i = findmax(sqrt.(sum(A.^2, dims = 1)))   #Using the argmax
        i = getindex(i,2)
        q = A[:,i] / norm(A[:,i])
        rT = q' * A
        Q = [Q q]
        R = [R; rT]
        error[j] = norm(A)
        A = A - q * rT
    end
    Q = Q[:,2:end]
    R = R[2:end,:]
    return Q,R,error
end

#Part B
A = [1 2 2003 2005; 2 2 2002 2004; 3 2 2001 2003; 4 7 7005 7012]
Q1, R1, error1 = QR_std(A)

display(Q1)
display(R1)
plot(error1)
#Part C


M = load_mat_hw1(1000,100)
Q2, R2, error2 = QR_greedy(M)
display(Q2)
display(R2)
display(norm(M-Q2*R2))
plot(error2)
