using LinearAlgebra
using Random

function load_mat_hw1(n,p)  #From the given file "load_mat_hw1.jl"
    Random.seed!(0);
    p1 = Integer(floor(3*p/4));
    p2 = p-p1;
    Q = randn(n,p1);
    Z1 = Q*Diagonal(randn(1,p1).*exp.(-abs.(1:p1)))*randn(p1,p1);
    I = sortperm(vec(sum(abs.(Z1); dims=1)));
    Z1 = Z1[:,I];
    Q = randn(n,p2);
    Z2 = Q*Diagonal(randn(1,p2).*exp.(-abs.(1:p2)))*randn(p2,p2);
    Z = hcat(Z1,Z2);
    return Z;
end

X = load_mat_hw1(1000,100)

Xd = svd(X)
error = 1e-10
rank = (Xd.S .> error)' * (Xd.S .> error)  #Least rank to satisfy error requirement
U = Xd.U[:, 1:rank]
S= Diagonal(Xd.S[1:rank])
V = Xd.V[:,1:rank]

println("Least rank that satisfies the error requirement:")
println(rank)
println("\nError:")
display(norm(X-U*S*V')) #Should be less than error requirement 

# Part B 

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


# Part B 

Q, R, _ = QR_greedy(X)
tmp = svd(R)
U_hat = tmp.U
S = Diagonal(tmp.S)
V = tmp.V 
U = Q*U_hat

println("\nRank when using SVD from Algorithm 1:")
display((tmp.S .> 1e-12)'*(tmp.S .> 1e-12))




