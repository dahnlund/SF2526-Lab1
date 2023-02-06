using LinearAlgebra
using Random
include("utils.jl")


# Part A
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

Q, R, _ = QR_greedy(X)
tmp = svd(R)
U_hat = tmp.U
S = Diagonal(tmp.S)
V = tmp.V 
U = Q*U_hat

println("\nRank when using SVD from Algorithm 1:")
display((tmp.S .> 1e-12)'*(tmp.S .> 1e-12))




