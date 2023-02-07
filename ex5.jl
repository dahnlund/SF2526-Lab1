using LinearAlgebra
using Images, Colors
using Printf
using Plots
include("utils.jl")

# B) Load the video into one matrix
im = load("./testbild_snapshots/testbild_snapshots_0001.png")
vec, size_ = image2vec(im)
A = zeros((size(vec,1), 27))
A[:,1] = vec
for k in 2:27
    global A
    filename_k = @sprintf("./testbild_snapshots/testbild_snapshots_%04d.png",k)
    im_k = load(filename_k)
    vec_k,_ = image2vec(im_k)
    A[:,k] = vec_k
end

# A) Display one sample
ex_vec = A[:,17]
image = reshape(ex_vec, size_)
im1_color = colorview(RGB, image)

# C)

u = A[:,1]
v = ones(size(A)[2],1)

println("|| A - u*v' || = ")
display(norm(A - u*v')) #Should equal 0 since all the images are the same

# D)

QR_time = @elapsed begin # Time the algo

    Q, R, error = QR_greedy(A,10)   # Can use p as small as 1 since A is naturally a rank 1 matrix
    tmp = svd(R)
    U_hat = tmp.U
    S = Diagonal(tmp.S)
    V = tmp.V 
    U = Q*U_hat
end

# Compare QR_time with time from standard SVD 

SVD_time = @elapsed begin
    svd(A)
end


println("\nTime to compute SVD from Algo 1:\n")
display(QR_time)
println("\n Time to compute SVD from Julia's standard method:\n")
display(SVD_time)