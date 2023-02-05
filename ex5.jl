using LinearAlgebra
using Images, Colors
using Printf
using Plots

function image2vec(im)
    im_mat = channelview(im)
    n1, n2, n3 = size(im_mat)
    vec = reshape(im_mat, n1*n2*n3,1)
    return float.(vec), (n1,n2,n3)
end

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

display(norm(A - u*v')) #Should equal 0 since all the images are the same

# D)

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

QR_time = @elapsed begin # Time the algo

    Q, R, error = QR_greedy(A,10)   # Can use p as small as 1 since A is naturally a rank 1 matrix
    tmp = svd(R)
    U_hat = tmp.U
    S = Diagonal(tmp.S)
    V = tmp.V 
    U = Q*U_hat
end

display(norm(A - U*S*V'))

# D) Compare QR_time with time from standard SVD 

SVD_time = @elapsed begin
    svd(A)
end


println("\nTime to compute SVD from Algo 1:\n")
display(QR_time)
println("\n Time to compute SVD from Julia's standard method:\n")
display(SVD_time)