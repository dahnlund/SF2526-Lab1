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

#Create matrix representing video

im = load("./roundabout_snapshots/roundabout_snapshots_0001.png")
vec, size_ = image2vec(im)
A = zeros((size(vec,1), 56))
println(size(A))
A[:,1] = vec
for k in 2:56
    global A
    filename_k = @sprintf("./roundabout_snapshots/roundabout_snapshots_%04d.png",k)
    im_k = load(filename_k)
    vec_k,_ = image2vec(im_k)
    A[:,k] = vec_k
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
        A = A - q * rT
        error[j] = norm(A)
    end
    Q = Q[:,2:end]
    R = R[2:end,:]
    return Q,R,error
end

# Part A 

Q, R, error = QR_greedy(A)
tmp = svd(R)
U_hat = tmp.U
S = Diagonal(tmp.S)
V = tmp.V
U = Q*U_hat
plot_6A = plot(error)
savefig(plot_6A, "./figures/plot_6A")

# Part C

for k = [5 10 20]
    A_comp = U[:,1:k]*S[1:k,1:k]*V[:,1:k]'
    for i in 1:56
        ex_vec = A_comp[:,i]
        image = reshape(ex_vec, size_)
        image = map(clamp01nan, image)
        image_name = @sprintf("./compressed_images_rank%01d/comp%04d.png",k,i)
        save(image_name, colorview(RGB, image))
    end
end