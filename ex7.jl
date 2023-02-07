using LinearAlgebra
using Images, Colors
using Printf
using Plots
include("utils/utils.jl")

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


# Part A 

p_limit = 25
error = zeros(p_limit,1)
for p = 1:p_limit
    U,S,V = randomSVD(A, 5, p)
    global A_comp = U*S*V'
    error[p] = norm(A-A_comp)
end

for i in 1:56
    ex_vec = A_comp[:,i]
    image = reshape(ex_vec, size_)
    image = map(clamp01nan, image)
    image_name = @sprintf("./compressed_images_ex7/comp%04d.png",i)
    save(image_name, colorview(RGB, image))
end


plot3 = plot(error, yaxis=:log)
savefig(plot3, "./figures/plot3")

#Part B

SVD_time_julia = @elapsed begin
    svd(A)
end

SVD_time_QR_greedy = @elapsed begin
    Q, R, error = QR_greedy(A, 20)
    tmp = svd(R)
    U_hat = tmp.U
    S = Diagonal(tmp.S)
    V = tmp.V
    U = Q*U_hat
end

SVD_time_Random = @elapsed begin
    randomSVD(A, 5, 15)
end

println("\nCPU-time Julia's standard SVD algo:\n")
println(SVD_time_julia)
println("\nCPU-time SVD from QR_greedy (p = 20):\n")
println(SVD_time_QR_greedy)
println("\nCPU-time random SVD (s = 15):\n")
println(SVD_time_Random)