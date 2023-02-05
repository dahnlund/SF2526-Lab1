using LinearAlgebra
using Images, Colors
using Printf
using Plots


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


function randomSVD(A,k,p)
    #Stage A
    G = randn(size(A,2), k+p)
    Y = A*G
    Q,_,_ = QR_greedy(Y)

    #Stage B 
    B = Q'*A
    tmp = svd(B)
    U_hat = tmp.U
    S = Diagonal(tmp.S)
    V = tmp.V
    U = Q*U_hat

    return U,S,V
end

#Create matrix representing video

function image2vec(im)
    im_mat = channelview(im)
    n1, n2, n3 = size(im_mat)
    vec = reshape(im_mat, n1*n2*n3,1)
    return float.(vec), (n1,n2,n3)
end

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
    image_name = @sprintf("./compressed_images/comp%04d.png",i)
    save(image_name, colorview(RGB, image))
end


plot3 = plot(error, yaxis=:log)
savefig(plot3, "./figures/plot3")