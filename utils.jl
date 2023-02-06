"""
This is the file where all the used functions in the lab is placed
"""

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
        A = A - q * rT
        error[j] = norm(A)
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
        A = A - q * rT
        error[j] = norm(A)
    end
    Q = Q[:,2:end]
    R = R[2:end,:]
    return Q,R,error
end

function image2vec(im)
    im_mat = channelview(im)
    n1, n2, n3 = size(im_mat)
    vec = reshape(im_mat, n1*n2*n3,1)
    return float.(vec), (n1,n2,n3)
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

    return U[:,1:k],S[1:k,1:k],V[:,1:k]
end
