using LinearAlgebra;
using Random;

function load_mat_hw1(n,p)
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
