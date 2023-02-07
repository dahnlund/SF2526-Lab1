using LinearAlgebra
using MAT
using Images, Colors, Plots
using Printf
include("./utils/utils.jl")


# Read the .mat file
zalando_items = matread("zalando_items.mat")

# A 

# Sandal is item number 6

item5 = get(zalando_items, "item5", 0)


zalando_plot(item5[:,1])
#=
function [C,Z]=ID_col(A,kk)
    #=
    A naive way to compute column ID by via the CPQR-factorization.
    
    kk = number of vectors wanted
    C is selection of the columns of A
    For sufficiently large kk, we hope that
      A approx C*Z
    First: Essentially same as Algorithm 1 (but with pivoting) with kk steps
    =#
    [Q,R,P]=qr(A); 
    Qs=Q(:,1:kk); 
    Rs=R(1:kk,:);
    # Now: Compute the ID col
    R11=Rs(??)
    R12=Rs(??)
    C=A*P(:,??);
    I=eye(kk,kk);
    Z= [I, inv(R11)*R12]*P';
end
#=