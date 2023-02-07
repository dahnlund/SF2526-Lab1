using LinearAlgebra
using Plots
using Random
using StatsPlots, DataFrames
include("utils/utils.jl")

#Part B
A = [1 2 2003 2005; 2 2 2002 2004; 3 2 2001 2003; 4 7 7005 7012]
Q1, R1, error1 = QR_std(A)

display(Q1)
display(R1)
plot1 = plot(error1, yaxis=:log)
savefig(plot1, "./figures/plot1")

#Part C


M = load_mat_hw1(1000,100)
Q2, R2, error2 = QR_std(M)
display(norm(M-Q2*R2))

# Part D

M = load_mat_hw1(1000,100)
Q2, R2, error3 = QR_greedy(M)
display(norm(M-Q2*R2))
plot2 = plot([error2, error3],yaxis=:log, label = ["Standard" "Greedy"], legend=:bottomleft)
savefig(plot2, "./figures/plot2")