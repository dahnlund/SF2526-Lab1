using LinearAlgebra
using MAT
using Images, Colors
using Plots
using Printf
include("./utils/utils.jl")
include("./utils/zalando_export.jl")

# Read the .mat file
zalando_items = matread("zalando_items.mat")

# A 

# Sandal is item number 6
item5 = get(zalando_items, "item5", 0)

# Testing the following ranks
ranks = [1 2 4 6 10 20 40 50 80 100 120 150 200 250 300 400 500 600 700]
errors = []
for i in ranks
  C,Z = ID_col(item5, i)
  ID_comp = C*Z
  global errors
  errors = [errors; norm(item5 - ID_comp) / norm(item5)]
end
ID_errors = errors

errors = []
for i in ranks
  tmp = svd(item5) #Using Julias pre-implemented SVD
  Uc = tmp.U[:,1:i]
  Sc = Diagonal(tmp.S)[1:i,1:i]
  Vc = tmp.V[:,1:i]
  SVD_comp = Uc*Sc*Vc'
  global errors
  errors = [errors; norm(item5 - SVD_comp) / norm(item5)]
end
SVD_errors = errors

plot_8A = plot(ranks',[ID_errors, SVD_errors],xaxis = ("Rank"),yaxis=("Relative error", :log), label = ["ID_col" "SVD"], legend=:bottomleft, dpi = 200)
savefig(plot_8A, "./figures/plot8A")

display("Relative error from ID_col at rank 100:\n")
display(ID_errors[length(ID_errors)])
display("\nRelative error from SVD at rank 100:\n")
display(SVD_errors[length(SVD_errors)])


# Part B 

item2 = get(zalando_items, "item2", 0) # Our favourite item is the pull-over shirt
for k in 1:3
  zalando_plot(item2[:,k], k) # Plot a sample and save in 'image_ex8'"
end

SVD_comp = svd(item2) #Using Julias pre-implemented SVD
Uc = SVD_comp.U[:,1:3]
C,Z = ID_col(item2, 3)

for k in 1:3
  zalando_plot(Uc[:,k], k * 11)
  zalando_plot(C[:,k], k* 111)
end