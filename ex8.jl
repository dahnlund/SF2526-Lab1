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
