using LinearAlgebra

A = [5 -1; 5 7]

beta = 1/sqrt(2)

U = [2/sqrt(40) -3/sqrt(10); 6/sqrt(40) 1/sqrt(10)]

S = [sqrt(80) 0; 0 sqrt(20)]

V =  beta * [1 -1; 1 1]

display(U*S*V') # Verify calculation

display(U*U') #Should be I
display(V*V') #Should be I

#Theoretical norm(A-X), where X is a matrix of rank 1 is sqrt(20)

#Now computated solution

X = [2 2; 6 6]

display(norm(A-X)) #Should give a numerical solution to sqrt(20)

display(sqrt(20))