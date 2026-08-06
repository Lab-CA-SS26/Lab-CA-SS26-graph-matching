using DelimitedFiles

# generates random adjacency matrices for two undirected graphs without self-loops
# values are between 0 and one at default, can adjust "factor"
# saved in matrix1.csv and matrix2.csv
matrixSize1 = 5
matrixSize2 = 3
factor = 1
M = rand(Float64, matrixSize1, matrixSize1)
N = rand(Float64, matrixSize2, matrixSize2)

for i = 1:matrixSize1
    M[i,i] = 0
    for j = 1:i-1
        M[j,i] *= factor
        M[i,j] = M[j,i]
    end
end
for i = 1:matrixSize2
    N[i,i] = 0
    for j = 1:i-1
        N[j,i] *= factor
        N[i,j] = N[j,i]
    end
end

display(M)
display(N)

writedlm( "matrix1.csv",  M, ' ')
writedlm( "matrix2.csv",  N, ' ')