include("src/mlGraphMatching.jl")
using .mlGraphMatching

function main()
    println(mlGraphMatching.generate_random_graphs(5, 2.0, 0.0, type="R"))
    println("Accuracy: ", mlGraphMatching.acc([1, 2, 3, 4], [1, 3, 2, 4]))
    println("NCE: ", mlGraphMatching.nce([
                                          0 1 0; 
                                          1 0 1; 
                                          0 1 0
                                          ], [
                                          0 1 1; 
                                          1 0 0; 
                                          1 0 0
                                          ], [
                                          2, 1, 3
                                          ]))
end

main()