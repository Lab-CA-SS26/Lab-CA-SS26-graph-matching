using DelimitedFiles
using LinearAlgebra, Permutations
include("src/pathGraphMatching.jl")
using .pathGraphMatching

function main()
    qapLib_example_list = [
        "Chr12c",
        "Chr15a",
        "Chr15c",
        "Chr20b",
        "Chr22b",
        "Esc16b",
        "Rou12",
        "Rou15",
        "Rou20",
        "Tai15a",
        "Tai17a",
        "Tai20a",
        "Tai30a",
        "Tai35a",
        "Tai40a"
    ]
    ts1 = "2026-07-30_14-23-01"
    ts2 = "2026-08-04_16-31-52"
    println("="^30)
    println("Ex   OPT   ϵ=0.1   ϵ=0.01")

    for i in 1:length(qapLib_example_list)
        opt = "QapLib/$(qapLib_example_list[i])Opt.csv"
        res_1 = "Results/$(ts1)/$(ts1)_$(qapLib_example_list[i])_0.1_0.1_true.txt"
        res_2 = "Results/$(ts2)/$(ts2)_$(qapLib_example_list[i])_0.01_0.01_true.txt"
        if isfile(opt) && isfile(res_1) && isfile(res_2)
            G = readdlm("QapLib/$(qapLib_example_list[i])2.csv")
            H = readdlm("QapLib/$(qapLib_example_list[i])1.csv")
            p_opt_qap = readdlm(opt, Int64)
            p_opt_qap = vec(p_opt_qap)
            p_opt_qap = Matrix(Permutation(p_opt_qap))
            println("$(qapLib_example_list[i])   $(pathGraphMatching.qapVal(p_opt_qap, G, H))   $(readlines(res_1)[13])   $(readlines(res_2)[13])")
        end
    end
end

main()