include("src/pathGraphMatching.jl")
using .pathGraphMatching
using TOML
using DataFrames, CSV, DelimitedFiles, Dates
using LinearAlgebra, Permutations
using FrankWolfe

function main()
    ϵ_λ_f=0.5
    ϵ_λ_p=0.5
    solveQAP=false
    return_log=true
    return_dataPoints=true
    verbose=true
    verbose_FW=false

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
    
    optVals = []
    algVals = []
    for qapLib_example in qapLib_example_list
        println("="^30)
        println(qapLib_example)
        m1_file = "QapLib/$(qapLib_example)1.csv"
        m2_file = "QapLib/$(qapLib_example)2.csv"
        # read matrices G and H
        G = readdlm(m2_file)
        H = readdlm(m1_file)

        p_opt, log_string, dataPoints = pathGraphMatching.pathAlgorithm(G, H, ϵ_λ_f, ϵ_λ_p; 
            solveQAP=solveQAP,
            return_log=return_log,
            return_dataPoints=return_dataPoints,
            verbose=verbose,
            verbose_FW=verbose_FW
        )
        
        println("-"^30)
        p_opt = pathGraphMatching.permVtM(p_opt)
        println("Solving QAP: ", solveQAP)
        println("GM Cost:")
        println("F0: ", pathGraphMatching.f0(p_opt, G, H))
        println("F1: ", pathGraphMatching.f1(p_opt, G, H))
        p_opt_qap = readdlm("QapLib/$(qapLib_example)Opt.csv", Int64)
        p_opt_qap = vec(p_opt_qap)
        p_opt_qap = Matrix(Permutation(p_opt_qap))
        push!(optVals, pathGraphMatching.qapVal(p_opt_qap, G, H))
        push!(algVals, pathGraphMatching.qapVal(p_opt, G, H))
        println("MIN       ALG")
        println(last(optVals), "       " , last(algVals))
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        results_filename = "Results/$(timestamp)_$(qapLib_example)_$(ϵ_λ_f)_$(ϵ_λ_p)_$(solveQAP).txt"
        write(results_filename, log_string)

        # plotAll(dataPoints.λ_list, dataPoints.f0_list, dataPoints.f1_list, dataPoints.fλ_list, "$(qapLib_example) ϵλf=$(ϵ_λ_f) ϵλp=$(ϵ_λ_p) $(solveQAP ? "QAP" : "GM")")
        dataPoints_filename = "Results/$(timestamp)_$(qapLib_example)_$(ϵ_λ_f)_$(ϵ_λ_p)_$(solveQAP)_dataPoints.txt"
        if dataPoints !== nothing
            header = ["lambda" "f0" "f1" "fla"]
            data = [dataPoints.λ_list dataPoints.f0_list dataPoints.f1_list dataPoints.fλ_list]
            matrix = [header; data]
            writedlm(dataPoints_filename, matrix, '\t')
        end

        println("")
        println("NAME           MIN         ALG")
        for i in 1:length(optVals)
            println(qapLib_example_list[i], "       " ,optVals[i], "       ", algVals[i])
        end
    end
end

main()