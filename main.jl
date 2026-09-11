include("src/pathGraphMatching.jl")
using .pathGraphMatching
using TOML
using DataFrames, CSV, DelimitedFiles, Dates
using LinearAlgebra, Permutations
using FrankWolfe

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
    println("="^30)
    eps_list=[0.25, 0.1, 0.01]
    solveQAP=true
    return_log=true
    return_dataPoints=true
    verbose=true
    verbose_FW=false

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    qapLib_ex_name = []
    optVals = []
    algVals = []

    for ϵ in eps_list
        pathRes = "Results/$(ϵ)_$(ϵ)_$(solveQAP)_wrongGradient"
        mkdir(pathRes)
        for qapLib_example in qapLib_example_list
            println("="^30)
            G = readdlm("QapLib/$(qapLib_example)2.csv")
            H = readdlm("QapLib/$(qapLib_example)1.csv")
            ϵ_λ_f = ϵ
            ϵ_λ_p = ϵ
            println(qapLib_example)
            println("ϵ_λ_f = $(ϵ_λ_f), ϵ_λ_p = $(ϵ_λ_p)")

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
            push!(qapLib_ex_name, qapLib_example)
            println("MIN       ALG")
            println(last(optVals), "       " , last(algVals))
            results_filename = "$(pathRes)/$(timestamp)_$(qapLib_example)_$(ϵ_λ_f)_$(ϵ_λ_p)_$(solveQAP).txt"
            write(results_filename, log_string)

            dataPoints_filename = "$(pathRes)/$(timestamp)_$(qapLib_example)_$(ϵ_λ_f)_$(ϵ_λ_p)_$(solveQAP)_dataPoints.txt"
            if dataPoints !== nothing
                header = ["lambda" "f0" "f1" "fla"]
                data = [dataPoints.λ_list dataPoints.f0_list dataPoints.f1_list dataPoints.fλ_list]
                matrix = [header; data]
                writedlm(dataPoints_filename, matrix, '\t')
            end

            println("")
            println("NAME           MIN         ALG")
            for i in 1:length(optVals)
                println(qapLib_ex_name[i], "       " ,optVals[i], "       ", algVals[i])
            end

        end
    end
    #=for i in 1:length(qapLib_example_list)
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
    end=#
end

main()