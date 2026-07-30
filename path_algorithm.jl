using Revise
includet("GraphMatchingUtils.jl")
using .GraphMatchingUtils
includet("TestingUtils.jl")
using .TestingUtils
using TOML
using DataFrames, CSV, DelimitedFiles, Dates
using LinearAlgebra, Permutations
using FrankWolfe

function main()
    # read input and configurations from config.toml (input example given in config.template.toml)
    config = TOML.parsefile("config.toml")
    solveQAP = config["dataInput"]["solveQAP"]  # if true, the QAP is solved instead of the graph matching problem
    ϵ_λ_f = config["dataInput"]["epsilon_lambda_f"] # threshold for change in Fλ between iterations
    ϵ_λ_p = config["dataInput"]["epsilon_lambda_p"] # threshold for change in P between iterations
    print_FrankWolfe = config["printing"]["print_FrankWolfe"]   # whether to print FrankWolfe's output or not

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

        p_opt, log_string, dataPoints = pathAlgorithm(G, H, ϵ_λ_f, ϵ_λ_p; 
            solveQAP=solveQAP,
            return_log=true,
            return_dataPoints=true,
            verbose=true,
            verbose_FW=print_FrankWolfe
        )
        
        println("-"^30)
        p_opt = GraphMatchingUtils.permVtM(p_opt)
        println("Solving QAP: ", solveQAP)
        println("GM Cost:")
        println("F0: ", f0(p_opt, G, H))
        println("F1: ", f1(p_opt, G, H))
        p_opt_qap = readdlm("QapLib/$(qapLib_example)Opt.csv", Int64)
        p_opt_qap = vec(p_opt_qap)
        p_opt_qap = Matrix(Permutation(p_opt_qap))
        push!(optVals, GraphMatchingUtils.qapVal(p_opt_qap, G, H))
        push!(algVals, GraphMatchingUtils.qapVal(p_opt, G, H))
        println(optVals)
        println("MIN       ALG")
        println(last(optVals), "       " , last(algVals))
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        results_filename = "Results/$(timestamp)_$(qapLib_example)_$(ϵ_λ_f)_$(ϵ_λ_p)_$(solveQAP).txt"
        write(results_filename, log_string)

        plotAll(dataPoints.λ_list, dataPoints.f0_list, dataPoints.f1_list, dataPoints.fλ_list, "$(qapLib_example) ϵλf=$(ϵ_λ_f) ϵλp=$(ϵ_λ_p) $(solveQAP ? "QAP" : "GM")")
    
        println("")
        println("NAME           MIN         ALG")
        for i in 1:length(optVals)
            println(qapLib_example_list[i], "       " ,optVals[i], "       ", algVals[i])
        end
    end
    
    
    
    #=
    # save results to file
    open(results_filename, "w") do io
        println(io, "Timestamp: $(timestamp)")
        println(io)
        println(io, "="^60)
        println(io, "Results for graph matching/QAP problem")
        println(io, "="^60)
        println(io)
        println(io, "QapLib file: $(qapLib_example)")
        println(io, "ϵ_λ_f: $(ϵ_λ_f)")
        println(io, "ϵ_λ_p: $(ϵ_λ_p)")
        println(io, "solveQAP: $(solveQAP)")
        println(io)
        println(io, "Runtime: $(elapsed_time) seconds")
        println(io, "λ Iterations: $(count_iter)")
        println(io)
        if !solveQAP
            println(io, "Cost at start:")
            println(io, "F0: $(f0(p_start, G, H))")
            println(io, "F1: $(f1(p_start, G, H))")
            println(io)
            println(io, "Cost at end:")
            println(io, "F0: $(f0(p_opt, G, H))")
            println(io, "F1: $(f1(p_opt, G, H))")
        else
            println(io, "Value of QAP")
            println(io, "$(GraphMatchingUtils.qapVal(p_opt, G, H))")
            println(io, "Optimal: $(GraphMatchingUtils.qapVal(p_opt_qap, G, H))")
        end
        println(io)
        println(io, "-"^60)
        println(io, "Resulting Matrix P")
        println(io, "-"^60)
        println(io)
        show(io, "text/plain", two_row(Permutation(p_opt)))
        if solveQAP
            println(io)
            println(io, "-"^60)
            println(io, "Optimal Matrix P")
            println(io, "-"^60)
            println(io)
            show(io, "text/plain", two_row(Permutation(p_opt_qap)))
        end
    end
    println("Results saved")

    if !solveQAP
        plotTitle = "$(qapLib_example) ϵλf=$(ϵ_λ_f) ϵλp=$(ϵ_λ_p) GM"
    else
        plotTitle = "$(qapLib_example) ϵλf=$(ϵ_λ_f) ϵλp=$(ϵ_λ_p) QAP"
    end
    plotAll(λ_list, f0_list, f1_list, fλ_list, plotTitle)

    println("END")
    println("-----------------------")
    =#
end

main()