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
    println("Ex   OPT   ϵ=$(ϵ_list[1])   ϵ=$(ϵ_list[2])")

    for qapLib_example in qapLib_example_list
        res_1 = "Results/$(qapLib_example)"
        res_2 = "QapLib/$(qapLib_example)2.csv"
        # read matrices G and H
        G = readdlm(m2_file)
        H = readdlm(m1_file)

        p_opt, log_string, dataPoints = pathGraphMatching.pathAlgorithm(G, H, ϵ, ϵ; 
            solveQAP=true,
            return_log=true,
            return_dataPoints=true,
            verbose=false,
            verbose_FW=false
        )
        
        println("-"^30)
        p_opt = pathGraphMatching.permVtM(p_opt)
    end
end

main()