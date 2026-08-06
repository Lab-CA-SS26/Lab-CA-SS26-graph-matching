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

    res1_list = [
        "2026-07-30_14-23-01_Chr12c",
        "2026-07-30_14-25-27_Chr15a",
        "2026-07-30_14-28-53_Chr15c",
        "2026-07-30_14-33-55_Chr20b",
        "2026-07-30_14-39-09_Chr22b",
        "2026-07-30_14-56-07_Esc16b",
        "2026-07-30_14-59-29_Rou12",
        "2026-07-30_15-09-08_Rou15",
        "2026-07-30_15-30-12_Rou20",
        "2026-07-30_15-39-55_Tai15a",
        "2026-07-30_15-55-30_Tai17a",
        "2026-07-30_16-15-19_Tai20a",
        "2026-07-30_17-36-29_Tai30a",
        "2026-07-30_19-40-37_Tai35a",
        "2026-07-31_10-25-14_Tai40a"
    ]

    res2_list = [
        "2026-08-04_16-31-52_Chr12c",
        "2026-08-04_16-56-55_Chr15a",
        "2026-08-04_17-21-17_Chr15c",
        "2026-08-04_17-59-41_Chr20b",
        "2026-08-04_18-42-33_Chr22b",
        "2026-08-04_20-43-41_Esc16b",
        "2026-08-04_",
        "2026-08-04_",
        "2026-08-04_",
        "2026-08-04_",
        "2026-08-04_",
    ]


    println("="^30)
    println("Ex   OPT   ϵ=0.1   ϵ=0.01")

    for i in 1:length(qapLib_example_list)
        opt = "QapLib/$(aqpLib_example)Opt.csv"
        res_1 = "Results/$(res1_list[î])_0.1_0.1_true.txt"
        res_2 = "Results/$(res2_list[i])_0.01_0.01_true.txt"
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