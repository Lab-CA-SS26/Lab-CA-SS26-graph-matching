using Plots

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
    eps_list=[0.1, 0.01]
    solveQAP_list=[true]

    value_list_1 = []
    value_list_2 = []
    for ϵ in eps_list
        ts=""
        if(ϵ==0.1)
            ts="2026-07-30_14-23-01"
        elseif(ϵ==0.01)
            ts="2026-08-04_16-31-52"
        end

        for solveQAP in solveQAP_list
            pathRes = "Results/$(ϵ)_$(ϵ)_$(solveQAP)"
            for qapLib_example in qapLib_example_list
                value = parse(Float64, strip(readlines("$(pathRes)/$(ts)_$(qapLib_example)_$(ϵ)_$(ϵ)_$(solveQAP).txt")[13]))
                if ϵ == 0.1
                    push!(value_list_1, value)
                elseif ϵ == 0.01
                    push!(value_list_2, value)
                end
            end
        end

    end

    plot(qapLib_example_list, value_list_1, label="ϵ=0.1", marker=:circle)
    plot!(qapLib_example_list, value_list_2, label="ϵ=0.01", marker=:square)

    println(value_list_1)
    println(value_list_2)

end

main()