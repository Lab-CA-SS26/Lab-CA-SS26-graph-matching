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
    qapLib_example = config["dataInput"]["qapLib_example"]  # see https://qaplib.mgi.polymtl.ca/
    m1_file = "QapLib/$(qapLib_example)1.csv"   # contains first adjacency matrix
    m2_file = "QapLib/$(qapLib_example)2.csv"   # contains second adjacency matrix
    ϵ_λ_f = config["dataInput"]["epsilon_lambda_f"] # threshold for change in Fλ between iterations
    ϵ_λ_p = config["dataInput"]["epsilon_lambda_p"] # threshold for change in P between iterations
    print_FrankWolfe = config["printing"]["print_FrankWolfe"]   # whether to print FrankWolfe's output or not

    println("-----------------------")
    println("START")
    
    # read matrices G and H
    G = readdlm(m1_file)
    H = readdlm(m2_file)
    
    # if graphs have different sizes extend the smaller one by zero rows and columns (as stated in the paper)
    diffSize = size(G,1)-size(H,1)
    if diffSize > 0
        # G is greater
        H = cat(H,zeros(diffSize,diffSize); dims=(1,2))
    elseif diffSize < 0
        # H is greater
        diffSize = abs(diffSize)
        G = cat(G,zeros(diffSize,diffSize); dims=(1,2))
    end
    m_size = size(G,1)
    
    println("G:")
    display(G)
    println("H:")
    display(H)
    
    println("Start timer")
    t1 = time()
    
    # allocate fixed space for the gradient matrices so that they don't allocate new space in each calculation
    storage0 = Matrix{Float64}(undef, m_size, m_size)
    storage1 = Matrix{Float64}(undef, m_size, m_size)

    # Start with P as the identity matrix
    p_start = Matrix(1.0I, m_size, m_size)
    lmo = FrankWolfe.BirkhoffPolytopeLMO() #via Hungarian algorithm


    # find initial minimum of F0 (F1 for QAP)
    # TODO use Newton instead of FrankWolfe for initialization as stated in paper's implementation details
    if !solveQAP
        init_f   = FλForP(0.0, G, H)
        init_∇!  = ∇FλForP!(storage0, storage1, 0.0, G, H)
    else
        init_f   = FλForP_QAP(0.0, G, H)
        init_∇!  = ∇FλForP_QAP!(storage0, storage1, 0.0, G, H)
    end
    
    global p_opt, _ = FrankWolfe.frank_wolfe(
    init_f, init_∇!, lmo, p_start;
    epsilon = 1e-8,
    max_iteration = 10_000,
    verbose=print_FrankWolfe
    )

    # dλ_min is minimum possible change in λ between iterations as stated in the paper
    global dλ_min = 1.0e-05
    # change in λ is dynamically adjusted; starts at minimum
    global dλ = dλ_min
    # begin with λ=0; iteratively increase up until 1
    global λ = 0.0
    
    # redefine f0, f1 and fλ depending on whether the QAP should be solved or not, s.t. f0 is always convex and f1 is always concave.
    if !solveQAP
        f0 = GraphMatchingUtils.f0
        f1 = GraphMatchingUtils.f1
        fλ = GraphMatchingUtils.fλ
    else
        f0 = (P, G, H) -> -GraphMatchingUtils.f1(P, G, H)
        f1 = (P, G, H) -> -GraphMatchingUtils.f0(P, G, H)
        fλ = GraphMatchingUtils.fλ_QAP
    end
    
    count_iter = 0
    λ_list = [λ]
    f0_list = [f0(p_opt,G,H)]
    f1_list = [f1(p_opt,G,H)]
    fλ_list = [fλ(p_opt,λ,G,H)]

    while(λ < 1.0)
        count_iter += 1
        # set first possible value for λ_new
        local λ_new = λ + dλ

        # calculate local optimum w.r.t. initial λ_new
        if !solveQAP
            fλ_new_minimize = FλForP(λ_new, G, H)
            ∇fλ_new_minimize = ∇FλForP!(storage0, storage1, λ_new, G, H)
        else
            fλ_new_minimize = FλForP_QAP(λ_new, G, H)
            ∇fλ_new_minimize = ∇FλForP_QAP!(storage0, storage1, λ_new, G, H)
        end
        p_new, _ = frank_wolfe(
            fλ_new_minimize, ∇fλ_new_minimize, lmo, p_opt; 
            epsilon = 1e-8,
            max_iteration = 10_000,
            verbose=print_FrankWolfe
        )
        p_change_normed = norm(p_new - p_opt) / sqrt(2 * m_size)

        p_last::Union{Nothing, Matrix{Float64}} = nothing

        # update dλ until criterion is met
        # TODO implemented new stopping criterion. Need to still find out ϵ_f and ϵ_p values from FrankWolfe implementation and calculate ϵ_λ_f and ϵ_λ_p with added input M.
        # d_λ is doubled until one value is larger than it's threshold (or new λ is already 1)
        while abs(fλ(p_new,λ_new,G,H)-fλ(p_opt,λ,G,H)) < ϵ_λ_f   &&   p_change_normed < ϵ_λ_p   &&   λ_new < one(Float64)
            global dλ = 2*dλ
            λ_new = min(λ + dλ, one(Float64))
            println("dλ = ", dλ)

            if !solveQAP
                fλ_new_minimize = FλForP(λ_new, G, H)
                ∇fλ_new_minimize = ∇FλForP!(storage0, storage1, λ_new, G, H)
            else
                fλ_new_minimize = FλForP_QAP(λ_new, G, H)
                ∇fλ_new_minimize = ∇FλForP_QAP!(storage0, storage1, λ_new, G, H)
            end
            p_last = p_new
            p_new, _ = frank_wolfe(
                fλ_new_minimize, ∇fλ_new_minimize, lmo, p_opt; 
                epsilon = 1e-8,
                max_iteration = 10_000,
                verbose = print_FrankWolfe
            )
            p_change_normed = norm(p_new - p_opt) / sqrt(2 * m_size)
        end
        
        # if the last while loop's condition is not met (anymore), dλ is one step too large and can be halved once directly
        global dλ = max(dλ/2,dλ_min)
        λ_new = λ + dλ
        println("dλ = ", dλ)
        if !isnothing(p_last)
            p_new = p_last
        else
            if !solveQAP
                fλ_new_minimize = FλForP(λ_new, G, H)
                ∇fλ_new_minimize = ∇FλForP!(storage0, storage1, λ_new, G, H)
            else
                fλ_new_minimize = FλForP_QAP(λ_new, G, H)
                ∇fλ_new_minimize = ∇FλForP_QAP!(storage0, storage1, λ_new, G, H)
            end
            p_new, _ = frank_wolfe(
                fλ_new_minimize, ∇fλ_new_minimize, lmo, p_opt; 
                epsilon = 1e-8,
                max_iteration = 10_000,
                verbose = print_FrankWolfe
            )
            p_change_normed = norm(p_new - p_opt) / sqrt(2 * m_size)
        end

        # d_λ is halved until both values are smaller than their thresholds (or dλ is already at minimum)
        while (abs(fλ(p_new,λ_new,G,H)-fλ(p_opt,λ,G,H)) > ϵ_λ_f   ||   p_change_normed > ϵ_λ_p)   &&   dλ > dλ_min
            global dλ = max(dλ/2,dλ_min)
            λ_new = min(λ + dλ, one(Float64))
            println("dλ = ", dλ)

            if !solveQAP
                fλ_new_minimize = FλForP(λ_new, G, H)
                ∇fλ_new_minimize = ∇FλForP!(storage0, storage1, λ_new, G, H)
            else
                fλ_new_minimize = FλForP_QAP(λ_new, G, H)
                ∇fλ_new_minimize = ∇FλForP_QAP!(storage0, storage1, λ_new, G, H)
            end
            p_new, _ = frank_wolfe(
                fλ_new_minimize, ∇fλ_new_minimize, lmo, p_opt; 
                epsilon = 1e-8,
                max_iteration = 10_000,
                verbose = print_FrankWolfe
            )
            p_change_normed = norm(p_new - p_opt) / sqrt(2 * m_size)
        end
        println("λ: ",λ," + ",dλ," = ",λ_new)
        global λ = λ_new
        # criterion is met, λ is set correctly and p_new contans the local optimum w.r.t. the new λ. Set p_opt to p_new for next iteration.

        p_opt = p_new

        push!(λ_list, λ)
        push!(f0_list, f0(p_opt,G,H))
        push!(f1_list, f1(p_opt,G,H))
        push!(fλ_list, fλ(p_opt,λ,G,H))

        # stop immediately if FrankWolfe arrives at a Permutationmatrix as this is a feasible minimum
        if GraphMatchingUtils.isPerm(p_opt)
            println("DONE")
            println("P:")
            display(two_row(Permutation(p_opt)))
            println("Inv(P)")
            display(two_row(inv(Permutation(p_opt))))
            break
        else
            # println("P:")
            # display(p_opt)
            println("CONTINUE")
        end
    end
    p_opt = inv(p_opt) # invert P to get the correct mapping from H to G
    elapsed_time = time() - t1
    println("Elapsed time: ", elapsed_time, " seconds")

    println("Solving QAP: ", solveQAP)
    println("Cost at start:")
    println("F0: ", f0(p_start, G, H))
    println("F1: ", f1(p_start, G, H))
    println("Cost at end:")
    println("F0: ", f0(p_opt, G, H))
    println("F1: ", f1(p_opt, G, H))
    println("Value of QAP")
    println("G -> H: ", GraphMatchingUtils.qapVal(p_opt, G, H))
    println("H -> G: ", GraphMatchingUtils.qapVal(p_opt, H, G))
    println("Optimum of ",qapLib_example,": ")
    p_opt_qap = readdlm("QapLib/$(qapLib_example)Opt.csv", Int64)
    p_opt_qap = vec(p_opt_qap)
    p_opt_qap = Matrix(Permutation(p_opt_qap))
    display(two_row(Permutation(p_opt_qap)))
    println(GraphMatchingUtils.qapVal(p_opt_qap, H, G))

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_filename = "Results/$(timestamp)_$(qapLib_example)_$(ϵ_λ_f)_$(ϵ_λ_p)_$(solveQAP).txt"
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
            println(io, "F0: $(f0(p_start, H, G))")
            println(io, "F1: $(f1(p_start, H, G))")
            println(io)
            println(io, "Cost at end:")
            println(io, "F0: $(f0(p_opt, G, H))")
            println(io, "F1: $(f1(p_opt, G, H))")
            println(io, "F0: $(f0(p_opt, H, G))")
            println(io, "F1: $(f1(p_opt, H, G))")
        else
            println(io, "Value of QAP")
            println(io, "G -> H: $(GraphMatchingUtils.qapVal(p_opt, G, H)) (ignore)")
            println(io, "H -> G: $(GraphMatchingUtils.qapVal(p_opt, H, G))")
            println(io, "Optimal: $(GraphMatchingUtils.qapVal(p_opt_qap, H, G))")
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
end

main()