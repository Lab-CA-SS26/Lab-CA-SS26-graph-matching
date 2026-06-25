using Revise
includet("GraphMatchingUtils.jl")
using .GraphMatchingUtils
using TOML
using DataFrames, CSV, DelimitedFiles, LinearAlgebra, Permutations
using FrankWolfe

function main()
    # read input and configurations from config.toml (input example given in config.template.toml)
    config = TOML.parsefile("config.toml")
    qapLib_example = config["dataInput"]["qapLib_example"]
    m1_file = "QapLib/$(qapLib_example)1.csv"
    m2_file = "QapLib/$(qapLib_example)2.csv"
    ϵ_λ_f = config["dataInput"]["epsilon_lambda_f"]
    ϵ_λ_p = config["dataInput"]["epsilon_lambda_p"]
    print_FrankWolfe = config["printing"]["print_FrankWolfe"]

    println("-----------------------")
    println("START")

    # read matrices G and H
    G = readdlm(m1_file)
    H = readdlm(m2_file)

    # if graphs have different sizes extend the smaller one by zero rows and columns
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

    # define F0 and F1 and their gradients dependent only on P as G and H are constant matrices from here on
    f0_minimize(P) = GraphMatchingUtils.f0(P,G,H)
    ∇f0_minimize!(storage, P) = GraphMatchingUtils.∇f0!(storage, P, G, H)
    f1_minimize(P) = GraphMatchingUtils.f1(P,G,H)
    ∇f1_minimize!(storage, P) = GraphMatchingUtils.∇f1!(storage, P, G, H)
    # allocate fixed space for the gradient matrices so that they don't allocate new space for each calculation
    storage0 = Matrix{Float64}(undef, m_size, m_size)
    storage1 = Matrix{Float64}(undef, m_size, m_size)

    # Start with P as the identity matrix
    p_start = Matrix(1.0I, m_size, m_size)
    lmo = FrankWolfe.BirkhoffPolytopeLMO() #via Hungarian algorithm

    history = []

    prev_f = Ref{Union{Nothing,Float64}}(nothing)
    prev_x = Ref{Any}(nothing)

    callback = function (state, args...)
        f_current = state.primal
        x_current = state.x

        f_change =
            prev_f[] === nothing ? NaN : abs(f_current - prev_f[])

        x_change =
            prev_x[] === nothing ? NaN : norm(x_current - prev_x[])

        if state.t == 1
            f_change_sum = 0
        else
            f_change_sum = history[end].f_change_sum + f_change
        end

        push!(history, (
            iter = state.t,
            primal = state.primal,
            dual = state.dual,
            dual_gap = state.dual_gap,
            f_change = f_change,
            f_change_sum = f_change_sum,
            x_change = x_change,
            gamma = state.gamma,
        ))

        prev_f[] = f_current
        prev_x[] = copy(x_current)

        return true
    end

    # find minimum of F0
    # TODO use Newton instead of FrankWolfe for initialization as stated in paper's implementation details
    global p_opt, _ = FrankWolfe.frank_wolfe(
    f0_minimize, ∇f0_minimize!, lmo, p_start;
    epsilon = 1e-8,
    max_iteration = 10_000,
    callback = callback,
    )

    # dλ_min is minimum possible change in λ between iterations as stated in the paper
    global dλ_min = 1.0e-05
    # change in λ is dynamically adjusted; starts at minimum
    global dλ = dλ_min
    # begin with λ=0; iteratively increase up until 1
    global λ = 0
    # Fλ is linear combination of F0 & F1 dependent on λ starting at F0
    fλ(P, λ) = (1-λ) * GraphMatchingUtils.f0(P, G, H)  +  λ * GraphMatchingUtils.f1(P, G, H)

    while(λ < 1.0)
        # set first possible value for λ_new and find best one in the following part
        local λ_new = λ + dλ


        # calculate local optimum for λ_new
        fλ_new_minimize(P) = fλ(P, λ_new)
        ∇fλ_new_minimize!(storageλ_new, P) = GraphMatchingUtils.∇fλ!(storageλ_new, storage0, storage1, P, λ_new, G, H)
        p_new, _ = frank_wolfe(
            fλ_new_minimize, ∇fλ_new_minimize!, lmo, p_opt; 
            epsilon = 1e-8,
            max_iteration = 10_000,
            callback = callback,
            verbose=print_FrankWolfe
        )
        #println(abs(fλ(p_new,λ_new)-fλ(p_opt,λ_new))," = ",history[end].f_change_sum," ?")

        # update dλ until criterion is met
        # TODO implemented new stopping criterion. Need to still find out ϵ_f and ϵ_p values from FrankWolfe implementation and calculate ϵ_λ_f and ϵ_λ_p with added input M.
        # Is ϵ_λ_f just epsilon from the input?
        # first d_λ is doubled until one value is larger than it's threshold (or new λ is larger than 1)
        while abs(fλ(p_new,λ_new)-fλ(p_opt,λ)) < ϵ_λ_f   &&   norm(p_new - p_opt) < ϵ_λ_p   &&   λ_new < one(Float64)
            # println("|",fλ(p_opt,λ_new)," - ",fλ(p_opt,λ),"| = ")
            println(abs(fλ(p_new,λ_new)-fλ(p_opt,λ)), " < " , ϵ_λ_f, " AND ")
            println(norm(p_new - p_opt), " < " , ϵ_λ_p)
            global dλ = min(2*dλ,one(Float64))
            λ_new = λ + dλ
            println("dλ = ", dλ)

            fλ_new_minimize(P) = fλ(P, λ_new)
            ∇fλ_new_minimize!(storageλ_new, P) = GraphMatchingUtils.∇fλ!(storageλ_new, storage0, storage1, P, λ_new, G, H)
            p_new, _ = frank_wolfe(
                fλ_new_minimize, ∇fλ_new_minimize!, lmo, p_opt; 
                epsilon = 1e-8,
                max_iteration = 10_000,
                callback = callback,
                verbose = print_FrankWolfe
            )
            #println(abs(fλ(p_new,λ_new)-fλ(p_opt,λ_new))," = ",history[end].f_change_sum," ?")
        end

        # now d_λ is halved until both values are smaller than their thresholds (or dλ is smaller than minimum)
        while (abs(fλ(p_new,λ_new)-fλ(p_opt,λ)) > ϵ_λ_f   ||   norm(p_new - p_opt) > ϵ_λ_p)   &&   dλ > dλ_min
            # println("|",fλ(p_opt,λ_new)," - ",fλ(p_opt,λ),"| = ")
            println(abs(fλ(p_new,λ_new)-fλ(p_opt,λ)), " > " , ϵ_λ_f, " OR ")
            println(norm(p_new - p_opt), " > " , ϵ_λ_p)
            global dλ = max(dλ/2,dλ_min)
            λ_new = λ + dλ
            println("dλ = ", dλ)

            fλ_new_minimize(P) = fλ(P, λ_new)
            ∇fλ_new_minimize!(storageλ_new, P) = GraphMatchingUtils.∇fλ!(storageλ_new, storage0, storage1, P, λ_new, G, H)
            p_new, _ = frank_wolfe(
                fλ_new_minimize, ∇fλ_new_minimize!, lmo, p_opt; 
                epsilon = 1e-8,
                max_iteration = 10_000,
                callback = callback,
                verbose = print_FrankWolfe
            )
            #println(abs(fλ(p_new,λ_new)-fλ(p_opt,λ_new))," = ",history[end].f_change_sum," ?")
        end
        println("λ: ",λ," + ",dλ," = ",λ_new)
        global λ = λ_new
        # criterion is met, λ is set correctly

        # set λ as constant and define F_λ and it's gradient only over P
        fλ_minimize(P) = fλ(P, λ)
        ∇fλ_minimize!(storageλ, P) = GraphMatchingUtils.∇fλ!(storageλ, storage0, storage1, P, λ, G, H)

        # use FrankWolfe Algorithm with adjusted Fλ function
        # starting at the current doubly stochastic matrix and save solution as new minimum
        local p_temp = p_opt
        global p_opt, _ = frank_wolfe(
            fλ_minimize, ∇fλ_minimize!, lmo, p_temp; 
            epsilon = 1e-8,
            max_iteration = 10_000,
            callback = callback,
            verbose = print_FrankWolfe
        )

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

    println("Cost at start:")
    println("F0: ", GraphMatchingUtils.f0(p_start, G, H))
    println("F1: ", GraphMatchingUtils.f1(p_start, G, H))
    println("Cost at end:")
    println("F0: ", GraphMatchingUtils.f0(p_opt, G, H))
    println("F1: ", GraphMatchingUtils.f1(p_opt, G, H))
    println("Value of QAP")
    println("G -> H: ", GraphMatchingUtils.qapVal(p_opt, G, H))
    println("H -> G: ", GraphMatchingUtils.qapVal(p_opt, H, G))
    println("Optimum of ",qapLib_example,": ")
    p_opt = readdlm("QapLib/$(qapLib_example)Opt.csv", Int64)
    p_opt = vec(p_opt)
    println(p_opt)
    p_opt = Matrix(Permutation(p_opt))
    println(GraphMatchingUtils.qapVal(p_opt, G, H))

    df_history = DataFrame(history)
    CSV.write("frank_wolfe_history.csv", df_history)
    println("History saved")

    println("END")
    println("-----------------------")
end

main()