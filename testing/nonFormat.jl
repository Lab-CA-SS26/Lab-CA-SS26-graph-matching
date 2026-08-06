# define callback function and save FW iteration data in "history"
    history = []

    prev_f = Ref{Union{Nothing,Float64}}(nothing)
    prev_x = Ref{Any}(nothing)

    callback = function (state, args...)
        f_current = state.primal
        x_current = state.x

        f_change =
            prev_f[] === nothing ? NaN : abs(f_current - prev_f[])
        decrease =
            prev_f[] === nothing ? NaN : (prev_f[] - f_current > 0)
        f_change_sum =
            state.t === 1 ? 0 : history[end].f_change_sum + f_change
        x_change =
            prev_x[] === nothing ? NaN : norm(x_current - prev_x[])

        push!(history, (
            iter = state.t,
            primal = state.primal,
            dual = state.dual,
            dual_gap = state.dual_gap,
            f_change = f_change,
            decrease = decrease,
            f_change_sum = f_change_sum,
            x_change = x_change,
            gamma = state.gamma,
        ))

        prev_f[] = f_current
        prev_x[] = copy(x_current)

        return true
    end

     #=
    df_history = DataFrame(history)
    CSV.write("frank_wolfe_history.csv", df_history)
    println("History saved")
    =#


 #= not needed with new stopping criterions because FrankWolfe is already called in the while loops above
        # set λ as constant and define F_λ and it's gradient only over P
        if !solveQAP
            fλ_minimize = FλForP(λ, G, H)
            ∇fλ_minimize = ∇FλForP!(storage0, storage1, λ, G, H)
        else
            fλ_minimize = FλForP_QAP(λ, G, H)
            ∇fλ_minimize = ∇FλForP_QAP!(storage0, storage1, λ, G, H)
        end

        # use FrankWolfe Algorithm with adjusted Fλ function
        # starting at the current doubly stochastic matrix and save solution as new minimum
        local p_temp = p_opt
        global p_opt, _ = frank_wolfe(
            fλ_minimize, ∇fλ_minimize, lmo, p_temp; 
            epsilon = 1e-8,
            max_iteration = 10_000,
            callback = callback,
            verbose = print_FrankWolfe
        )
        =#