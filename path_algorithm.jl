using Revise
includet("GraphMatchingUtils.jl")
using .GraphMatchingUtils
using CSV, DelimitedFiles, SparseArrays, LinearAlgebra, Permutations
using FrankWolfe, Hungarian

println("-----------------------")
println("START")


# TODO find out how to correctly set ϵ_λ
ϵ_λ = 10.0

# read matrices G and H
m1_file = "QapLib\\Chr12c1.csv"
m2_file = "QapLib\\Chr12c2.csv"
G = readdlm(m1_file, Int)
H = readdlm(m2_file, Int)

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

println("G:")
display(G)
println("H:")
display(H)

# calculate F0 and F1 and there gradients dependent only on P as G and H are constant matrices
f0_minimize(P) = GraphMatchingUtils.f0(P,G,H)
∇f0_minimize!(storage, P) = GraphMatchingUtils.∇f0!(storage, P, G, H)
storage0 = Matrix{Float64}(undef, m_size, m_size)
f1_minimize(P) = GraphMatchingUtils.f1(P,G,H)
∇f1_minimize!(storage, P) = GraphMatchingUtils.∇f1!(storage, P, G, H)
storage1 = Matrix{Float64}(undef, m_size, m_size)

# Start with P as the identity matrix
p_start = Matrix(1.0I, m_size, m_size)
lmo = FrankWolfe.BirkhoffPolytopeLMO() #via Hungarian algorithm
# find minimum of F0
# TODO use Newton instead of FrankWolfe for initialization as stated in paper's implementation details
p_opt, _ = frank_wolfe(f0_minimize, ∇f0_minimize!, lmo, p_start; verbose=true);
display(p_opt)

# dλ_min is minimum change in λ between iterations as stated in the paper
dλ_min = 1.0e-05
# change in λ is dynamically adjusted; starts at minimum
dλ = dλ_min
# begin with λ=0 and iteratively increase up until 1
λ = 0
# Fλ is linear combination of F0 and F1 dependent on λ starting at F0
f_λ(P, λ) = (1-λ) * GraphMatchingUtils.f0(P, G, H)  +  λ * GraphMatchingUtils.f1(P, G, H)

while(λ < 1.0)
    # set first possible value for λ_new and find best one in the following part
    λ_new = λ + dλ

    # update dλ until criterion is met
    # TODO change criterion as stated in paper's implementation details. Need to understand ϵ first.
    # first d_λ is doubled until the value is larger than ϵ_λ (or new λ is larger than 1)
    while abs(f_λ(p_opt,λ_new)-f_λ(p_opt,λ)) ≤ ϵ_λ   &&   λ_new < one(Float64)
        println(abs(f_λ(p_opt,λ_new)-f_λ(p_opt,λ)), " ≤ " , ϵ_λ)
        dλ = min(2*dλ,one(Float64))
        λ_new = λ + dλ
        println("dλ = ", dλ)
    end
    # now d_λ is halved until the value is slightly smaller then ϵ_λ (or d_λ is smaller than minimum)
    while abs(f_λ(p_opt,λ_new)-f_λ(p_opt,λ)) > ϵ_λ && dλ > dλ_min
        println(abs(f_λ(p_opt,λ_new)-f_λ(p_opt,λ)), " > " , ϵ_λ)
        dλ = max(dλ/2,dλ_min)
        λ_new = λ + dλ
        println("dλ = ", dλ)
    end
    println("λ: ",λ," → ",λ_new)
    λ = λ_new
    # criterion is met, λ is set correctly

    # set λ as constant and define F_λ and it's gradient only over P
    f_λ_minimize(P) = f_λ(P, λ)
    ∇f_λ_minimize!(storageλ, P) = GraphMatchingUtils.∇f_λ!(storageλ, storage0, storage1, P, λ, G, H)

    # use FrankWolfe Algorithm with adjusted Fλ function
    # starting at the current doubly stochastic matrix and save solution as new minimum
    p_temp = p_opt
    global p_opt, _ = frank_wolfe(f_λ_minimize, ∇f_λ_minimize!, lmo, p_temp; verbose=false);
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
println(GraphMatchingUtils.f0(p_start, G, H))
println("Cost at end:")
println(GraphMatchingUtils.f0(p_opt, G, H))
println("Value of QAP")
println(GraphMatchingUtils.qapVal(p_opt, G, H))
println("Optimum for QAP")
p_opt = [7,5,1,3,10,4,8,6,9,11,2,12]
println(p_opt)
p_opt = Matrix(Permutation(p_opt))
println(GraphMatchingUtils.qapVal(p_opt, G, H))
println("END")
println("-----------------------")
