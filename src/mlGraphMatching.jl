module mlGraphMatching
    using Graphs

    export generate_random_graphs

    function generate_random_graphs(n::Int, d::Float64, p_noise::Float64;
        type::String="Regular"  # Bernoulli, Regular, ErdosRenyi
        )

        (type=="Regular" || type=="R") && return generate_random_regular_graphs(n, Int.(d), p_noise)
        (type=="Bernoulli" || type=="B") && return generate_random_bernoulli_graphs(n, d, p_noise)
        (type=="ErdosRenyi" || type=="E") && return generate_random_erdosRenyi_graphs(n, d, p_noise)
        return nothing
    end

    function generate_random_regular_graphs(n::Int, d::Int, p_noise::Float64)
        A = Graphs.random_regular_graph(n, d)

        return A
    end

    # TODO: Implement
    function generate_random_bernoulli_graphs(n::Int, d::Float64, p_noise::Float64)
        return nothing
    end

    # TODO: Implement
    function generate_random_erdosRenyi_graphs(n::Int, d::Float64, p_noise::Float64)
        return nothing
    end

    function acc(π::Vector{Int}, π_AB::Vector{Int})
        return sum(π .== π_AB) / length(π)
    end

    function nce(A::Matrix{Int}, B::Matrix{Int}, π::Vector{Int})
        sum = 0
        for i in CartesianIndices(A)
            sum += A[i] * B[π[i[1]], π[i[2]]]
        end
        return sum / 2
    end
end