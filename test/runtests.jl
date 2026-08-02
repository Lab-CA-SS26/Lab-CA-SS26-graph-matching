using Test
using pathGraphMatching

@testset "pathGraphMatching.jl Tests" begin
    P, _, _ = pathAlgorithm([
        1.0 2.0;
        3.0 4.0
        ],[
        1.0 2.0;
        3.0 4.0
        ],
        0.1, 0.1)
    @test P == [1, 2]

    P, _, _ = pathAlgorithm([
        1.0 2.0;
        3.0 4.0
        ],[
        4.0 3.0;
        2.0 1.0
        ],
        0.1, 0.1)
    @test P == [2, 1]

end