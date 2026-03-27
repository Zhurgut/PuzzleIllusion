
include("generate_puzzle.jl")

using Test

# 12 solutions
# Piece[Piece(l=0, t=0, b=2, r=-1) Piece(l=1, t=0, b=-6, r=-1) Piece(l=1, t=0, b=4, r=-2) Piece(l=2, t=0, b=4, r=-3) Piece(l=3, t=0, b=-1, r=0); 
#     Piece(l=0, t=-2, b=3, r=4) Piece(l=-4, t=6, b=4, r=6) Piece(l=-6, t=-4, b=4, r=-6) Piece(l=6, t=-4, b=6, r=4) Piece(l=-4, t=1, b=-2, r=0); 
#     Piece(l=0, t=-3, b=1, r=-6) Piece(l=6, t=-4, b=-6, r=6) Piece(l=-6, t=-4, b=-6, r=6) Piece(l=-6, t=-6, b=-4, r=4) Piece(l=-4, t=2, b=-2, r=0); 
#     Piece(l=0, t=-1, b=5, r=-4) Piece(l=4, t=6, b=-6, r=4) Piece(l=-4, t=6, b=4, r=-6) Piece(l=6, t=4, b=-6, r=-4) Piece(l=4, t=2, b=-2, r=0); 
#     Piece(l=0, t=-5, b=0, r=3) Piece(l=-3, t=6, b=0, r=5) Piece(l=-5, t=-4, b=0, r=2) Piece(l=-2, t=6, b=0, r=2) Piece(l=-2, t=2, b=0, r=0)]

# 3 solutions, one 2x2 block can be reassembled into itself (not just rotated)
# Piece[Piece(l=0, t=0, b=3, r=-2) Piece(l=2, t=0, b=5, r=-3) Piece(l=3, t=0, b=5, r=-1) Piece(l=1, t=0, b=6, r=-2) Piece(l=2, t=0, b=4, r=0); 
#     Piece(l=0, t=-3, b=1, r=6) Piece(l=-6, t=-5, b=-5, r=-7) Piece(l=7, t=-5, b=5, r=6) Piece(l=-6, t=-6, b=7, r=-6) Piece(l=6, t=-4, b=4, r=0); 
#     Piece(l=0, t=-1, b=1, r=7) Piece(l=-7, t=5, b=5, r=-5) Piece(l=5, t=-5, b=5, r=6) Piece(l=-6, t=-7, b=-6, r=-5) Piece(l=5, t=-4, b=-1, r=0); 
#     Piece(l=0, t=-1, b=1, r=5) Piece(l=-5, t=-5, b=5, r=-5) Piece(l=5, t=-5, b=6, r=-5) Piece(l=5, t=6, b=5, r=6) Piece(l=-6, t=1, b=4, r=0); 
#     Piece(l=0, t=-1, b=0, r=1) Piece(l=-1, t=-5, b=0, r=3) Piece(l=-3, t=-6, b=0, r=2) Piece(l=-2, t=-5, b=0, r=1) Piece(l=-1, t=-4, b=0, r=0)]



function generate_stats(w, h, nr_trials=100, test=false)
    sol1 = Matrix{Piece}(undef, h, w)
    sol2 = Matrix{Piece}(undef, h, w)
    sols = []

    all_pieces_unique = 0
    pieces_not_rot_symmetric = 0
    no_same_pairs_exist = 0
    all_nr_sols = []
    all_nr_connections = []
    all_nr_inner_connections = []
    exactly2sols = 0

    for t = 1:nr_trials
        random_puzzle!(sol1, sol2) # random puzzle with at least two solutions
        resolve_connections!(sol1, sol2) # use as many different connectors as possible

        # check if puzzle fulfills our constraints
        # ideally we would check if it has exactly 2 solutions, but that's not possible due to the combinatorial blow up
        # the problem is NP complete https://en.wikipedia.org/wiki/Edge-matching_puzzle
        # but we can check some obvious things to make sure there arent any additional solutions

        

        pieces_unique = pieces_are_unique(sol1)
        
        if test
            success, sols = get_all_solutions(sol1)
            nr_sols = length(sols)
            @assert !success || nr_sols >= 2
        end
        
        if (test && (success || nr_sols > 2)) || (!test && pieces_unique)
            if test
                @test pieces_unique || nr_sols > 2
                @test !pieces_symmetric || nr_sols > 2
                @test !pairs_exist1 || nr_sols > 2
                @test !pairs_exist2 || nr_sols > 2
            end

            success, sols = get_all_solutions(sol1)
            nr_sols = length(sols)

            pieces_symmetric = rot_symmetric_pieces_exist(sol1) 
            pairs_exist1 = same_pair_exists(sol1) 
            pairs_exist2 = same_pair_exists(sol2) 

            all_pieces_unique += Int(pieces_unique)
            pieces_not_rot_symmetric += Int(!pieces_symmetric)
            no_same_pairs_exist += Int(!(pairs_exist1 || pairs_exist2))
            exactly2sols += Int(nr_sols == 2)

            outer = length(stats)
            inner = nr_connections_inside(sol1)
            
            push!(all_nr_sols, nr_sols)
            push!(all_nr_connections, outer)
            push!(all_nr_inner_connections, inner)
        else
            outer = length(stats)
            inner = nr_connections_inside(sol1)
            println("failure:")
            println("total connections: $outer, inner_connections: $inner")
            display(sol1)
        end

    end
    
    println("$all_pieces_unique / $nr_trials: all pieces are unique")
    println("$pieces_not_rot_symmetric / $nr_trials: no pieces are rotationally symmetric")
    println("$no_same_pairs_exist / $nr_trials: all pairs of two pieces are distinct")
    println("$exactly2sols / $nr_trials: have exactly 2 solutions")

    return all_nr_sols, all_nr_connections, all_nr_inner_connections
end


@testset "stats" begin
    all_nr_sols, all_nr_connections, all_nr_inner_connections = generate_stats(5, 5, 100, true)
end

all_nr_sols, all_nr_connections, all_nr_inner_connections = generate_stats(6, 6, 100000)

using Plots, Statistics

nr_connections = Set(all_nr_connections) |> collect |> sort
nr_inner_conns = Set(all_nr_inner_connections) |> collect |> sort
avg_nr_sols_per_connections = [mean(all_nr_sols[all_nr_connections .== c]) for c in nr_connections]
avg_nr_sols_per_inner_conns = [mean(all_nr_sols[all_nr_inner_connections .== c]) for c in nr_inner_conns]
median_nr_sols_per_connections = [median(all_nr_sols[all_nr_connections .== c]) for c in nr_connections]
median_nr_sols_per_inner_conns = [median(all_nr_sols[all_nr_inner_connections .== c]) for c in nr_inner_conns]

bar(title="avg nr sols per nr connections", ylabel="nr solutions", xlabel="nr connections", nr_connections, avg_nr_sols_per_connections)
bar(title="avg nr sols per nr inner connections", ylabel="nr solutions", xlabel="nr connections", nr_inner_conns, avg_nr_sols_per_inner_conns)
bar(title="median nr sols per nr connections", ylabel="nr solutions", xlabel="nr connections", nr_connections, median_nr_sols_per_connections)
bar(title="median nr sols per nr inner connections", ylabel="nr solutions", xlabel="nr connections", nr_inner_conns, median_nr_sols_per_inner_conns)

for c in nr_connections
    histogram(all_nr_sols[all_nr_connections .== c], bins=20, xlims=(-5, 106), title="distribution of nr sols, nr connections = $c", xlabel="nr solutions") |> display
end

for c in nr_inner_conns
    histogram(all_nr_sols[all_nr_inner_connections .== c], bins=20, xlims=(-5, 106), title="distribution of nr sols, nr inner connections = $c", xlabel="nr solutions") |> display
end



ref = default_puzzle(16, 16)
test_puzzle = default_puzzle(16, 16)
random_puzzle!(ref, test_puzzle)
resolve_connections!(ref, test_puzzle)
while !pieces_are_unique(test_puzzle)
    random_puzzle!(ref, test_puzzle)
    resolve_connections!(ref, test_puzzle)
end

using BenchmarkTools

@btime pieces_are_unique(test_puzzle)
@btime pieces_are_unique2(test_puzzle)

# function pieces_are_unique(puzzle)
#     n = length(puzzle)
#     for i = 1:n, j=i+1:n
#         if puzzle[i] == puzzle[j]
#             return false
#         end
#     end
#     return true
# end


let pieces::Vector{NTuple{4, Int16}} = NTuple{4, Int}[]

    global function pieces_are_unique2(puzzle)
        empty!(pieces)
        for i in eachindex(puzzle)
            for r in allrotations(puzzle[i])
                push!(pieces, (Int16(r.top.id), Int16(r.right.id), Int16(r.bottom.id), Int16(r.left.id)))
            end
        end

        sort!(pieces, alg=QuickSort)

        for i = 1:length(pieces)-1
            if pieces[i] == pieces[i+1]
                return false
            end
        end

        return true

    end
end