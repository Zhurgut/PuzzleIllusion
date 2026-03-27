using LinearAlgebra, Statistics
using Combinatorics
using BenchmarkTools

struct Connection
    id::Int
end

Edge() = Connection(0)
Male(i) = Connection(abs(i))
Female(i) = Connection(-abs(i))

invert(c::Connection) = Connection(-c.id)
is_fit(c1, c2) = c1.id == -c2.id

fitting_connection_nr(c) = -c.id

struct Piece
    top::Connection
    right::Connection
    bottom::Connection
    left::Connection
end

function Piece(t::Integer, r::Integer, b::Integer, l::Integer)
    t2 = Connection(t)
    r2 = Connection(r)
    b2 = Connection(b)
    l2 = Connection(l)
    Piece(t2, r2, b2, l2)
end

function Base.show(io::IO, p::Piece)
    print(io, "Piece(l=$(p.left.id), t=$(p.top.id), b=$(p.bottom.id), r=$(p.right.id))")
end

function allrotations(p::Piece)
    p4 = Piece(p.right, p.bottom, p.left, p.top)
    p3 = Piece(p.bottom, p.left, p.top, p.right)
    p2 = Piece(p.left, p.top, p.right, p.bottom)
    return (p, p2, p3, p4)
end

rotate(p, amt) = allrotations(p)[mod(amt, 4) + 1]

function Base.:(==)(p1::Piece, p2::Piece)
    rots = allrotations(p2)
    for p in rots
        if (p.top, p.right, p.bottom, p.left) == (p1.top, p1.right, p1.bottom, p1.left)
            return true
        end
    end
    return false
end

function rand_perm!(array)
    n = length(array)
    array .= 1:n
    for i=1:n
        j = rand(1:n)
        array[i], array[j] = array[j], array[i]
    end
    return array
end

let perm::Vector{Int} = collect(1:10)

    global function edge_perm(w, h)
        nr_edges = 2w + 2h - 4
        corners = (1, w, w+h-1, 2w+h-2)
        resize!(perm, nr_edges)
        p = rand_perm!(perm)

        # set the corners in corner positions
        c = (corners[3], corners[2], corners[1], corners[4])
        for (i, ci) in enumerate(c)
            idx = findfirst(==(ci), p)
            p[corners[i]], p[idx] = p[idx], p[corners[i]]
        end

        prev(i, nr_edges) = mod(i-2, nr_edges) + 1
        next(i, nr_edges) = mod(i, nr_edges) + 1
        value_fits_at_position(v, i, p, N) = prev(v, N) != p[prev(i, N)] && next(v, N) != p[next(i, N)] 

        # make sure edges all have different connections
        for i = 1:nr_edges
            if i ∈ corners continue end

            if !value_fits_at_position(p[i], i, p, nr_edges)
                # value at p[i] needs to go somewhere else
                success = false

                for j=1:nr_edges
                    k = mod(i+j-1, nr_edges) + 1
                    if k ∈ corners continue end

                    if value_fits_at_position(p[i], k, p, nr_edges) && value_fits_at_position(p[k], i, p, nr_edges)
                        p[i], p[k] = p[k], p[i]
                        success = true
                        break
                    end
                end

                @assert success
            end
        end

        return p

    end

end




function default_puzzle!(m)
    h, w = size(m)
    row_off = 2w-1
    for c=1:w, r=1:h
        p = Piece(c + (r-1)*row_off, c+w+1+(r-1)*row_off, c + r*row_off, c+w+(r-1)*row_off)
        m[r, c] = Piece(invert(p.top), p.right, p.bottom, invert(p.left))
    end
    for c=1:w
        p = m[1, c]
        m[1, c] = Piece(Edge(), p.right, p.bottom, p.left)
        p = m[end, c]
        m[end, c] = Piece(p.top, p.right, Edge(), p.left)
    end
    for r=1:h
        p = m[r, 1]
        m[r, 1] = Piece(p.top, p.right, p.bottom, Edge())
        p = m[r, end]
        m[r, end] = Piece(p.top, Edge(), p.bottom, p.left)
    end
    m
end

default_puzzle(w, h) = default_puzzle!(Matrix{Piece}(undef, h, w))



function from_edge_to_cartesian(e, w, h)
    # return row_idx, col_idx, rotation
    if 1 <= e <= w-1
        return (1, e, 1)
    elseif w <= e <= w+h-2
        return (e-w+1, w, 2)
    elseif w+h-1 <= e <= 2w+h-3
        return (h, w - (e - (w+h-1)), 3)
    else 2w+h-2 <= e <= 2(w+h)-4
        return (h - (e - (2w+h-2)), 1, 4)
    end
end

function piece_fits(p, r, c, puzzle)
    above, toright, toleft, below = puzzle[r-1, c], puzzle[r, c+1], puzzle[r, c-1], puzzle[r+1, c]
    return is_fit(above.bottom, p.top)  && is_fit(toright.left, p.right) && 
           is_fit(toleft.right, p.left) && is_fit(below.top, p.bottom)
end

function piece_fits_nowhere(p, r, c, puzzle)
    above, toright, toleft, below = puzzle[r-1, c], puzzle[r, c+1], puzzle[r, c-1], puzzle[r+1, c]
    return !is_fit(above.bottom, p.top)  && !is_fit(toright.left, p.right) && 
           !is_fit(toleft.right, p.left) && !is_fit(below.top, p.bottom)
end


let filling_perm::Vector{Int} = zeros(Int, 10)

    global function random_puzzle!(sol1, out)
        sol1 = default_puzzle!(sol1)
        fill!(out, Piece(0,0,0,0))
        h,w = size(sol1)
        
        # edge pieces
        edges_perm = edge_perm(w, h)
        
        for (i, e) in enumerate(edges_perm)
            in_r, in_c, in_rot = from_edge_to_cartesian(i, w, h)
            out_r, out_c, out_rot = from_edge_to_cartesian(e, w, h)
            
            out[out_r, out_c] = rotate(sol1[in_r, in_c], out_rot - in_rot)
        end

        # filling pieces
        resize!(filling_perm, (w-2)*(h-2))
        rand_perm!(filling_perm)
        for (i, e) in enumerate(filling_perm)
            in_r, in_c = (i-1) ÷ (h-2) + 1, (i-1) % (w-2) + 1
            piece = sol1[in_r+1, in_c+1]
            out_r, out_c = (e-1) ÷ (h-2) + 1, (e-1) % (w-2) + 1
            
            success = false
            init_r = rand(1:4)
            for r=0:3
                rotated = rotate(piece, init_r + r)
                if piece_fits_nowhere(rotated, out_r+1, out_c+1, out)
                    out[out_r+1, out_c+1] = rotated
                    success = true
                    break
                end
            end
            if !success
                println("fail")
                return random_puzzle(w,h)
            end
        end

        sol1, out
    end
end

stats = []

let map::Vector{Int} = zeros(Int, 1024)

    global function get_mapping!(graph)
        n = (size(graph, 1) - 1) ÷ 2
        resize!(map, 2n+1)
        fill!(map, 0)

        for connection_id=1:n
            if !any(graph)
                global stats = []
                for id=1:connection_id-1
                    push!(stats, 0.5*sum((abs(m) == id for m in map)))
                end
                break
            end

            start = findfirst(graph) |> Tuple
            map[start[1]] = connection_id
            map[start[2]] = -connection_id

            graph[start[1], start[2]] = graph[start[2], start[1]] = false
            next_row = findfirst(@view graph[:, start[2]])
            if isnothing(next_row)
                continue
            end

            crt = next_row, start[2]
            while next_row != start[1]
                graph[crt[1], crt[2]] = graph[crt[2], crt[1]] = false
                next_col = findfirst(@view graph[crt[1], :])

                crt = crt[1], next_col
                map[crt[1]] = connection_id
                map[crt[2]] = -connection_id

                graph[crt[1], crt[2]] = graph[crt[2], crt[1]] = false
                next_row = findfirst(@view graph[:, crt[2]])
                crt = next_row, crt[2]
            end

            graph[crt[1], crt[2]] = graph[crt[2], crt[1]] = false
        end
        
        map
    end
end


let graph::Matrix{Bool} = zeros(Bool, 1024, 1024)

    global function resolve_connections!(def_sol, sol2)
        n = def_sol[end, end].left.id |> Int |> abs 
        if size(graph) != (2n+1, 2n+1)
            graph = zeros(Bool, 2n+1, 2n+1)
        else
            fill!(graph, false)
        end
        h, w = size(def_sol)

        for c=1:w, r=1:h
            d = def_sol[r, c]
            s = sol2[r, c]
            if r < h
                d2 = def_sol[r+1, c]
                i1, i2 = d.bottom.id, d2.top.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true

                s2 = sol2[r+1, c]
                i1, i2 = s.bottom.id, s2.top.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true
            end
            if r > 1
                d2 = def_sol[r-1, c]
                i1, i2 = d.top.id, d2.bottom.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true

                s2 = sol2[r-1, c]
                i1, i2 = s.top.id, s2.bottom.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true
            end
            if c < w
                d2 = def_sol[r, c+1]
                i1, i2 = d.right.id, d2.left.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true

                s2 = sol2[r, c+1]
                i1, i2 = s.right.id, s2.left.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true
            end
            if c > 1
                d2 = def_sol[r, c-1]
                i1, i2 = d.left.id, d2.right.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true

                s2 = sol2[r, c-1]
                i1, i2 = s.left.id, s2.right.id
                graph[i1+n+1, i2+n+1] = graph[i2+n+1, i1+n+1] = true
            end
        end

        mapping = get_mapping!(graph)
        
        for c=1:w, r=1:h
            p = def_sol[r, c]
            def_sol[r, c] = Piece(
                Connection(mapping[p.top.id + n + 1]),
                Connection(mapping[p.right.id + n + 1]),
                Connection(mapping[p.bottom.id + n + 1]),
                Connection(mapping[p.left.id + n + 1]),
            )

            p = sol2[r, c]
            sol2[r, c] = Piece(
                Connection(mapping[p.top.id + n + 1]),
                Connection(mapping[p.right.id + n + 1]),
                Connection(mapping[p.bottom.id + n + 1]),
                Connection(mapping[p.left.id + n + 1]),
            )
        end

        def_sol, sol2
    end

end

function pieces_are_unique(puzzle)
    n = length(puzzle)
    for i = 1:n, j=i+1:n
        if puzzle[i] == puzzle[j]
            return false
        end
    end
    return true
end


function rot_symmetric_pieces_exist(puzzle)
    for p in puzzle
        r = rotate(p, 2)
        if (p.top, p.right, p.bottom, p.left) == (r.top, r.right, r.bottom, r.left)
            return true
        end
    end
    return false
end

let pairs::Vector{Tuple{Piece, Piece}} = []

    global function same_pair_exists(puzzle)
        h, w = size(puzzle)
        empty!(pairs)
        for c=1:w-1, r=1:h
            push!(pairs, (puzzle[r, c], puzzle[r, c+1]))
        end
        for c=1:w, r=1:h-1
            push!(pairs, (rotate(puzzle[r, c], -1), rotate(puzzle[r+1, c], -1)))
        end

        # check that each pair cannot be reassembled differently into itself
        for p in pairs
            for r1=1:4, r2=1:4
                u = (rotate(p[2], r1), rotate(p[1], r2))
                if is_fit(u[1].right, u[2].left)
                    if (p[1].left, p[1].top, p[2].top, p[2].right, p[2].bottom, p[1].bottom) ==
                        (u[1].left, u[1].top, u[2].top, u[2].right, u[2].bottom, u[1].bottom)
                        return true
                    end
                end
            end
        end

        n = length(pairs)
        for i in 1:n
            p1, p2 = pairs[i]
            push!(pairs, (rotate(p2, 2), rotate(p1, 2)))
        end
        n = length(pairs)
        for i=1:n, j=i+1:n
            p = pairs[i]
            u = pairs[j]
            if (p[1].left, p[1].top, p[2].top, p[2].right, p[2].bottom, p[1].bottom) ==
                (u[1].left, u[1].top, u[2].top, u[2].right, u[2].bottom, u[1].bottom)
                return true
            end
        end
        return false
    end

end

function all_solutions!(solution, pieces, next_r, next_c, solutions; start_time=time(), max_time=5, max_nr_solutions=100)
    if time() - start_time > max_time || length(solutions) >= max_nr_solutions
        return false
    end

    timeout = false

    h, w = size(solution)

    required_top = nothing
    required_bottom = nothing
    required_right = nothing
    required_left = nothing
    
    required_left = next_c == 1 ? Edge() : invert(solution[next_r, next_c-1].right)
    required_top = next_r == 1 ? Edge() : invert(solution[next_r-1, next_c].bottom)
    if next_c == w required_right = Edge() end
    if next_r == h required_bottom = Edge() end

    for i in eachindex(pieces)
        p = pieces[i]
        for r = allrotations(p)
            if r.left == required_left && 
                r.top == required_top && 
                (isnothing(required_right) || r.right == required_right) && 
                (isnothing(required_bottom) || r.bottom == required_bottom)

                solution[next_r, next_c] = r
                
                if (next_r, next_c) == (h, w)
                    push!(solutions, copy(solution))
                    continue
                end

                begin
                    popat!(pieces, i)

                    nc = next_c % w + 1
                    nr = nc == 1 ? next_r + 1 : next_r
                    success = all_solutions!(solution, pieces, nr, nc, solutions, start_time=start_time, max_time=max_time)
                    if !success
                        timeout = true
                    end

                    insert!(pieces, i, p)
                end

            end
        end
    end

    success = !timeout
    return success

end

function get_all_solutions(puzzle)
    sols = []
    success = all_solutions!(copy(puzzle), vec(puzzle[2:end]), 1, 2, sols)
    return success, sols
end

function nr_connections_inside(puzzle)
    s = Set(Int[])
    h, w = size(puzzle)
    for c=2:w-1, r=2:h-1
        push!(s, abs(puzzle[r, c].top.id))
        push!(s, abs(puzzle[r, c].right.id))
        push!(s, abs(puzzle[r, c].bottom.id))
        push!(s, abs(puzzle[r, c].left.id))
    end
    return length(s)
end


function generate_puzzle(w, h, nr_trials=10000)
    sol1 = Matrix{Piece}(undef, h, w)
    sol2 = Matrix{Piece}(undef, h, w)
    best_s1, best_s2, best_inner, best_total = (sol1, sol2, 0, 0)
    sols = []

    for t = 1:nr_trials
        random_puzzle!(sol1, sol2) # random puzzle with at least two solutions
        resolve_connections!(sol1, sol2) # use as many different connectors as possible

        # check if puzzle fulfills our constraints
        # ideally we would check if it has exactly 2 solutions, but that's not possible due to the combinatorial blow up
        # the problem is NP complete https://en.wikipedia.org/wiki/Edge-matching_puzzle
        # but we can check some obvious things to make sure there arent any additional solutions

        if !pieces_are_unique(sol1) 
            # print(".")
            continue 
        end # no duplicate pieces
        if rot_symmetric_pieces_exist(sol1) 
            # print("+")
            continue 
        end # every piece must not be rotationally symmetric
        if same_pair_exists(sol1) 
            print("(*)")
            continue 
        end # all pairs of pieces must not be rotationally symmetric
        if same_pair_exists(sol2) 
            print("(*)")
            continue 
        end # and no two pairs of pieces must be the same

        outer = length(stats)
        inner = nr_connections_inside(sol1)
        if inner > best_inner || inner == best_inner && outer > best_total
            best_s1 = copy(sol1)
            best_s2 = copy(sol2)
            best_inner = inner
            best_total = outer
        end

        # sols = get_all_solutions(sol1)

        # println()
        # println("trial $t success, $(length(sols)) solutions")

        println("($(length(sols)), $(nr_connections_inside(sol1)), $(length(stats)), $(sort(stats))),")

        # if length(sols) > 100
        #     return sol1, sol2
        # end
        # return sol1, sol2
    end
    
    println("total connections: $best_total")
    println("inner connections: $best_inner")
    # println("nr_solutions: $(length(get_all_solutions(best_s1)))")
    best_s1, best_s2

end


