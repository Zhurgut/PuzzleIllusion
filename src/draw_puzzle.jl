using Dierckx, Plots, PicturaShapes, TikzPictures

point(angle) = cos(angle), sin(angle)

# define some points which a parametric spline will go through to create a knobly bit
function random_connector(;
    max_vertical_offset = 0.03,
    min_vertical_offset = 0.005,
    outer_slide = 0.05,
    radius = 0.025,
)
    
    vertical_off1 = min_vertical_offset + (max_vertical_offset-min_vertical_offset) * rand()
    vertical_off2 = min_vertical_offset + (max_vertical_offset-min_vertical_offset) * rand()

    # map x ∈ [a, b] to [A, B]
    map(x, a, b, A, B) = (x - a) * (B - A) / (b - a) + A

    # points between the piece corner and the knob
    outer_points = [
        (0.25 + map(rand(), 0, 1, -outer_slide, +outer_slide),  -vertical_off1),
        (0.75 + map(rand(), 0, 1, -outer_slide, +outer_slide),  -vertical_off2)
    ]

    # The spline will go through a random point on the circle with center 
    # at the points and radius
    knob_points = [
        (0.407, 0.075 - vertical_off1),
        (0.375, 0.2   - vertical_off1),
        (0.5,   0.275 - vertical_off1),
        (0.625, 0.2   - vertical_off2),
        (0.593, 0.075 - vertical_off2),
    ]

    points = zeros(2, 11)
    points[:, 1] .= (0,0)
    points[:, 2] .= (0.01, 0)
    points[:, 3] .= outer_points[1]
    for (i, c) in enumerate(knob_points)
        if i ∈ [4, 8]
            # a bit of tighter radius around the neck of the knob
            points[:, i+3] .= c .+ 0.7radius .* point(2π * rand())
        else
            points[:, i+3] .= c .+ radius .* point(2π * rand())
        end
    end
    points[:, end-2] .= outer_points[2]
    points[:, end-1] .= (0.99, 0)
    points[:, end] .= (1,0)

    spl = ParametricSpline(points)

    # X = 0:0.01:1
    # p = spl.(X)
    # plot(map(x->x[1], p), map(x->x[2], p), ratio=1, ylims=(-0.2, 0.4), xlims=(-0.1, 1.1)) |> display

    return spl
end

# tells if (px, py) is below the curve defined by the knob spline
# spline_points = spline.(ts), ts[i] ∈ [0,1]
function is_below(px, py, ts, spline_points, spline)
    i = argmin(i->((px - spline_points[i][1])^2 + (py - spline_points[i][2])^2), 1:length(spline_points))
    sx, sy = spline_points[i]
    t = ts[i]
    tangent = derivative(spline, t)
    to_point = [px - sx, py - sy]
    cross = tangent[1] * to_point[2] - tangent[2] * to_point[1]

    return cross <= 0

end

function draw_path!(io, points)
    for i=1:length(points)-1
        p = points[i]
        t = points[i+1]
        println(io, "\\draw ($(p.x),$(p.y)) -- ($(t.x), $(t.y));")
    end
end

# will transform the std_points that define the knob curve to the correct location and draw them
# side = :right, or :bottom
function draw_connector!(io, r, c, side, std_points, male=false)
    points = if male
        [PicturaShapes.rotate(p, π) + Point(1, 0) for p in std_points]
    else
        std_points
    end

    points = if side == :right
        [PicturaShapes.rotate(p, 0.5π) + Point(1, 0) for p in points]
    elseif side == :bottom
        [PicturaShapes.rotate(p, π) + Point(1, 1) for p in points]
    else
        error("invalid value for 'side'")
    end

    points = [p + Point(c-1, r-1) for p in points]
    
    draw_path!(io, points)
end

function draw_puzzle(puzzle, draw_points)
    h, w = size(puzzle)
    io = IOBuffer()
    println(io, "\\draw (0,0) -- (0, $h) -- ($w, $h) -- ($w, 0) -- (0,0);")
    println(io, "\\draw (-1, -1) -- (-1, $h+1) -- ($w+1, $h+1) -- ($w+1, -1) -- (-1,-1);")

    for c=1:w, r=1:h
        piece = puzzle[r, c]
        if piece.right != Edge()
            draw_connector!(io, r, c, :right, draw_points[abs(piece.right.id)], piece.right.id > 0)
        end
        if piece.bottom != Edge()
            draw_connector!(io, r, c, :bottom, draw_points[abs(piece.bottom.id)], piece.bottom.id > 0)
        end
    end

    return TikzPicture(String(take!(io)), options="yscale=-1")
end

# out_type can also be SVG, TEX or TIKZ
function draw_puzzles(puzzle, puzzle2, out_name::String, connectors, out_type=PDF)
    # mirror connectors here, because later we are going to do yscale=-1, which is going to mirror them back
    draw_points = [
        [Point(1 - p[1], p[2]) for p in connector.(0:0.01:1)] for connector in connectors
    ]

    picture = draw_puzzle(puzzle, draw_points)
    save(out_type(out_name * "1"), picture)

    picture = draw_puzzle(puzzle2, draw_points)
    save(out_type(out_name * "2"), picture)
end



# returns a matrix filled with indeces, each index corresponds to one puzzle piece, e.g.
# [1 1 1 3 3 3]
# [1 1 3 3 3 3]
# [1 1 1 3 4 3]
# [2 1 2 4 4 4]
# [2 2 2 2 4 4]
# [2 2 2 4 4 4]
function get_piece_map(puzzle, S, splines)
    h, w = size(puzzle)
    piece_index = reshape(1:h*w, h, w)
    map = repeat(piece_index, inner=(S,S))
    ts = 0:0.01:1
    spline_points = [spline.(ts) for spline in splines]

    for c=1:w, r=1:h
        piece = puzzle[r, c]
        if piece.right != Edge()
            pi = piece_index[r, c]
            ri = piece_index[r, c+1]

            connector = splines[abs(piece.right.id)]
            spl_points = spline_points[abs(piece.right.id)]

            R_off, C_off = S*(r-1), S*(c-1)
            out = @view(map[R_off+1:R_off+S, C_off+1:C_off+2S])
        

            for mr=1:S, mc=1:2S
                
                if piece.right.id > 0

                    p = (1/S) * Point(mr - 0.5, mc - 0.5) + Point(0, -1)

                    @assert -1 <= p.y <= 1
                    @assert 0 <= p.x <= 1
                    is_left = is_below(p.x, p.y, ts, spl_points, connector) 
                    
                    if is_left && out[mr, mc] == ri
                        out[mr, mc] = pi
                    elseif !is_left && out[mr, mc] == pi
                        out[mr, mc] = ri
                    end
                else

                    p = (1/S) * Point(S - mr + 0.5, 2S - mc + 0.5) + Point(0, -1)
                    
                    @assert -1 <= p.y <= 1
                    @assert 0 <= p.x <= 1
                    is_right = is_below(p.x, p.y, ts, spl_points, connector) 

                    if is_right && out[mr, mc] == pi
                        out[mr, mc] = ri
                    elseif !is_right && out[mr, mc] == ri
                        out[mr, mc] = pi
                    end
                end
            end 
        end
        if piece.bottom != Edge()
            pi = piece_index[r, c]
            bi = piece_index[r+1, c]

            connector = splines[abs(piece.bottom.id)]
            spl_points = spline_points[abs(piece.bottom.id)]

            R_off, C_off = S*(r-1), S*(c-1)
            out = @view(map[R_off+1:R_off+2S, C_off+1:C_off+S])

            for mr=1:2S, mc=1:S

                if piece.bottom.id > 0

                    p = (1/S) * Point(S - mc + 0.5, mr - 0.5) + Point(0, -1)

                    is_top = is_below(p.x, p.y, ts, spl_points, connector) 

                    if is_top && out[mr, mc] == bi
                        out[mr, mc] = pi
                    elseif !is_top && out[mr, mc] == pi
                        out[mr, mc] = bi
                    end
                    
                else

                    p = (1/S) * Point(mc - 0.5, 2S - mr + 0.5) + Point(0, -1)

                    is_bottom = is_below(p.x, p.y, ts, spl_points, connector) 

                    if is_bottom && out[mr, mc] == pi
                        out[mr, mc] = bi
                    elseif !is_bottom && out[mr, mc] == bi
                        out[mr, mc] = pi
                    end
                    
                end

            end
        end
    end

    return map
end



function save_permutation_with_round_knobs(puzzle, sol, connectors, out_folder, F=1.0)
    h, w = size(puzzle)
    S = Int(64*F)
    H, W = S*h, S*w
    def_y = reshape(repeat(1:H, W), H, W)
    def_x = transpose(reshape(repeat(1:W, H), W, H))

    out_x = similar(def_x)
    out_y = similar(def_y)

    map = get_piece_map(puzzle, S, connectors)

    BORDER = S ÷ 2
    piece_index_array = fill(false, 2BORDER + S, 2BORDER + S)
    piece_array_x = fill(0, 2BORDER + S, 2BORDER + S)
    piece_array_y = fill(0, 2BORDER + S, 2BORDER + S)

    for c=1:w, r=1:h
        piece = puzzle[r, c]
        idx = (c-1)*h + r
        
        # copy piece indeces from map, and origin x and y coordinates into array
        piece_index_array .= false
        for i=1:2BORDER + S, j=1:2BORDER + S
            i2, j2 = (r-1)*S+i-BORDER, (c-1)*S+j-BORDER
            if 1 <= i2 <= H && 1 <= j2 <= W
                piece_index_array[i, j] = map[i2, j2] == idx
                piece_array_x[i, j] = def_x[i2, j2]
                piece_array_y[i, j] = def_y[i2, j2]
            end
        end

        for c2=1:w, r2=1:h
            piece2 = sol[r2, c2]

            if piece == piece2

                for r=0:3
                    if to_tuple(rotate(piece, r)) == to_tuple(piece2)

                        # copy coordinates in array to out with correct rotation
                        piece_index_array = rotr90(piece_index_array, r)
                        piece_array_x = rotr90(piece_array_x, r)
                        piece_array_y = rotr90(piece_array_y, r)

                        for i=1:2BORDER + S, j=1:2BORDER + S
                            i2, j2 = (r2-1)*S+i-BORDER, (c2-1)*S+j-BORDER
                            if piece_index_array[i,j]
                                out_x[i2, j2] = piece_array_x[i, j]
                                out_y[i2, j2] = piece_array_y[i, j]
                            end
                        end
                        break
                    end
                end

                break
            end
        end
    end

    CSV.write(joinpath(out_folder, "perm_x.csv"), Tables.table(out_x), writeheader=false)
    CSV.write(joinpath(out_folder, "perm_y.csv"), Tables.table(out_y), writeheader=false)

    out_x, out_y
end