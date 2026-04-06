using Pictura, PicturaShapes

function load_perm(path)
    data = Tables.matrix(CSV.File(path, header=false))
end

function set_average!(out, weight, img1, img2)
    for i in eachindex(pixels(out))
        c1 = Pictura.floats(pixels(img1)[i])
        c2 = Pictura.floats(pixels(img2)[i])
        avg_color = color(
            weight[i] * c1.r + (1-weight[i]) * c2.r,
            weight[i] * c1.g + (1-weight[i]) * c2.g,
            weight[i] * c1.b + (1-weight[i]) * c2.b,
        )
        pixels(out)[i] = avg_color
    end
    updatepixels(out)
end

function invert_permutation(permutex, permutey)
    H, W = size(permutex)
    def_y = reshape(repeat(1:H, W), H, W)
    def_x = transpose(reshape(repeat(1:W, H), W, H))

    indices = CartesianIndex.(permutey, permutex)

    out_x = similar(def_x)
    out_y = similar(def_y)

    @view(out_x[indices]) .= def_x
    @view(out_y[indices]) .= def_y

    return out_x, out_y
end

function resolve_weights!(weight_matrix, w1_img, w2_img, permute_idx, inv_permute_idx)
    loadpixels(w1_img)
    loadpixels(w2_img)
    alpha = 0.2
    for i in eachindex(pixels(w1_img))
        p1 = pixels(w1_img)[i]
        
        if blue(p1) > 0 # blue is used to mark areas for erasing
            pixels(w1_img)[i] = color(0,0,0,0)
            pixels(w2_img)[inv_permute_idx[i]] = color(0,0,0,0)
        elseif green(p1) > 0
            pixels(w1_img)[i] = color(red(p1), 255, blue(p1), alpha)
            t = pixels(w2_img)[inv_permute_idx[i]]
            pixels(w2_img)[inv_permute_idx[i]] = color(255, green(t), blue(t), alpha)
        end

        p2 = pixels(w2_img)[i]

        if blue(p2) > 0
            pixels(w2_img)[i] = color(0,0,0,0)
            pixels(w1_img)[permute_idx[i]] = color(0,0,0,0)
        elseif green(p2) > 0
            pixels(w2_img)[i] = color(red(p2), 255, blue(p2), alpha)
            t = pixels(w1_img)[permute_idx[i]]
            pixels(w1_img)[permute_idx[i]] = color(255, green(t), blue(t), alpha)
        end
    end

    for i in eachindex(weight_matrix)
        p1 = pixels(w1_img)[i]
        p2 = pixels(w2_img)[inv_permute_idx[i]]

        w1 = green(p1) * (1/255) + 0.01
        w2 = green(p2) * (1/255) + 0.01

        weight_matrix[i] = w1 / (w1 + w2)
    end

    updatepixels(w1_img)
    updatepixels(w2_img)
end

function draw_weight(img1_path, img2_path, permutex, permutey)
    @pictura begin
        setup(800, 800)
        framerate(30)
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        W, H = width(img1), height(img1)

        avg_img2 = Pictura.Image(W, H)
        avg_img1 = Pictura.Image(W, H)

        weight1 = Pictura.Image(W, H)
        fill!(pixels(weight1), color(0,0,0,0))
        updatepixels(weight1)
        weight2 = Pictura.Image(W, H)
        fill!(pixels(weight2), color(0,0,0,0))
        updatepixels(weight2)

        inv_permutex, inv_permutey = invert_permutation(permutex, permutey)
        permute_indices = CartesianIndex.(permutey, permutex)
        inv_permute_indices = CartesianIndex.(inv_permutey, inv_permutex)

        weight = fill(0.5f0, H, W)
        img1_from2 = Pictura.Image(W, H)
        pixels(img1_from2) .= @view pixels(img2)[inv_permute_indices]
        updatepixels(img1_from2)

        @mousedragged begin
            x, y = (mouse().x % (width() ÷ 2)) / 0.5width(), (mouse().y % (height() ÷ 2)) / 0.5height()
            nostroke()
            if mouse().r
                fillcolor(0, 0, 255)
            else
                fillcolor(0, 255, 0)
            end
            if mouse().y < 0.5height()
                circle(weight1, x*W, y*H, 10)
            else
                circle(weight2, x*W, y*H, 10)
            end
        end

        @drawloop begin
            image(img1, dst_rect=Rect(0, 0, 0.5width(), 0.5height(), 0))
            image(img2, dst_rect=Rect(0, 0.5height(), 0.5width(), 0.5height(), 0))

            image(weight1, dst_rect=Rect(0, 0, 0.5width(), 0.5height(), 0))
            image(weight2, dst_rect=Rect(0, 0.5height(), 0.5width(), 0.5height(), 0))

            resolve_weights!(weight, weight1, weight2, permute_indices, inv_permute_indices)

            set_average!(avg_img1, weight, img1, img1_from2)
            pixels(avg_img2) .= @view pixels(avg_img1)[permute_indices]
            updatepixels(avg_img2)

            image(avg_img1, dst_rect=Rect(0.5width(), 0, 0.5width(), 0.5height(), 0))
            image(avg_img2, dst_rect=Rect(0.5width(), 0.5height(), 0.5width(), 0.5height(), 0))

            strokecolor(0.5)
            line(0, 0.5height(), width(), 0.5height())
            line(0.5width(), 0, 0.5width(), height())
        end

    end
end