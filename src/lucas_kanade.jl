struct LKPyramid
    layers::Vector{Matrix{Gray{Float64}}}

    Iy::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
    Ix::Union{Nothing, Vector{Matrix{Gray{Float64}}}}

    Iyy::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
    Ixx::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
    Iyx::Union{Nothing, Vector{Matrix{Gray{Float64}}}}
end

function LKPyramid(image, levels; downsample = 2, σ = 1.0, compute_gradients::Bool = true)
    pyramid = gaussian_pyramid(image, levels, downsample, σ)
    !compute_gradients &&
        return LKPyramid(pyramid, nothing, nothing, nothing, nothing, nothing)

    total_levels = levels + 1
    Iy = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Ix = Vector{Matrix{Gray{Float64}}}(undef, total_levels)

    Iyy = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Ixx = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Iyx = Vector{Matrix{Gray{Float64}}}(undef, total_levels)

    filling = Fill(zero(eltype(pyramid[1])))
    for (i, layer) in enumerate(pyramid)
        Iy[i], Ix[i] = imgradients(layer, KernelFactors.scharr, filling)
        Iyy[i], Ixx[i], Iyx[i] = compute_partial_derivatives(Iy[i], Ix[i])
    end
    LKPyramid(pyramid, Iy, Ix, Iyy, Ixx, Iyx)
end

function optflow(
    first_img::AbstractMatrix{T}, second_img::AbstractMatrix{T},
    points::Vector{SVector{2, Float64}}, algorithm::LucasKanade,
) where T <: Gray
    displacement = fill(SVector{2, Float64}(0.0, 0.0), length(points))
    first_pyramid = LKPyramid(first_img, algorithm.pyramid_levels)
    second_pyramid = LKPyramid(
        second_img, algorithm.pyramid_levels; compute_gradients=false,
    )
    optflow!(first_pyramid, second_pyramid, points, displacement, algorithm)
end

function optflow!(
    first_img::AbstractMatrix{T}, second_img::AbstractMatrix{T},
    points::Vector{SVector{2, Float64}}, displacement::Vector{SVector{2, Float64}},
    algorithm::LucasKanade,
) where T <: Gray
    first_pyramid = LKPyramid(first_img, algorithm.pyramid_levels)
    second_pyramid = LKPyramid(
        second_img, algorithm.pyramid_levels; compute_gradients=false,
    )
    optflow!(first_pyramid, second_pyramid, points, displacement, algorithm)
end

function optflow!(
    first_pyramid::LKPyramid, second_pyramid::LKPyramid,
    points::Vector{SVector{2, Float64}}, displacement::Vector{SVector{2, Float64}},
    algorithm::LucasKanade,
)
    enough_layers = (
        length(first_pyramid.layers) > algorithm.pyramid_levels &&
        length(second_pyramid.layers) > algorithm.pyramid_levels
    )
    !enough_layers && throw("Not enough layers in pyramids.")

    n_points = points |> length
    status = trues(n_points)

    window = algorithm.window_size
    window2x = 2 * window

    for level in (algorithm.pyramid_levels + 1):-1:1
        level_resolution = first_pyramid.layers[level] |> axes
        # Interpolate layer to get sub-pixel precision.
        # We never go out-of-bound, so there is no need to extrapolate.
        interploated_layer = interpolate(second_pyramid.layers[level], BSpline(Linear()))

        for did in 1:n_points
            @inbounds !status[did] && continue

            point = get_pyramid_coordinate(@inbounds(points[did]), level)

            grid, offsets = get_grid(point, point, window, level_resolution)
            G_inv, min_eigenvalue = compute_spatial_gradient(first_pyramid, grid, level)
            if min_eigenvalue < algorithm.eigenvalue_threshold
                @inbounds status[did] = false
                continue
            end

            pyramid_contribution = SVector{2}(0.0, 0.0)
            for k in 1:algorithm.iterations
                putative_flow = @inbounds displacement[did] + pyramid_contribution
                putative_correspondence = point + putative_flow
                if !lies_in(level_resolution, putative_correspondence)
                    @inbounds status[did] = false
                    break
                end

                new_grid, new_offsets = get_grid(
                    point, putative_correspondence, window, level_resolution,
                )
                # Recalculate gradient only if the grid changes.
                if new_grid != grid
                    grid, offsets = new_grid, new_offsets
                    G_inv, min_eigenvalue = compute_spatial_gradient(
                        first_pyramid, grid, level,
                    )
                    if min_eigenvalue < algorithm.eigenvalue_threshold
                        @inbounds status[did] = false
                        break
                    end
                else
                    grid, offsets = new_grid, new_offsets
                end

                estimated_flow = compute_flow_vector(
                    putative_correspondence,
                    first_pyramid, interploated_layer, level,
                    grid, offsets, G_inv,
                )
                pyramid_contribution += estimated_flow
                # Check if tracked point is out of image bounds.
                if !lies_in(level_resolution, point + pyramid_contribution)
                    @inbounds status[did] = false
                    break
                end
                # Epsilon termination criteria.
                abs(estimated_flow[1]) < algorithm.ϵ &&
                    abs(estimated_flow[2]) < algorithm.ϵ && break
            end
            @inbounds begin
            # Check if flow is too big.
            if status[did] && is_lost(displacement[did], window2x)
                status[did] = false
            end
            if status[did]
                displacement[did] = displacement[did] + pyramid_contribution
                level != 1 && (displacement[did] *= 2.0)
            end
            end
        end
    end

    displacement, status
end

function compute_partial_derivatives(Iy, Ix)
    Iyy = imfilter(Iy .* Iy, Kernel.gaussian(4))
    Ixx = imfilter(Ix .* Ix, Kernel.gaussian(4))
    Iyx = imfilter(Iy .* Ix, Kernel.gaussian(4))

    Iyy_integral_table = integral_image(Iyy)
    Ixx_integral_table = integral_image(Ixx)
    Iyx_integral_table = integral_image(Iyx)

    Iyy_integral_table, Ixx_integral_table, Iyx_integral_table
end

function _compute_spatial_gradient(
    grid, Iyy_integral, Iyx_integral, Ixx_integral,
)
    sum_Iyy = boxdiff(Iyy_integral, grid[1], grid[2])
    sum_Ixx = boxdiff(Ixx_integral, grid[1], grid[2])
    sum_Iyx = boxdiff(Iyx_integral, grid[1], grid[2])
    SMatrix{2, 2, Float64}(sum_Iyy, sum_Iyx, sum_Iyx, sum_Ixx)
end

function compute_spatial_gradient(pyramid::LKPyramid, grid, level)
    G = _compute_spatial_gradient(
        grid, pyramid.Iyy[level], pyramid.Iyx[level], pyramid.Ixx[level],
    )
    U, S, V = G |> svd2x2
    G_inv = pinv2x2(U, S, V)
    min_eigenvalue = min(S[1, 1], S[2, 2]) / prod(length.(grid))

    G_inv, min_eigenvalue
end

function prepare_linear_system(corresponding_point, A, Iy, Ix, offsets, B)
    P, Q = size(A)
    by, bx = 0.0, 0.0

    @inbounds for q in 1:Q, p in 1:P
        r = corresponding_point[1] + offsets[1][p]
        c = corresponding_point[2] + offsets[2][q]

        δI = A[p, q] - B(r, c)
        by += δI * Iy[p, q]
        bx += δI * Ix[p, q]
    end

    SVector{2, Float64}(by, bx)
end

function compute_flow_vector(
    corresponding_point,
    first_pyramid::LKPyramid, etp, level,
    grid, offsets, G_inv,
)
    b = prepare_linear_system(
        corresponding_point,
        view(first_pyramid.layers[level], grid[1], grid[2]),
        view(first_pyramid.Iy[level], grid[1], grid[2]),
        view(first_pyramid.Ix[level], grid[1], grid[2]),
        offsets, etp,
    )
    G_inv * b
end

@inline is_lost(displacement, window) =
    @inbounds(displacement[1] > window || displacement[2] > window)

@inline lies_in(area, point) = @inbounds(
    first(area[1]) ≤ point[1] ≤ last(area[1]) &&
    first(area[2]) ≤ point[2] ≤ last(area[2])
)

"""
# Arguments
- `level`: Level of the pyramid in `[1, levels]` range.
"""
@inline get_pyramid_coordinate(point, level) = floor.(Int64, point ./ 2 ^ (level - 1))

function get_grid(point, new_point, window, image_axes)
    rows, cols = image_axes

    up = floor(Int64, min(window, min(point[1], new_point[1]) - first(rows)))
    down = floor(Int64, min(window, last(rows) - max(point[1], new_point[1])))
    left = floor(Int64, min(window, min(point[2], new_point[2]) - first(cols)))
    right = floor(Int64, min(window, last(cols) - max(point[2], new_point[2])))

    new_grid = (
        (point[1] - up):(point[1] + down),
        (point[2] - left):(point[2] + right),
    )
    offsets = (-up:down, -left:right)
    new_grid, offsets
end
