# TODO allow passing precomputed pyramids.
# TODO add epsilon termination criteria.

function optflow(
    first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2},
    points::Array{SVector{2, Float64}, 1}, algorithm::LucasKanade,
) where T <: Gray
    displacement = fill(SVector{2, Float64}(0.0, 0.0), length(points))
    optflow!(first_img, second_img, points, displacement, algorithm)
end

function optflow!(
    first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2},
    points::Array{SVector{2, Float64}, 1},
    displacement::Array{SVector{2, Float64}, 1}, algorithm::LucasKanade,
) where T <: Gray
    window = algorithm.window_size
    first_img = map(x -> isnan(x) ? zero(x) : x, first_img)
    second_img = map(x -> isnan(x) ? zero(x) : x, second_img)

    first_pyramid, second_pyramid = construct_pyramids(
        first_img, second_img, algorithm.pyramid_levels,
    )
    flow = fill(SVector{2, Float64}(0.0, 0.0), length(displacement))
    status = trues(length(displacement))

    for i = (algorithm.pyramid_levels + 1):-1:1
        itp = interpolate(second_pyramid[i], BSpline(Linear()))
        etp = extrapolate(itp, zero(eltype(second_pyramid[i])))
        Iy, Ix = imgradients(
            first_pyramid[i], KernelFactors.scharr,
            Fill(zero(eltype(first_pyramid[i]))),
        )

        Iyy_it, Ixx_it, Iyx_it = compute_partial_derivatives(Iy, Ix)

        inner_bounds = map(
            i -> first(i) + window:last(i) - window, axes(first_pyramid[i]),
        )

        for j = 1:length(displacement)
            !status[j] && continue

            point = get_pyramid_coordinates(points, i, j)

            if lies_in(inner_bounds, point)
                grid = SVector{2}(
                    (point[1] - window):(point[1] + window),
                    (point[2] - window):(point[2] + window),
                )
                G = compute_spatial_gradient(grid, Iyy_it, Iyx_it, Ixx_it)
                G_inv = G |> pinv
                min_eigenvalue = (G |> eigen).values |> minimum
                min_eigenvalue /= grid .|> length |> prod
            end

            pyramid_contribution = SVector{2}(0.0, 0.0)
            for k = 1:algorithm.iterations
                putative_flow = displacement[j] + pyramid_contribution
                putative_correspondence = point + putative_flow
                if !lies_in(axes(second_img), putative_correspondence)
                    declare_lost!(status, flow, j)
                    break
                end

                grid, offsets, is_truncated_window = get_grid(
                    first_pyramid[i], point, putative_flow,
                    window,
                )
                if is_truncated_window
                    G = compute_spatial_gradient(grid, Iyy_it, Iyx_it, Ixx_it)
                    G_inv = G |> pinv
                    min_eigenvalue = (G |> eigen).values |> minimum
                    min_eigenvalue /= grid .|> length |> prod
                end

                A = view(first_pyramid[i], grid[1], grid[2])
                Ix_window = view(Ix, grid[1], grid[2])
                Iy_window = view(Iy, grid[1], grid[2])

                # TODO Add epsilon termination criteria.
                estimated_flow = compute_flow_vector(
                    putative_correspondence, G_inv, A,
                    Iy_window, Ix_window, offsets, etp,
                )
                pyramid_contribution += estimated_flow

                if is_lost(
                    first_pyramid[i], point + pyramid_contribution,
                    min_eigenvalue, algorithm.eigenvalue_threshold,
                )
                    declare_lost!(status, flow, j)
                    break
                end
            end

            if status[j] && is_lost(displacement[j], 2 * window)
                declare_lost!(status, flow, j)
            end

            if status[j]
                flow[j] = pyramid_contribution
                displacement[j] = 2 * (displacement[j] + flow[j])
            end
        end
    end

    output_flow = 0.5 * displacement
    output_flow, status
end

function declare_lost!(status, flow, j)
    status[j] = false
    flow[j] = SVector{2, Float64}(0.0, 0.0)
end

function construct_pyramids(first_img, second_img, pyramid_levels)
    first_pyramid = gaussian_pyramid(first_img, pyramid_levels, 2, 1.0)
    second_pyramid = gaussian_pyramid(second_img, pyramid_levels, 2, 1.0)
    first_pyramid, second_pyramid
end

function compute_partial_derivatives(Iy, Ix)
    Iyy = imfilter(Iy .* Iy, Kernel.gaussian(1))
    Ixx = imfilter(Ix .* Ix, Kernel.gaussian(1))
    Iyx = imfilter(Iy .* Ix, Kernel.gaussian(1))

    Iyy_integral_table = integral_image(Iyy)
    Ixx_integral_table = integral_image(Ixx)
    Iyx_integral_table = integral_image(Iyx)

    Iyy_integral_table, Ixx_integral_table, Iyx_integral_table
end

function get_pyramid_coordinates(points, i, j)
    px = floor(Int, points[j][1] / 2 ^ (i - 1))
    py = floor(Int, points[j][2] / 2 ^ (i - 1))
    SVector{2}(px ,py)
end

function compute_spatial_gradient(
    grid, Iyy_integral, Iyx_integral, Ixx_integral,
)
    sum_Iyy = boxdiff(Iyy_integral, grid[1], grid[2])
    sum_Ixx = boxdiff(Ixx_integral, grid[1], grid[2])
    sum_Iyx = boxdiff(Iyx_integral, grid[1], grid[2])
    SMatrix{2, 2, Float64}(sum_Iyy, sum_Iyx, sum_Iyx, sum_Ixx)
end

"""
Evaluate Σ(δI .* I_y) and Σ(δI .* I_x), where δI = A .- B

B - is the extrapolated (for subpixel precision) second image pyramid layer.
δI = A - B - is the temporal image derivative at the point [x, y].

b = Σ_y Σ_x [δI * Iy, δI * Ix]
"""
function prepare_linear_system(corresponding_point, A, Iy, Ix, offsets, B)
    P, Q = size(A)

    b = SVector{2, Float64}(0.0, 0.0)
    for q in 1:Q, p in 1:P
        r = corresponding_point[1] + offsets[1][p]
        c = corresponding_point[2] + offsets[2][q]

        δI = A[p, q] - B(r, c)
        b += SVector{2, Float64}(δI * Iy[p, q], δI * Ix[p, q])
    end
    b
end

function compute_flow_vector(
    corresponding_point, G_inv, pyramid_window,
    Iy, Ix, offsets, etp,
)
    b = prepare_linear_system(
        corresponding_point, pyramid_window,
        Iy, Ix,
        offsets, etp,
    )
    G_inv * b
end

function is_lost(
    img::AbstractArray{T, 2}, point::SVector{2, U},
    min_eigenvalue::Float64, eigenvalue_threshold::Float64,
) where {T <: Gray, U <: Union{Int, Float64}}
    !(lies_in(axes(img), point)) && return true

    val = min_eigenvalue
    val < eigenvalue_threshold
end

function is_lost(point::SVector{2, Float64}, window_size::Int)
    point[1] > window_size || point[2] > window_size
end

function lies_in(
    area::Tuple{AbstractRange{Int}, AbstractRange{Int}}, point::SVector{2, T},
) where T <: Union{Int,Float64}
    first(area[1]) ≤ point[1] ≤ last(area[1]) &&
    first(area[2]) ≤ point[2] ≤ last(area[2])
end

function get_grid(
    img::AbstractArray{T, 2},
    point::SVector{2, U}, displacement::SVector{2, Float64},
    window_size::Int,
) where {T <: Gray, U <: Union{Int, Float64}}
    new_point = point + displacement
    allowed_area = map(
        i -> (first(i) + window_size):(last(i) - window_size), axes(img),
    )
    is_truncated_window = (
        !lies_in(allowed_area, new_point) || !lies_in(allowed_area, point)
    )

    first_axis, second_axis = axes(img)

    w_up = floor(Int64, min(window_size, min(
        point[1] - first(first_axis), new_point[1] - first(second_axis),
    )))
    w_down = floor(Int64, min(window_size, min(
        last(first_axis) - point[1], last(second_axis) - new_point[1],
    )))
    w_left = floor(Int64, min(window_size, min(
        point[2] - first(second_axis), new_point[2] - first(second_axis),
    )))
    w_right = floor(Int64, min(window_size, min(
        last(second_axis) - point[2], last(second_axis) - new_point[2],
    )))

    new_grid = (
        (point[1] - w_up):(point[1] + w_down),
        (point[2] - w_left):(point[2] + w_right),
    )
    offsets = (UnitRange(-w_up, w_down), UnitRange(-w_left, w_right))
    new_grid, offsets, is_truncated_window
end
