function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1},  algorithm::LucasKanade) where T <: Gray
    displacement = Array{SVector{2, Float64}, 1}(undef, length(points))
    for i in eachindex(displacement)
        displacement[i] = SVector{2, Float64}(0.0, 0.0)
    end
    optflow!(first_img, second_img, points, displacement,  algorithm)
end

function optflow!(
    first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2},
    points::Array{SVector{2, Float64}, 1},
    displacement::Array{SVector{2, Float64}, 1}, algorithm::LucasKanade,
) where T <: Gray
    # Replace NaN with zero in both images.
    first_img = map(x -> isnan(x) ? zero(x) : x, first_img)
    second_img = map(x -> isnan(x) ? zero(x) : x, second_img)

    # Construct Gaussian pyramids for both the images.
    first_pyramid, second_pyramid = construct_pyramids(
        first_img, second_img, algorithm.pyramid_levels,
    )
    flow, status, reliability = initialise_arrays(length(displacement))

    for i = (algorithm.pyramid_levels + 1):-1:1
        @info "Pyramid level $i"
        # Extrapolate the second image so as to get intensity at non-integral points.
        itp = interpolate(second_pyramid[i], BSpline(Linear()))
        etp = extrapolate(itp, zero(eltype(second_pyramid[i])))
        Iy, Ix = imgradients(
            first_pyramid[i], KernelFactors.scharr,
            Fill(zero(eltype(first_pyramid[i]))),
        )

        # Calculate Ixx, Iyy and Ixy taking Gaussian weights for pixels in the grid.
        # Return integral tables to facilitate fast sums over regions in Ixx, Iyy and Ixy.
        Iyy_integral_table, Ixx_integral_table, Iyx_integral_table = compute_partial_derivatives(Iy, Ix)

        # (Y, X) format.
        inner_bounds = map(
            i -> first(i) + algorithm.window_size:last(i) - algorithm.window_size,
            axes(first_pyramid[i]),
        )
        # Running the algorithm for each input point.
        @info "Points", length(displacement)
        for j = 1:length(displacement)
            !status[j] && continue

            # Position of the point in the current pyramid level.
            point = determine_pyramid_coordinates(points, i, j)
            @info "$j | Point $point"
            # If the window falls inside the first image we pre-compute the
            # spatial gradient matrix because in many instance the window
            # will also be completely contained inside the second image.  If
            # it turns out that the  corresponding window is not totally
            # contained inside the second window then we will recompute the
            # spatial gradient matrix again later.
            if lies_in(inner_bounds, point)
                square_grid = SVector{2}(
                    (point[1] - algorithm.window_size):(point[1] + algorithm.window_size),
                    (point[2] - algorithm.window_size):(point[2] + algorithm.window_size),
                )
                G = compute_spatial_gradient(
                    square_grid,
                    Iyy_integral_table,
                    Iyx_integral_table,
                    Ixx_integral_table,
                )
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
                    algorithm.window_size,
                )
                if is_truncated_window
                    G = compute_spatial_gradient(
                        grid,
                        Iyy_integral_table,
                        Iyx_integral_table,
                        Ixx_integral_table,
                    )
                end
                @info "IT $k | G", G

                A = view(first_pyramid[i], grid[1], grid[2])
                Ix_window = view(Ix, grid[1], grid[2])
                Iy_window = view(Iy, grid[1], grid[2])

                estimated_flow = compute_flow_vector(
                    putative_correspondence, G, A,
                    Iy_window, Ix_window, offsets, etp,
                )
                # TODO Add epsilon termination criteria.
                pyramid_contribution += estimated_flow
                @info "It $k | Point $j | Pyramid contribution $pyramid_contribution | Estimated Flow $estimated_flow"

                # Minimum eigenvalue in the matrix is used as a measure of reliability.
                F = eigen(G)
                min_eigenvalue = F.values[1] < F.values[2] ? F.values[1] : F.values[2]
                grid_size = prod(length.(grid))
                reliability[j] = min_eigenvalue / grid_size

                # If the point is too unreliable then it should be declared lost.
                if is_lost(
                    first_pyramid[i], point + pyramid_contribution,
                    min_eigenvalue, algorithm.eigenvalue_threshold, grid_size,
                )
                    @info "[x] Point $j is too unrealiable."
                    declare_lost!(status, flow, j)
                    break
                end
            end

            # If the optical flow is too large then the point should be declared lost.
            if status[j] && is_lost(displacement[j], 2 * algorithm.window_size)
                @info "[x] Point $j flow is flow too big"
                declare_lost!(status, flow, j)
            end

            if status[j]
                flow[j] = pyramid_contribution
                # A guess for the next level of the pyramid.
                displacement[j] = 2 * (displacement[j] + flow[j])
                @info "$j | Point $point | Displacement $(displacement[j])"
            end
        end
    end

    # Final optical flow.
    output_flow = 0.5 * displacement
    return output_flow, status
end

function declare_lost!(status, flow, j)
    status[j] = false
    flow[j] = SVector{2}(0.0,0.0)
end

function construct_pyramids(first_img, second_img, pyramid_levels)
    first_pyramid = gaussian_pyramid(first_img, pyramid_levels, 2, 1.0)
    second_pyramid = gaussian_pyramid(second_img, pyramid_levels, 2, 1.0)
    return first_pyramid, second_pyramid
end

function initialise_arrays(N::Int)
    flow = Array{SVector{2, Float64}, 1}(undef, N)
    for i in eachindex(flow)
        flow[i] = SVector{2, Float64}(0.0, 0.0)
    end
    status = trues(N)
    reliability = zeros(Float64, N)
    return flow, status, reliability
end

function compute_partial_derivatives(Iy, Ix)
    Iyy = imfilter(Iy .* Iy, Kernel.gaussian(1))
    Ixx = imfilter(Ix .* Ix, Kernel.gaussian(1))
    Iyx = imfilter(Iy .* Ix, Kernel.gaussian(1))

    Iyy_integral_table = integral_image(Iyy)
    Ixx_integral_table = integral_image(Ixx)
    Iyx_integral_table = integral_image(Iyx)
    return Iyy_integral_table, Ixx_integral_table, Iyx_integral_table
end

function determine_pyramid_coordinates(points, i, j)
    # TODO i - 1 is correct? Test with pyramids.
    px = floor(Int, points[j][1] / 2 ^ (i - 1))
    py = floor(Int, points[j][2] / 2 ^ (i - 1))
    point = SVector{2}(px ,py)
end

"""
grid in (Y, X) format.
"""
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
    by, bx = 0.0, 0.0
    P, Q = size(A)

    save("/home/pxl-th/B.png", B)

    for q = 1:Q, p = 1:P
        r = corresponding_point[1] + offsets[1][p]
        c = corresponding_point[2] + offsets[2][q]

        # TODO debug, visualize B
        δI = A[p, q] - B(r, c)
        @info "A $(A[p, q]), B $(B(r, c)), A - B $(δI)"
        by += δI * Iy[p, q]
        bx += δI * Ix[p, q]
    end

    # TODO b vector should decrease with the iterations.
    b = SVector{2}(by, bx)
    @info "B", b
    b
end

function compute_flow_vector(
    corresponding_point, G, pyramid_window,
    Ix, Iy, offsets, etp,
)
    # Solve a linear system of equations in order to determine the flow vector.
    b = prepare_linear_system(
        corresponding_point, pyramid_window,
        Iy, Ix,
        offsets, etp,
    )
    G_inv = pinv(G)
    # G_inv = pinv2x2(G)
    G_inv * b
end

function in_image(
    img::AbstractArray{T, 2}, point::SVector{2, U}, window::Int,
) where {T <: Gray, U <: Union{Int, Float64}}
    first_axis = first(axes(img))
    second_axis = last(axes(img))
    if point[1] < first(first_axis) || point[2] < first(second_axis) || point[1] > last(first_axis) || point[2] > last(second_axis)
        return false
    else
        return true
    end
end

function is_lost(
    img::AbstractArray{T, 2}, point::SVector{2, U}, min_eigenvalue::Float64,
    eigenvalue_threshold::Float64, window::Int,
) where {T <: Gray, U <: Union{Int, Float64}}
    !(in_image(img, point, window)) && return true

    val = min_eigenvalue / window
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
    # @info "Allowed area", allowed_area
    # @info "Point", point, "New point", new_point

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

    # (Y, X) format.
    new_grid = (
        (point[1] - w_up):(point[1] + w_down),
        (point[2] - w_left):(point[2] + w_right),
    )
    offsets = (UnitRange(-w_up, w_down), UnitRange(-w_left, w_right))
    # @info "New grid $new_grid | Offsets $offsets"
    return new_grid, offsets, is_truncated_window
end
