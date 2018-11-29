

function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1},  algorithm::LucasKanade) where T <: Gray
    displacement = Array{SVector{2, Float64}, 1}(undef, length(points))
    for i in eachindex(displacement)
            displacement[i] = SVector{2, Float64}(0.0, 0.0)
    end
    optflow!(first_img, second_img, points, displacement,  algorithm)
end

function optflow!(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1}, displacement::Array{SVector{2, Float64}, 1}, algorithm::LucasKanade) where T <: Gray

    # Convert (row, col) convention to (x,y) coordinate convention.
    points = map(x -> SVector(last(x), first(x)), points)
    map!(x -> SVector(last(x), first(x)), displacement, displacement)

    # Replace NaN with zero in both images.
    first_img = map(x -> isnan(x) ? zero(x) : x, first_img)
    second_img = map(x -> isnan(x) ? zero(x) : x, second_img)

    # Construct Gaussian pyramids for both the images.
    first_pyramid, second_pyramid =  construct_pyramids(first_img, second_img,  algorithm.pyramid_levels)
    flow, status, reliability = initialise_arrays(length(displacement))

    for i = algorithm.pyramid_levels+1:-1:1
        # Extrapolate the second image so as to get intensity at non-integral points.
        itp = interpolate(second_pyramid[i], BSpline(Linear()))
        etp = extrapolate(itp, zero(eltype(second_pyramid[i])))

        Ix, Iy = imgradients(first_pyramid[i], KernelFactors.scharr, Fill(zero(eltype(first_pyramid[i])))) # TODO: Fix row,col vs x,y convention
        # Calculate Ixx, Iyy and Ixy taking Gaussian weights for pixels in the grid.
        # Return integral tables to facilitate fast sums over regions in Ixx, Iyy and Ixy.
        Ixx_integral_table, Ixy_integral_table, Iyy_integral_table =  compute_partial_derivatives(Ix, Iy)

        inner_bounds = map(i -> first(i) +  algorithm.window_size:last(i) -  algorithm.window_size, axes(first_pyramid[i]))
        # Running the algorithm for each input point.
        for j = 1:length(displacement)
            if status[j]
                # Position of the point in the current pyramid level.
                point = determine_pyramid_coordinates(points, i, j)
                # If the window falls inside the first image we pre-compute the
                # spatial gradient matrix because in many instance the window
                # will also be completely contained inside the second image.  If
                # it turns out that the  corresponding window is not totally
                # contained inside the second window then we will recompute the
                # spatial gradient matrix again later.
                if lies_in(inner_bounds, point)
                    square_grid = SVector{2}(point[1] -  algorithm.window_size:point[1] +  algorithm.window_size, point[2] -  algorithm.window_size:point[2] +  algorithm.window_size)
                    G = compute_spatial_gradient(square_grid, Ixx_integral_table, Ixy_integral_table, Iyy_integral_table)
                end

                pyramid_contribution = SVector{2}(0.0,0.0)
                for k = 1:algorithm.iterations
                    putative_flow = displacement[j] + pyramid_contribution
                    putative_correspondence = point + putative_flow
                    if lies_in(axes(second_img), putative_correspondence)
                        grid, offsets, is_truncated_window = get_grid(first_pyramid[i], point, putative_flow, algorithm.window_size)
                        # If the grid is not a square then we cannot reuse our precomputed spatial gradient.
                        if is_truncated_window
                            G = compute_spatial_gradient(grid, Ixx_integral_table, Ixy_integral_table, Iyy_integral_table)
                        end
                        A = @inbounds view(first_pyramid[i], grid[1], grid[2])
                        Ix_window = @inbounds view(Ix, grid[1], grid[2])
                        Iy_window = @inbounds view(Iy, grid[1], grid[2])
                        estimated_flow = compute_flow_vector(point + putative_flow, G, A, Ix_window, Iy_window, grid, offsets, etp)

                        pyramid_contribution += estimated_flow
                        # Minimum eigenvalue in the matrix is used as a measure of reliability.
                        F =  eigen(G)
                        min_eigenvalue = F.values[1] < F.values[2] ? F.values[1] : F.values[2]
                        grid_size = prod(length.(grid))
                        reliability[j] = min_eigenvalue / grid_size

                        # If the point is too unreliable then it should be declared lost.
                        if is_lost(first_pyramid[i], point + pyramid_contribution, min_eigenvalue, algorithm.eigenvalue_threshold, grid_size)
                            declare_lost!(status, flow, j)
                            break
                        end
                    else
                        # In this instance the corresponding point lies outside of the second image so we declare it as lost.
                        declare_lost!(status, flow, j)
                        break
                    end
                end

                # If the optical flow is too large then the point should be declared lost.
                if is_lost(SVector(displacement[j]), 2* algorithm.window_size)
                    declare_lost!(status, flow, j)
                end

                if status[j]
                    flow[j] = pyramid_contribution
                    # A guess for the next level of the pyramid.
                    displacement[j] = 2*(displacement[j] + flow[j])
                end
            end
        end
    end

    # Final optical flow.
    output_flow = 0.5*displacement
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

function compute_partial_derivatives(Ix, Iy)
    Ixx = imfilter(Ix.*Ix, Kernel.gaussian(4))
    Iyy = imfilter(Iy.*Iy, Kernel.gaussian(4))
    Ixy = imfilter(Ix.*Iy, Kernel.gaussian(4))

    Ixx_integral_table = integral_image(Ixx)
    Ixy_integral_table = integral_image(Ixy)
    Iyy_integral_table = integral_image(Iyy)
    return Ixx_integral_table, Ixy_integral_table, Iyy_integral_table
end


function compute_flow_vector(corresponding_point, G, A, Ix, Iy, grid, offsets, etp)
    # Solve a linear system of equations in order to determine the flow vector.
    bk = prepare_linear_system(corresponding_point, A, Ix, Iy, grid, offsets, etp)
    G_inv = pinv2x2(G) #TODO: Add the pinv2x2 method to the StaticArrays package.
    ηk = G_inv*bk
end

function determine_pyramid_coordinates(points, i, j)
    px = floor(Int,(points[j][1]) / 2^i)
    py = floor(Int,(points[j][2]) / 2^i)
    point = SVector{2}(px ,py)
end

function compute_spatial_gradient(grid, Ixx_integral::AbstractArray, Ixy_integral::AbstractArray, Iyy_integral::AbstractArray)
    sum_Ixx = boxdiff(Ixx_integral, grid[1], grid[2])
    sum_Ixy = boxdiff(Ixy_integral, grid[1], grid[2])
    sum_Iyy = boxdiff(Iyy_integral, grid[1], grid[2])
    # Spatial gradient matrix.
    G = SMatrix{2,2,Float64}(sum_Ixx ,sum_Ixy, sum_Ixy, sum_Iyy)
end


function prepare_linear_system(corresponding_point, A, Ix, Iy, grid, offsets, etp)
    bx = 0.0
    by = 0.0
    P, Q = size(A)
    # Evaluates sum(δI.*I_x) and sum(δI.*I_y), where δI = A .- B
    @inbounds begin
        for q = 1:Q
           for p = 1:P
                Apq = A[p,q]
                r = corresponding_point[1] + offsets[1][p]
                c = corresponding_point[2] + offsets[2][q]
                Bpq = etp(r, c) # Subpixel precision
                bx = bx + (Apq - Bpq)*Ix[p,q]
                by = by + (Apq - Bpq)*Iy[p,q]
            end
        end
    end
    b = SVector{2}(bx, by)
end

function in_image(img::AbstractArray{T, 2}, point::SVector{2, U}, window::Int) where {T <: Gray, U <: Union{Int, Float64}}
    first_axis = first(axes(img))
    second_axis = last(axes(img))
    if point[1] < first(first_axis) || point[2] < first(second_axis) || point[1] > last(first_axis) || point[2] > last(second_axis)
        return false
    else
        return true
    end
end

function is_lost(img::AbstractArray{T, 2}, point::SVector{2, U}, min_eigenvalue::Float64, eigenvalue_threshold::Float64, window::Int) where {T <: Gray, U <: Union{Int, Float64}}
    if !(in_image(img, point, window))
        return true
    else
        val = min_eigenvalue/window

        if val < eigenvalue_threshold
            return true
        else
            return false
        end
    end
end

function is_lost(point::SVector{2, Float64}, window_size::Int)
    if point[1] > window_size || point[2] > window_size
        return true
    else
        return false
    end
end

function lies_in(area::Tuple{AbstractRange{Int},AbstractRange{Int}}, point::SVector{2, T}) where T <: Union{Int,Float64}
    if first(area[1]) <= point[1] && last(area[1]) >= point[1] && first(area[2]) <= point[2] && last(area[2]) >= point[2]
        return true
    else
        return false
    end
end

function get_grid(img::AbstractArray{T, 2}, point::SVector{2, U}, displacement::SVector{2, Float64}, window_size::Int) where {T <: Gray, U <: Union{Int, Float64}}
    first_axis, second_axis = axes(img)
    allowed_area = map(i -> first(i) + window_size:last(i) - window_size, axes(img))
    new_point = point + displacement

    if !lies_in(allowed_area, new_point) || !lies_in(allowed_area, point)
        is_truncated_window = true
    else
        is_truncated_window = false
    end

    # TODO: Convert these loops to explicit formulae.
    w_up = 0
    while point[1] - (w_up + 1) > first(first_axis)  && new_point[1] - (w_up + 1) >  first(first_axis)  &&  (w_up + 1) < window_size
        w_up += 1
    end

    w_down = 0
    while point[1] + (w_down + 1)  < last(first_axis)  && new_point[1] + (w_down + 1) <  last(first_axis)  && (w_down + 1) < window_size
        w_down += 1
    end

    w_left = 0
    while point[2] - (w_left + 1) > first(second_axis) && new_point[2] - (w_left + 1) >  first(second_axis) && (w_left + 1) < window_size
        w_left += 1
    end

    w_right = 0
    while point[2] + (w_right + 1) < last(second_axis) && new_point[2] + (w_right + 1) >  last(second_axis) && (w_right + 1) < window_size
        w_right += 1
    end


    grid_1 = (point[1] - w_up:point[1] + w_down, point[2] - w_left:point[2] + w_right)
    offsets = (UnitRange(-w_up, w_down), UnitRange(-w_left, w_right))

    return grid_1, offsets, is_truncated_window
end
