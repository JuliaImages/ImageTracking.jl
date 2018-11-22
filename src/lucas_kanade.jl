

function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1}, algo::LucasKanade) where T <: Gray
    displacement = Array{SVector{2, Float64}, 1}(undef, length(points))
    for i in eachindex(displacement)
            displacement[i] = SVector{2, Float64}(0.0, 0.0)
    end
    optflow!(first_img, second_img, points, displacement, algo )
end

function optflow!(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, points::Array{SVector{2, Float64}, 1}, displacement::Array{SVector{2, Float64}, 1}, algo::LucasKanade) where T <: Gray

    # Convert (row, col) convention to (x,y) coordinate convention.
    points = map(x -> SVector(last(x), first(x)), points)
    map!(x -> SVector(last(x), first(x)), displacement, displacement)

    # Replace NaN with zero in both images.
    map!(x -> isnan(x) ? zero(x) : x, first_img, first_img)
    map!(x -> isnan(x) ? zero(x) : x, second_img, second_img)

    # Construct Gaussian pyramid for both the images.
    first_pyr = gaussian_pyramid(first_img, algo.max_level, 2, 1.0)
    second_pyr = gaussian_pyramid(second_img, algo.max_level, 2, 1.0)


    flow = Array{SVector{2, Float64}, 1}(undef, length(displacement))
    for i in eachindex(flow)
        flow[i] = SVector{2, Float64}(0.0, 0.0)
    end
    status_array = trues(length(displacement))
    reliability_array = zeros(Float64, length(displacement))

    # Iterating over each level of the pyramids.
    for i = algo.max_level+1:-1:1
        # Find x and y image gradients with scharr kernel (As written in Refernce [2])
        Ix, Iy = imgradients(first_pyr[i], KernelFactors.scharr, Fill(zero(eltype(first_pyr[i]))))

        # # Extrapolate the second image so as to get intensity at non integral points.
        #itp = interpolate(second_pyr[i], BSpline(Quadratic(Flat(Interpolations.OnGrid()))))
        itp = interpolate(second_pyr[i], BSpline(Linear()))
        etp = extrapolate(itp, zero(eltype(second_pyr[i])))

        # Calculate Ixx, Iyy and Ixy taking Gaussian weights for pixels in the grid.
        Ixx = imfilter(Ix.*Ix, Kernel.gaussian(4))
        Iyy = imfilter(Iy.*Iy, Kernel.gaussian(4))
        Ixy = imfilter(Ix.*Iy, Kernel.gaussian(4))

        # Running the algorithm for each input point
        for j = 1:length(displacement)

            # Check if point is lost
            if status_array[j]
                # Position of point in current pyramid level
                px = floor(Int,(points[j][1]) / 2^i)
                py = floor(Int,(points[j][2]) / 2^i)
                point = SVector{2}(px ,py)
                # Bounds for the search window
                if all(x->isa(x, Base.OneTo), axes(first_pyr[i]))
                    rs = max(1,point[1] - algo.window_size):min(axes(first_pyr[i])[1].stop, point[1] + algo.window_size)
                    cs = max(1,point[2] - algo.window_size):min(axes(first_pyr[i])[2].stop, point[2] + algo.window_size)
                    grid = SVector{2}(rs, cs)
                else
                    rs = max(axes(first_pyr[i])[1].start, point[1] - algo.window_size):min(axes(first_pyr[i])[1].stop, point[1] + algo.window_size)
                    cs = max(axes(first_pyr[i])[2].start, point[2] - algo.window_size):min(axes(first_pyr[i])[2].stop, point[2] + algo.window_size)
                    grid = SVector{2}(rs, cs)
                end

                # TODO Replace with integral images for fast lookup of sums.
                sum_Ixx = sum(@inbounds view(Ixx,grid[1], grid[2]))
                sum_Ixy = sum(@inbounds view(Ixy,grid[1], grid[2]))
                sum_Iyy = sum(@inbounds view(Iyy,grid[1], grid[2]))


                # Spatial gradient matrix.
                G = SMatrix{2,2,Float64}(sum_Ixx ,sum_Ixy, sum_Ixy, sum_Iyy)
                current_pyramid_flow = SVector{2}(0.0,0.0)

                for k = 1:algo.iterations
                    # Diff flow between images to calculate the image difference
                    diff_flow = displacement[j] + current_pyramid_flow
                    # Check if the search window is present completely in the image
                    grid_1, offsets, flag = get_grid(first_pyr[i], grid, point, diff_flow, algo.window_size)
                    # Calculate the flow
                    A = view(first_pyr[i], grid_1[1], grid_1[2])
                    I_x = @view Ix[grid_1[1],grid_1[2]]
                    I_y = @view Iy[grid_1[1],grid_1[2]]
                    bx = 0.0
                    by = 0.0
                    P, Q = size(A)
                    # Evaluates sum(δI.*I_x) and sum(δI.*I_y) where δI = A .- B
                    corresponding_point = point + diff_flow
                    @inbounds begin
                        for q = 1:Q
                           for p = 1:P
                                Apq = A[p,q]
                                r = corresponding_point[1] + offsets[1][p]
                                c = corresponding_point[2] + offsets[2][q]
                                Bpq = etp(r, c) # This seems to be a bottleneck
                                bx = bx + (Apq - Bpq)*I_x[p,q]
                                by = by + (Apq - Bpq)*I_y[p,q]
                            end
                        end
                    end
                    bk = SVector{2}(bx, by)

                    if !flag
                        # TODO Replace with integral images for fast lookup of sums.
                        Ixx_view =  @inbounds view(Ixx,grid_1[1], grid_1[2])
                        Ixy_view =  @inbounds view(Ixy,grid_1[1], grid_1[2])
                        Iyy_view =  @inbounds view(Iyy,grid_1[1], grid_1[2])
                        sum_Ixx = sum(Ixx_view)
                        sum_Ixy = sum(Ixy_view)
                        sum_Iyy = sum(Iyy_view)
                        # Spatial gradient matrix.
                        G = SMatrix{2,2,Float64}(sum_Ixx ,sum_Ixy, sum_Ixy, sum_Iyy)
                    end

                    # # Use a custom Moore-Pensore inverse for a 2-by-2 matrix
                    # # because the default  pinv(Array(G)) allocates memory.
                    # # TODO: Add the pinv2x2 method to the StaticArrays package.
                    G_inv = pinv2x2(G)
                    ηk = G_inv*bk
                    current_pyramid_flow = ηk + current_pyramid_flow #slow

                    # Minimum eigenvalue in the matrix is used as the error function.
                    F =  eigen(G)
                    min_eigenvalue = F.values[1] < F.values[2] ? F.values[1] : F.values[2]
                    grid_size = (grid_1[1].stop - grid_1[1].start + 1) * (grid_1[2].stop - grid_1[2].start + 1)
                    reliability_array[j] = min_eigenvalue / grid_size

                    # Check whether point is lost.
                    if is_lost(first_pyr[i], point + current_pyramid_flow, min_eigenvalue, algo.eigenvalue_threshold, grid_size)
                        status_array[j] = false
                        flow[j] = SVector{2}(0.0,0.0)
                        break
                    end
                end

                if status_array[j]
                    flow[j] = current_pyramid_flow
                end

                # Guess for the next level of pyramid.
                displacement[j] = 2*(displacement[j] + flow[j])
                # Declare point lost if very high displacement
                if is_lost(SVector(0.5*displacement[j]), 2*algo.window_size)
                    status_array[j] = false
                    flow[j] = SVector{2}(0.0,0.0)
                end
            end
        end
    end

    # Final output flow
    output_flow = 0.5*displacement
    return output_flow, status_array
end

function in_image(img::AbstractArray{T, 2}, point::SVector{2, U}, window::Int) where {T <: Gray, U <: Union{Int, Float64}}
    if all(x->isa(x, Base.OneTo), axes(img))
        if point[1] < 1 || point[2] < 1 || point[1] > axes(img)[1].stop || point[2] > axes(img)[2].stop
            return false
        end
    else
        if point[1] < axes(img)[1].start || point[2] < axes(img)[2].start || point[1] > axes(img)[1].stop || point[2] > axes(img)[2].stop
            return false
        end
    end
    return true
end

function is_lost(img::AbstractArray{T, 2}, point::SVector{2, U}, min_eigen::Float64, min_eigen_thresh::Float64, window::Int) where {T <: Gray, U <: Union{Int, Float64}}
    if !(in_image(img, point, window))
        return true
    else
        val = min_eigen/window

        if val < min_eigen_thresh
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

function lies_in(area::Tuple{UnitRange{Int},UnitRange{Int}}, point::SVector{2, T}) where T <: Union{Int,Float64}
    if area[1].start <= point[1] && area[1].stop >= point[1] && area[2].start <= point[2] && area[2].stop >= point[2]
        return true
    else
        return false
    end
end

function get_grid(img::AbstractArray{T, 2}, grid::AbstractArray{UnitRange{Int}, 1}, point::SVector{2, U}, displacement::SVector{2, Float64}, window_size::Int) where {T <: Gray, U <: Union{Int, Float64}}
    first_axis, second_axis = axes(img)
    allowed_area = map(i -> first(i) + window_size:last(i) - window_size, axes(img))
    new_point = point + displacement

    if !lies_in(allowed_area, new_point)
        flag = false
    else
        flag = true
    end

    # TODO: Convert these loops to explicit formulae.
    w_up = -1
    while point[1] - w_up > first(first_axis) + 1  && new_point[1] - w_up >  first(first_axis) + 1  && w_up < window_size
        w_up += 1
    end

    w_down = -1
    while point[1] + w_down < last(first_axis) - 1 && new_point[1] + w_down <  last(first_axis) - 1  && w_down < window_size
        w_down += 1
    end

    w_left = -1
    while point[2] - w_left > first(second_axis) + 1 && new_point[2] - w_left >  first(second_axis) + 1 && w_left < window_size
        w_left += 1
    end

    w_right = -1
    while point[2] + w_right < last(second_axis) - 1 && new_point[2] + w_right >  last(second_axis) - 1 && w_right < window_size
        w_right += 1
    end

    grid_1 = (point[1] - w_up:point[1] + w_down, point[2] - w_left:point[2] + w_right)
    offsets = (UnitRange(-w_up, w_down), UnitRange(-w_left, w_right))

    return grid_1, offsets, flag
end
