"""
    LK(Args...)

A differential method for optical flow estimation developed by Bruce D. Lucas
and Takeo Kanade. It assumes that the flow is essentially constant in a local
neighbourhood of the pixel under consideration, and solves the basic optical flow
equations for all the pixels in that neighbourhood, by the least squares criterion.

The different arguments are:

 -  prev_points       =  Vector of SVector{2} for which the flow needs to be found
 -  next_points       =  Vector of SVector{2} containing initial estimates of new positions of
                         input features in next image
 -  window_size       =  Size of the search window at each pyramid level; the total size of the
                         window used is 2*window_size + 1
 -  max_level         =  0-based maximal pyramid level number; if set to 0, pyramids are not used
                         (single level), if set to 1, two levels are used, and so on
 -  estimate_flag     =  true -> Use next_points as initial estimate
                         false -> Copy prev_points to next_points and use as estimate
 -  term_condition    =  The termination criteria of the iterative search algorithm i.e the number of iterations
 -  min_eigen_thresh  =  The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of optical
                         flow equations, divided by number of pixels in a window; if this value is less than
                         min_eigen_thresh, then a corresponding feature is filtered out and its flow is not processed
                         (Default value is 10^-6)

## References

B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision,"
DARPA Image Understanding Workshop, pp 121-130, 1981.

J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas kanadefeature tracker description of the
algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.
"""

struct LK{T <: Int64, F <: Float64, V <: Bool} <: OpticalFlowAlgo
    prev_points::Array{SVector{2, T}, 1}
    next_points::Array{SVector{2, F}, 1}
    window_size::T
    max_level::T
    estimate_flag::V
    term_condition::T
    min_eigen_thresh::F
end

LK(prev_points::Array{SVector{2, T}, 1}, next_points::Array{SVector{2, F}, 1}, window_size::T, max_level::T, estimate_flag::V, term_condition::T) where {T <: Int64, F <: Float64, V <: Bool} = LK{T, F, V}(prev_points, next_points, window_size, max_level, estimate_flag, term_condition, 0.000001)

function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, algo::LK{}) where T <: Gray
    if algo.estimate_flag
        @assert size(prev_points) == size(next_points)
    end

    # Construct gaussian pyramid for both the images
    first_pyr = gaussian_pyramid(first_img, algo.max_level, 2, 1.0)
    second_pyr = gaussian_pyramid(second_img, algo.max_level, 2, 1.0)

    # Initilisation of the output arrays
    guess_flow = Array{MVector{2, Float64}, 1}(length(algo.prev_points))
    final_flow = Array{MVector{2, Float64}, 1}(length(algo.prev_points))
    status_array = trues(size(algo.prev_points))
    error_array = zeros(Float64, size(algo.prev_points))

    for i = 1:length(algo.prev_points)
        if algo.estimate_flag
            guess_flow[i] = MVector{2}((algo.next_points[i][1] - algo.prev_points[i][1])/2^algo.max_level,(algo.next_points[i][2] - algo.prev_points[i][2])/2^algo.max_level)
        else
            guess_flow[i] = MVector{2}(0.0,0.0)
        end
        final_flow[i] = MVector{2}(0.0,0.0)
    end
    min_eigen = 0.0
    grid_size = 0

    # Iterating over each level of the pyramids
    for i = algo.max_level+1:-1:1
        # Find x and y image gradients with scharr kernel (As written in Refernce [2])
        Ix, Iy = imgradients(first_pyr[i], KernelFactors.scharr, Fill(zero(eltype(first_pyr[i]))))

        # Extrapolate the second image so as to get intensity at non integral points
        itp = interpolate(second_pyr[i], BSpline(Quadratic(Flat())), OnGrid())
        etp = extrapolate(itp, zero(eltype(second_pyr[i])))

        # Calculate Ixx, Iyy and Ixy taking Gaussian weights for pixels in the grid
        Ixx = imfilter(Ix.*Ix, Kernel.gaussian(4))
        Iyy = imfilter(Iy.*Iy, Kernel.gaussian(4))
        Ixy = imfilter(Ix.*Iy, Kernel.gaussian(4))

        # Running the algorithm for each input point
        for j = 1:length(algo.prev_points)

            # Check if point is lost
            if status_array[j]
                # Position of point in current pyramid level
                point = MVector{2}(floor(Int,(algo.prev_points[j][1])/2^i),floor(Int,(algo.prev_points[j][2])/2^i))
                # Bounds for the search window
                grid = [max(1,point[1] - algo.window_size):min(size(first_pyr[i])[1], point[1] + algo.window_size), max(1,point[2] - algo.window_size):min(size(first_pyr[i])[2],point[2] + algo.window_size)]

                # Spatial Gradient Matrix
                G = [sum(Ixx[grid...]) sum(Ixy[grid...])
                     sum(Ixy[grid...]) sum(Iyy[grid...])]

                temp_flow = MVector{2}(0.0,0.0)

                # Iterating till terminating condition
                for k = 1:algo.term_condition
                    # Diff flow between images to calculate the image difference
                    diff_flow = guess_flow[j] + temp_flow

                    # Check if the search window is present completely in the image
                    grid_1, grid_2, flag = get_grid(first_pyr[i], grid, SVector{2}(point), diff_flow, algo.window_size)

                    # Calculate the flow
                    δI = first_pyr[i][grid_1...] .- etp[grid_2...]
                    I_x = Ix[grid_1...]
                    I_y = Iy[grid_1...]
                    bk = [sum(δI.*I_x)
                          sum(δI.*I_y)]

                    if !flag
                        G = [sum(Ixx[grid_1...]) sum(Ixy[grid_1...])
                             sum(Ixy[grid_1...]) sum(Iyy[grid_1...])]
                    end

                    G_inv = pinv(G)
                    ηk = G_inv*bk
                    temp_flow = ηk + temp_flow

                    # Minimum eigenvalue in the matrix is used as the error function
                    min_eigen = eigmin(G)
                    grid_size = (grid_1[1].stop - grid_1[1].start + 1) * (grid_1[2].stop - grid_1[2].start + 1)
                    error_array[j] = min_eigen/grid_size

                    # Check whether point is lost
                    if is_lost(first_pyr[i], SVector{2}(point + temp_flow), min_eigen, algo.min_eigen_thresh, grid_size)
                        status_array[j] = false
                        final_flow[j] = MVector{2}(0.0,0.0)
                        break
                    end
                end

                if status_array[j]
                    final_flow[j] = temp_flow
                end
                # Guess for the next level of pyramid
                guess_flow[j] = 2*(guess_flow[j] + final_flow[j])

                # Declare point lost if very high displacement
                if is_lost(SVector(0.5*guess_flow[j]), 2*algo.window_size)
                    status_array[j] = false
                    final_flow[j] = MVector{2}(0.0,0.0)
                end
            end
        end
    end
    # Final output flow
    output_flow = 0.5*guess_flow
    return output_flow, status_array, error_array
end

function in_image(img::AbstractArray{T, 2}, point::SVector{2, U}, window::Int64) where {T <: Gray, U <: Union{Int64, Float64}}
    #TODO: Discuss the lower limit for in_image bound
    if point[1] < 1 || point[2] < 1 || point[1] > size(img)[1] || point[2] > size(img)[2]
        return false
    end
    return true
end

function is_lost(img::AbstractArray{T, 2}, point::SVector{2, U}, min_eigen::Float64, min_eigen_thresh::Float64, window::Int64) where {T <: Gray, U <: Union{Int64, Float64}}
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

function is_lost(point::SVector{2, Float64}, window_size::Int64)
    if point[1] > window_size || point[2] > window_size
        return true
    else
        return false
    end
end

function lies_in(area::Array{UnitRange{Int64},1}, point::SVector{2, T}) where T <: Union{Int64,Float64}
    if area[1].start <= point[1] && area[1].stop >= point[1] && area[2].start <= point[2] && area[2].stop >= point[2]
        return true
    else
        return false
    end
end

function get_grid(img::AbstractArray{T, 2}, grid::Array{UnitRange{Int64}, 1}, point::SVector{2, U}, diff_flow::SVector{2, Float64}, window_size::Int64) where {T <: Gray, U <: Union{Int64, Float64}}
    allowed_area = [map(i -> 1+window_size:i-window_size, size(img))...]

    if !lies_in(allowed_area, point + diff_flow)
        new_point = point + diff_flow

        grid_1 = Array{UnitRange{Int64}, 1}(2)
        grid_2 = Array{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 1}(2)

        x21 = clamp(new_point[2]-window_size, 1, size(img)[2])
        x22 = clamp(new_point[2]+window_size, 1, size(img)[2])
        #TODO: Handle case for upper bound
        if x21 == 1
            x22 = ceil(x22)
            x11 = convert(Int64, x21)
            x12 = convert(Int64, x22)
        else
            x11 = point[2] - window_size
            x12 = point[2] + window_size
        end

        if x11 < 1
            x21 += (1 - x11)
            x11 = 1
        end

        y21 = clamp(new_point[1]-window_size, 1, size(img)[1])
        y22 = clamp(new_point[1]+window_size, 1, size(img)[1])
        if y21 == 1
            y22 = ceil(y22)
            y11 = Int(y21)
            y12 = Int(y22)
        else
            y11 = point[1] - window_size
            y12 = point[1] + window_size
        end

        if y11 < 1
            y21 += (1 - y11)
            y11 = 1
        end

        grid_1 = [y11:y12,x11:x12]
        grid_2 = [y21:y22,x21:x22]
        flag = false
    else
        grid_1 = grid
        grid_2 = grid + diff_flow
        flag = true
    end
    return grid_1, grid_2, flag
end
