"""
    LK(Args...)

A differential method for optical flow estimation developed by Bruce D. Lucas 
and Takeo Kanade. It assumes that the flow is essentially constant in a local 
neighbourhood of the pixel under consideration, and solves the basic optical flow 
equations for all the pixels in that neighbourhood, by the least squares criterion.

The different arguments are:

 -  prev_points       =  Vector of Coordinates for which the flow needs to be found
 -  next_points       =  Vector of Coordinates containing initial estimates of new positions of 
                         input features in next image
 -  window_size       =  Size of the search window at each pyramid level; the total size of the 
                         window used is 2*window_size + 1
 -  max_level         =  0-based maximal pyramid level number; if set to 0, pyramids are not used 
                         (single level), if set to 1, two levels are used, and so on
 -  estimate_flag     =  0 -> Use next_points as initial estimate (Default Value)
                         1 -> Copy prev_points to next_points and use as estimate
 -  term_condition    =  The termination criteria of the iterative search algorithm i.e the number of iterations
 -  min_eigen_thresh  =  The algorithm calculates the minimum eigenvalue of a (2 x 2) normal matrix of optical 
                         flow equations, divided by number of pixels in a window; if this value is less than 
                         min_eigen_thresh, then a corresponding feature is filtered out and its flow is not processed
                         (Default value is 0.1)

## References

B. D. Lucas, & Kanade. "An Interative Image Registration Technique with an Application to Stereo Vision," 
DARPA Image Understanding Workshop, pp 121-130, 1981.

J.-Y. Bouguet, “Pyramidal implementation of the afﬁne lucas kanadefeature tracker description of the 
algorithm,” Intel Corporation, vol. 5,no. 1-10, p. 4, 2001.
"""

struct LK{} <: OpticalFlowAlgo
    prev_points::Array{Coordinate{Int64}, 1}
    next_points::Array{Coordinate{Float64}, 1}
    window_size::Int64
    max_level::Int64
    estimate_flag::Bool
    term_condition::Int64
    min_eigen_thresh::Float64
end

function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, algo::LK{}) where T <: Gray
    first_pyr = gaussian_pyramid(first_img, max_level, 2, 1.0)
    second_pyr = gaussian_pyramid(second_img, max_level, 2, 1.0)

    guess_flow = Array{Coordinate{Float64},1}(size(prev_points)[1])
    final_flow = Array{Coordinate{Float64},1}(size(prev_points)[1])
    status_array = trues(size(prev_points))
    error_array = zeros(Float64, size(prev_points))

    for i = 1:size(prev_points)[1]
        if flag
            guess_flow[i] = Coordinate((next_points[i].x - prev_points[i].x)/2^max_level,(next_points[i].y - prev_points[i].y)/2^max_level)
        else
            guess_flow[i] = Coordinate(0.0,0.0)
        end
        final_flow[i] = Coordinate(0.0,0.0)
    end
    min_eigen = 0.0
    grid_size = 0

    for i = max_level+1:-1:1
        Ix, Iy = imgradients(first_pyr[i], KernelFactors.scharr, Fill(zero(eltype(first_pyr[i]))))

        itp = interpolate(second_pyr[i], BSpline(Quadratic(Flat())), OnGrid())
        etp = extrapolate(itp, zero(eltype(second_pyr[i])))

        Ixx = imfilter(Ix.*Ix, Kernel.gaussian(4))
        Iyy = imfilter(Iy.*Iy, Kernel.gaussian(4))
        Ixy = imfilter(Ix.*Iy, Kernel.gaussian(4))

        for j = 1:size(prev_points)[1]

            if status_array[j]
                point = Coordinate(floor(Int,(prev_points[j].x)/2^i),floor(Int,(prev_points[j].y)/2^i))
                grid = [max(1,point.y-window_size):min(size(first_pyr[i])[1],point.y+window_size), max(1,point.x-window_size):min(size(first_pyr[i])[2],point.x+window_size)]

                G = [sum(Ixx[grid...]) sum(Ixy[grid...])
                     sum(Ixy[grid...]) sum(Iyy[grid...])]

                temp_flow = Coordinate(0.0,0.0)

                for k = 1:term_condition
                    diff_flow = guess_flow[j] + temp_flow

                    grid_1, grid_2, flag = get_grid(first_pyr[i], grid, point, diff_flow, window_size)

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

                    min_eigen = eigmin(G)
                    error_array[j] = min_eigen
                    grid_size = (grid_1[1].stop - grid_1[1].start + 1) * (grid_1[2].stop - grid_1[2].start + 1)

                    if is_lost(first_pyr[i], temp_flow, min_eigen, min_eigen_thresh, grid_size)
                        status_array[j] = false
                        final_flow[j] = Coordinate(0.0,0.0)
                        break
                    end
                end

                if status_array[j]
                    final_flow[j] = temp_flow
                end
            end
        end
        guess_flow = 2*(guess_flow .+ final_flow)
    end
    output_flow = 0.5*guess_flow
    return output_flow, status_array, error_array
end

function in_image(img::AbstractArray{T, 2}, point::Coordinate, min_eigen_thresh::Float64, window::Int64) where T <: Gray
    #TODO: Lower limit for in_image bound
    if point.x < -5 || point.y < -5 || point.x > size(img)[2] || point.y > size(img)[1]
        return false
    end
    return true
end

function is_lost(img::AbstractArray{T, 2}, point::Coordinate, min_eigen::Float64, min_eigen_thresh::Float64, window::Int64) where T <: Gray
    if !(in_image(img, point, min_eigen_thresh, window))
        return true
    else
        val = min_eigen/window
        println(val)
        if val < min_eigen_thresh
            return true
        else
            return false
        end
    end
end

function lies_in(area::Array{UnitRange{Int64},1}, point::Coordinate{T}) where T <: Union{Int64,Float64}
    if area[1].start <= point.y && area[1].stop >= point.y && area[2].start <= point.x && area[2].stop >= point.x
        return true
    else
        return false
    end
end

function get_grid(img::AbstractArray{T, 2}, grid::Array{UnitRange{Int64}, 1}, point::Coordinate, diff_flow::Coordinate, window_size::Int64) where T <: Gray
    allowed_area = [map(i -> 1+window_size:i-window_size, size(img))...]

    if !lies_in(allowed_area, point + diff_flow)
        new_point = point + diff_flow

        grid_1 = Array{UnitRange{Int64}, 1}(2)
        grid_2 = Array{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 1}(2)

        x21 = clamp(new_point.x-window_size, 1, size(img)[2])
        x22 = clamp(new_point.x+window_size, 1, size(img)[2])
        if x21 == 1
            x22 = ceil(x22)
            x11 = convert(Int64, x21)
            x12 = convert(Int64, x22)
        else
            x11 = point.x - window_size
            x12 = point.x + window_size
        end

        if x11 < 1
            x21 += (1 - x11)
            x11 = 1
        end

        y21 = clamp(new_point.y-window_size, 1, size(img)[1])
        y22 = clamp(new_point.y+window_size, 1, size(img)[1])
        if y21 == 1
            y22 = ceil(y22)
            y11 = Int(y21)
            y12 = Int(y22)
        else
            y11 = point.y - window_size
            y12 = point.y + window_size
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
