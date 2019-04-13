"""
```
visualize_flow(flow, ColorBased(), RasterConvention())
visualize_flow(flow, ColorBased(), CartesianConvention())
```

Returns an image that matches the dimensions of the input matrix and that depicts the 
orientation and magnitude of the optical flow vectors using the HSV color space.

# Details
Hue encodes angle between optical flow vector and the x-axis in the image plane;
saturation encodes the ratio between the individual vector magnitudes and the maximum magnitude 
among the whole motion field, and the values always equal one.

# Arguments
The flow parameter needs to be a two-dimensional arrays of length-2 vectors (of type SVector) 
which represent the displacement of each pixel.
The convention parameter specifies the convention the flow vectors are using.

# Example

Compute the HSV encoded visualization of flow vectors in (row, column) convention.
```julia

using ImageTracking

hsv = visualize_flow(flow, ColorBased(), RasterConvention())

imshow(RGB.(hsv))
```

Compute the HSV encoded visualization of flow vectors in (x, y) convention.
```julia

using ImageTracking

hsv = visualize_flow(flow, ColorBased(), CartesianConvention())

imshow(RGB.(hsv))
``` 

# References
[1] S. Baker, D. Scharstein, JP Lewis, S. Roth, M.J. Black, and R. Szeliski. A database and
evaluation methodology for optical flow.International Journal of Computer Vision, 92(1):1–31, 2011.
"""
function visualize_flow(flow::Array{SVector{2, Float64}, 2}, method::ColorBased, convention::RasterConvention)
	# Convert from (row,column) to (x,y) convention.
	map!(x-> SVector(last(x),first(x)), flow, flow)
	
    # Display optical flow as an image, with hue encoding the orientation and
    # saturation encoding the relative magnitude.
    max_norm = maximum(map(norm, flow))
    normalised_flow = map(PolarFromCartesian(), flow / max_norm)
    hsv = zeros(HSV{Float32},size(flow))

    for i in eachindex(flow)
        hsv[i] = HSV((normalised_flow[i].θ + pi) * 180 / pi, normalised_flow[i].r, 1)
    end

    return hsv
    
end

function visualize_flow(flow::Array{SVector{2, Float64}, 2}, method::ColorBased, convention::CartesianConvention)
    # Display optical flow as an image, with hue encoding the orientation and
    # saturation encoding the relative magnitude.
    max_norm = maximum(map(norm, flow))
    normalised_flow = map(PolarFromCartesian(), flow / max_norm)
    hsv = zeros(HSV{Float32},size(flow))

    for i in eachindex(flow)
        hsv[i] = HSV((normalised_flow[i].θ + pi) * 180 / pi, normalised_flow[i].r, 1)
    end

    return hsv
    
end


function is_flow_known(p::SVector{2, Float64})
    return !isnan(first(p)) && !isnan(last(p)) && abs(first(p)) < 1e9 && abs(last(p)) < 1e9
end


"""
```
read_flow_file(file_path)
```

Returns a 2-Dimensional array that depicts the optical flow.

# Details
Reads simple .flo file format which is used for optical flow evaluation.
The .flo file should store 2-band float image for horizontal (u) and vertical (v) flow components.
Floats are stored in little endian format.

bytes contain -
  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
          (just a sanity check that floats are represented correctly)
  4-7     width as an integer
  8-11    height as an integer
  12-end  data (width*height*2*4 bytes total)
          the float values for u and v, interleaved, in row order, i.e.,
          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...

# Arguements
The file_path parameter needs to be a string, storing the address of .flo file.

# Example

Read flow file from .flo file and visualize the flow in HSV
```julia

using ImageTracking

file_path = "./Example_flow.flo"
flow = read_flow_file(file_path)
result = visualize_flow(flow, ColorBased(), RasterConvention())
imshow(RGB.(result))
```

# References
[1] http://vision.middlebury.edu/flow/data/
"""
# first four bytes to be same in little endian
# check this when reading a .flo file
FLOAT_TAG = 202021.25
function read_flow_file(file_path::String)
    
    io = open(file_path, "r")
    seekstart(io)
    
    tag = read(io, Float32) # reading the initial tag 
    
    if tag != FLOAT_TAG
        print("read_from_file: Problem reading file ", filepath)
        return -1
    end
    
    width  = read(io, Int32)
    height = read(io, Int32)
    
    flow = Array{Array{Float64 ,1}, 2}(undef, height, width)
    for i in 1:height
        for j in 1:width
            if eof(io) == false
                fx = read(io, Float32)
                fy = read(io, Float32)
                flow[i, j] = [fx, fy]
            end
        end
    end
    close(io)
    return convert(Array{SArray{Tuple{2},Float64,1,2},2}, flow)
end


"""
```
evaluate_flow_error(ground_truth_flow, estimated_flow, EndpointError())
```

Returns a 2-Dimensional array that matches the dimensions of the input flow vector and
that depicts the end point error between the estimated flow and the ground truth flow.

# Details
If the estimated flow at a point is (u0, v0) and ground truth flow is (u1, v1), then 
error will be sqrt[(u0 - u1)^2 + (v0 - v1)^2] at that point.

# Arguments
The flow parameters needs to be two-dimensional arrays of length-2 vectors (of type SVector) 
which represent the displacement of each pixel.

# Example

Compute the end point error between two flows.
```julia

using ImageTracking

result = evaluate_flow_error(ground_truth_flow, estimated_flow, EndpointError())

imshow(result)
```

# References
[1] S. Baker, D. Scharstein, JP Lewis, S. Roth, M.J. Black, and R. Szeliski. A database and
evaluation methodology for optical flow.International Journal of Computer Vision, 92(1):1–31, 2011.
"""
function evaluate_flow_error(ground_truth_flow::Array{SVector{2, Float64}, 2}, estimated_flow::Array{SVector{2, Float64}, 2}, error_type::EndpointError)
    result = Array{Float64, 2}(undef, size(estimated_flow))
    for i in 1:size(estimated_flow)[1]
        for j in 1:size(estimated_flow)[2]
            p1 = estimated_flow[i, j]
            p2 = ground_truth_flow[i, j]

            if is_flow_known(p1) && is_flow_known(p2) 
                δ = p1 - p2
                result[i, j] = sqrt(sum(δ.^2))
            else
                result[i, j] = NaN
            end
        end
    end
    
    return result
end


"""
```
evaluate_flow_error(ground_truth_flow, estimated_flow, AngularError())
```

Returns a 2-Dimensional array that matches the dimensions of the input flow vector and
that depicts the angle error between the estimated flow and the ground truth flow.

# Details
If the estimated flow at a point is (u0, v0) and ground truth flow is (u1, v1), it 
calculates the angle between (u0, v0, 1) and (u1, v1, 1) vectors as measure for error. 

# Arguments
The flow parameters needs to be two-dimensional arrays of length-2 vectors (of type SVector) 
which represent the displacement of each pixel.

# Example

Compute the angle error between two flows.
```julia

using ImageTracking

result = evaluate_flow_error(ground_truth_flow, estimated_flow, AngularError())

imshow(result)
```

# References
[1] S. Baker, D. Scharstein, JP Lewis, S. Roth, M.J. Black, and R. Szeliski. A database and
evaluation methodology for optical flow.International Journal of Computer Vision, 92(1):1–31, 2011.
"""
function evaluate_flow_error(ground_truth_flow::Array{SVector{2, Float64}, 2}, estimated_flow::Array{SVector{2, Float64}, 2}, error_type::AngularError)
    result = Array{Float64, 2}(undef, size(estimated_flow))
    for i in 1:size(estimated_flow)[1]
        for j in 1:size(estimated_flow)[2]
            if is_flow_known(estimated_flow[i, j]) && is_flow_known(ground_truth_flow[i, j])
                p1 = append!(convert(Array{Float64, 1}, ground_truth_flow[i, j]), 1.0)
                p2 = append!(convert(Array{Float64, 1}, estimated_flow[i, j]), 1.0)
                cosine = dot(p1, p2)/(norm(p1)*norm(p2))
                if cosine > 1
                    result[i, j] = 0
                else
                    result[i, j] = acos(cosine)
                end
                if result[i, j] <= 1e-5
                    result[i, j] = 0
                end
            else
                result[i, j] = NaN
            end
        end
    end
    return result
end


"""
```
calculate_statistics(error)
```
Returns mean and standard deviation, as float values, RX and AX statistics, as dictionaries, for the error input flow.

# Details
RX and AX are the robustness statistics. RX denotes the percentage of pixels that have
an error measure over X. AX denotes the accuracy of the error measure at the Xth percentile.

# Arguements
The error parameters needs to be two-dimensional arrays of length-2 vectors (of type SVector) 
which represent the error between two flows.

# Example 

Compute the Mean, SD, RX, AX stats of the flow error calculated using endpoint error method.
```julia

using ImageTracking

error = evaluate_flow_error(ground_truth_flow, estimated_flow, EndpointError())
mean, sd, rx, ax = calculate_statistics(error) 
```

Compute the mean, SD, RX, AX stats of the flow error calculated using angula error method.
```julia

using ImageTracking

error = evaluate_flow_error(ground_truth_flow, estimated_flow, AngularError())
mean, sd, rx, ax = calculate_statistics(error) 
```

# References
[1] S. Baker, D. Scharstein, JP Lewis, S. Roth, M.J. Black, and R. Szeliski. A database and
evaluation methodology for optical flow.International Journal of Computer Vision, 92(1):1–31, 2011.
"""
function calculate_statistics(error::Array{Float64, 2})
    R_thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
    A_thresholds = [0.5, 0.75, 0.95]
    
    # mean and standard deviations
    mean_value = mean(skipmissing(error))
    std_value = std(skipmissing(error); mean=mean_value)

    # RX stats
    RX_count = Dict()
    for rx in R_thresholds
        count_above_thresh = count(x->(x > rx), error)
        RX_count[rx] = count_above_thresh / (size(error)[1]*size(error)[2])
    end
    
    # AX stats
    AX_count = Dict()
    hist = imhist(error)
    max_val = maximum(hist[2])
    total_pix = size(error)[1]*size(error)[2]
    pixel_cumulative_count = cumsum(hist[2])
    for ax in A_thresholds
        cutoff = floor(ax*3.0 + 0.5)
        count_below_thresh = count(x->(x < cutoff), pixel_cumulative_count)
        AX_count[ax] = count_below_thresh / (length(hist[1])*max_val)
    end

    return mean_value, std_value, RX_count, AX_count
end

