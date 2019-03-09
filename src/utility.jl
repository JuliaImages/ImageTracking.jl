"""
```
visualize_flow(ColorBased(), flow)
visualize_flow(ColorBased(), flow; convention="row_col")
visualize_flow(ColorBased(), flow; convention="x_y")
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
The convention parameter is a optional keyword arguement to specify the convention the flow vectors are using.

# Example

Compute the HSV encoded visualization of flow vectors in (row, column) convention.
```julia

using ImageTracking

hsv = visualize_flow(ColorBased(), flow)

imshow(RGB.(hsv))
```

Compute the HSV encoded visualization of flow vectors in (row, column) convention.
```julia

using ImageTracking

hsv = visualize_flow(ColorBased(), flow, convention="row_col")

imshow(RGB.(hsv))
```

Compute the HSV encoded visualization of flow vectors in (x, y) convention.
```julia

using ImageTracking

hsv = visualize_flow(ColorBased(), flow, convention="x_y")

imshow(RGB.(hsv))
``` 

# References
[1] S. Baker, D. Scharstein, JP Lewis, S. Roth, M.J. Black, and R. Szeliski. A database and
evaluation methodology for optical flow.International Journal of Computer Vision, 92(1):1–31, 2011.
"""
function visualize_flow(method::ColorBased, flow::Array{SVector{2, Float64}, 2}; convention="row_col")
 
    if convention == "row_col"
	# Convert from (row,column) to (x,y) convention.
	map!(x-> SVector(last(x),first(x)), flow, flow)
    end	
	
    # Display optical flow as an image, with hue encoding the orientation and
    # saturation encoding the relative magnitude.
    max_norm = maximum(map(norm,flow))
    normalised_flow = map(PolarFromCartesian(),flow / max_norm)
    hsv = zeros(HSV{Float32},size(flow))

    for i in eachindex(flow)
        hsv[i] = HSV((normalised_flow[i].θ + pi) * 180 / pi, normalised_flow[i].r, 1)
    end

    return hsv
    
end

struct Point{T}
    x::T
    y::T
end

function is_flow_correct(p::Point)
    return !isnan(p.x) && !isnan(p.y) && abs(p.x) < 1e9 && abs(p.y) < 1e9
end

"""
```
end_point_error(ground_truth_flow, estimated_flow)
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

result = end_point_error(ground_truth_flow, estimated_flow)

imshow(result)
```

# References
[1] 
"""
function end_point_error(ground_truth_flow::Array{SVector{2, Float64}, 2}, estimated_flow::Array{SVector{2, Float64}, 2})
    result = Array{Float64, 2}(undef, size(estimated_flow)[1], size(estimated_flow)[2])
    for i in 1:size(estimated_flow)[1]
        for j in 1:size(estimated_flow)[2]
            p1 = Point{Float64}(estimated_flow[i, j][1], estimated_flow[i, j][2])
            p2 = Point{Float64}(ground_truth_flow[i, j][1], ground_truth_flow[i, j][2])

            if is_flow_correct(p1) && is_flow_correct(p2) 
                diff = Point{Float64}(p1.x - p2.x, p1.y - p2.y)
                result[i, j] = sqrt(diff.x^2 + diff.y^2)
            else
                result[i, j] = NaN
            end
        end
    end
    return result
end
