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
``

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
