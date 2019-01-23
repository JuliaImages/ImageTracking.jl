"""
```
visualize_optical_flow_hsv(flow)
```

Return the visualization of an optical flow vector `flow` in hsv color space.

Here, hue encodes angle between optical flow vector and the x-axis in the  image plane,
saturation encodes the ratio between the individual vector magnitudes and the maximum magnitude 
among the whole motion field.

"""
function visualize_optical_flow_hsv(flow::Array{SVector{2, Float64}, 2})
 
    # Convert from (row,column) to (x,y) convention.
    map!(x-> SVector(last(x),first(x)), flow, flow)

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
